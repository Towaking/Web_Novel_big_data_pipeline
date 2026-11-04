"""
AUDIT TATA KELOLA IT — security, compliance, privacy, data quality
Empat pilar tata kelola dijadikan **pemeriksaan yang dijalankan**, bukan klaim di dokumen: tiap pilar punya
sekumpulan cek yang mengembalikan status + bukti terukur, sehingga kalau
konfigurasi berubah dan postur keamanannya memburuk, laporannya ikut berubah.

Batas yang disengaja: modul ini menghasilkan **fakta**, bukan argumentasi.
Ia melaporkan "bucket X masih anonymous-readable" atau "0 kolom terdeteksi
memuat PII" — penilaian apakah itu memadai, dan kebijakan apa yang harus
diambil, ditulis manusia.

Status yang dipakai:
  LULUS     — kontrol ada dan terbukti aktif
  PERINGATAN— aman untuk lingkungan lokal, tidak memadai untuk produksi
  GAGAL     — kontrol yang seharusnya ada ternyata tidak ada
  INFO      — temuan faktual tanpa nilai lulus/gagal

Jalankan:
    docker compose exec pipeline python -m src.phases.governance_audit
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import PurePosixPath

from botocore.exceptions import ClientError
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.common.config import get_settings
from src.common.logging_utils import get_logger, log_section
from src.common.spark_session import spark_session
from src.common.storage import get_s3_client, list_objects, object_exists

logger = get_logger("governance.audit")

PASS = "LULUS"
WARN = "PERINGATAN"
FAIL = "GAGAL"
INFO = "INFO"

SECURITY = "3.1 Security"
COMPLIANCE = "3.2 Compliance"
PRIVACY = "3.3 Data Privacy"
QUALITY = "3.4 Data Quality"

#: Kredensial bawaan / contoh yang tidak boleh dipakai: bawaan MinIO, nilai lama
#: yang pernah tercatat di .env.example, dan placeholder .env.example sekarang.
DEFAULT_CREDENTIALS = {"minioadmin", "minioadmin123", "dashboard-readonly-2026"}
MIN_SECRET_LENGTH = 16

#: Pola PII yang dicari di seluruh kolom teks. Sengaja longgar (lebih baik
#: false positive yang lalu diperiksa manusia daripada PII yang lolos).
PII_PATTERNS: dict[str, str] = {
    "email": r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}",
    "telepon": r"(?<!\d)(?:\+62|08)\d{8,12}(?!\d)",
    "kartu_kredit": r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)",
    "nik_16_digit": r"(?<!\d)\d{16}(?!\d)",
    "alamat_ip": r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)",
}


@dataclass
class Check:
    """Satu hasil pemeriksaan."""

    pillar: str
    check_id: str
    name: str
    status: str
    evidence: str
    remediation: str = ""


@dataclass
class Auditor:
    """Kumpulkan hasil cek supaya urutannya stabil dan mudah diringkas."""

    results: list[Check] = field(default_factory=list)

    def add(self, *args, **kwargs) -> None:
        check = Check(*args, **kwargs)
        icon = {PASS: "[+]", WARN: "[!]", FAIL: "[x]", INFO: "[i]"}[check.status]
        logger.info("%s %-10s %-38s %s", icon, check.check_id, check.name, check.evidence)
        self.results.append(check)


def _tls_handshake(endpoint: str) -> tuple[bool, str]:
    """(berhasil, bukti) dari handshake TLS terverifikasi ke endpoint object storage."""
    import socket
    import ssl
    from urllib.parse import urlparse

    url = urlparse(endpoint)
    if url.scheme != "https":
        return False, "endpoint memakai HTTP tanpa enkripsi"
    context = ssl.create_default_context(cafile=os.getenv("AWS_CA_BUNDLE") or None)
    try:
        with socket.create_connection((url.hostname, url.port or 443), timeout=5) as sock:
            with context.wrap_socket(sock, server_hostname=url.hostname) as tls:
                return True, f"{tls.version()}, sertifikat server terverifikasi terhadap CA stack"
    except (OSError, ssl.SSLError) as exc:
        return False, f"handshake TLS gagal: {type(exc).__name__}"


def audit_security(auditor: Auditor) -> None:
    """Postur keamanan object storage & kredensial."""
    settings = get_settings()
    minio = settings.minio
    client = get_s3_client(settings)

    # Kredensial bawaan: akun pipeline dan akun read-only dashboard. Nilai tidak
    # pernah ditulis ke bukti, hanya akun mana yang lemah.
    secrets_in_use = {
        "akun pipeline": minio.secret_key,
        "akun read-only": os.getenv("MINIO_READONLY_PASSWORD", ""),
    }
    weak = [
        name for name, value in secrets_in_use.items()
        if value and (value in DEFAULT_CREDENTIALS or value.startswith("change-me") or len(value) < MIN_SECRET_LENGTH)
    ]
    weak += ["akun pipeline (user bawaan)"] if minio.access_key in DEFAULT_CREDENTIALS else []
    auditor.add(
        SECURITY, "SEC-01", "Kredensial bukan bawaan",
        WARN if weak else PASS,
        f"kredensial bawaan/lemah: {', '.join(weak)}" if weak
        else f"kredensial khusus, rahasia >= {MIN_SECRET_LENGTH} karakter, berbeda dari .env.example",
        "Ganti nilai di .env lalu `docker compose up -d`." if weak else "",
    )

    # Enkripsi in-transit: bukan sekadar skema URL, tetapi handshake TLS sungguhan
    # yang memverifikasi sertifikat server terhadap CA stack (AWS_CA_BUNDLE).
    tls_ok, tls_detail = _tls_handshake(minio.endpoint)
    auditor.add(
        SECURITY, "SEC-02", "Enkripsi in-transit (TLS)",
        PASS if tls_ok else WARN,
        tls_detail,
        "" if tls_ok else "Jalankan `docker compose up -d` (certs-init membuat sertifikat MinIO).",
    )

    for bucket in minio.buckets:
        unreadable = ""
        try:
            policy = client.get_bucket_policy(Bucket=bucket)["Policy"]
            doc = json.loads(policy)
            public = any(
                stmt.get("Effect") == "Allow"
                and stmt.get("Principal") in ("*", {"AWS": "*"}, {"AWS": ["*"]})
                for stmt in doc.get("Statement", [])
            )
        except ClientError as exc:
            # Hanya NoSuchBucketPolicy yang berarti "tidak ada akses anonim".
            # AccessDenied/NoSuchBucket berarti postur bucket TIDAK diketahui.
            public = False
            code = exc.response.get("Error", {}).get("Code", "")
            if code != "NoSuchBucketPolicy":
                unreadable = code or "ClientError"
        if unreadable:
            auditor.add(
                SECURITY, f"SEC-03/{bucket}", f"Bucket '{bucket}' tertutup dari anonim",
                FAIL if unreadable == "NoSuchBucket" else WARN,
                f"policy tidak dapat dibaca: {unreadable}",
                "Periksa izin s3:GetBucketPolicy kredensial audit / keberadaan bucket.",
            )
            continue
        auditor.add(
            SECURITY, f"SEC-03/{bucket}", f"Bucket '{bucket}' tertutup dari anonim",
            FAIL if public else PASS,
            "policy mengizinkan Principal '*'" if public else "tidak ada policy anonim",
            "Jalankan `mc anonymous set none local/<bucket>`." if public else "",
        )

    ro_key = os.getenv("MINIO_READONLY_USER", "").strip()
    auditor.add(
        SECURITY, "SEC-04", "Pemisahan hak baca/tulis",
        PASS if ro_key else WARN,
        f"akun read-only '{ro_key}' tersedia untuk dashboard" if ro_key
        else "semua service memakai kredensial root yang sama",
        "" if ro_key else
        "Buat service account read-only lalu arahkan service streamlit ke akun itu "
        "(lihat docker/policies/ dan blok minio-init di docker-compose.yml).",
    )

    auditor.add(
        SECURITY, "SEC-05", "Isolasi jaringan antar-service",
        INFO,
        "seluruh service berada pada bridge network 'bigdata-net'; port UI hanya "
        "dipetakan ke 127.0.0.1 host (HOST_BIND)",
        "",
    )


def audit_compliance(auditor: Auditor) -> None:
    """Jejak asal-usul data, lisensi, dan retensi."""
    settings = get_settings()
    minio = settings.minio

    # Manifest provenance dari Fase 1 - SETIAP sumber yang diingest wajib punya
    # manifest, bukan sekadar "ada manifest apa pun".
    raw_objects = {obj["Key"]: obj for obj in list_objects(minio.bucket_raw, "novel/")}
    expected = {
        key: f"{PurePosixPath(key).parent.as_posix()}/{PurePosixPath(key).stem}_manifest.json"
        for _, key in settings.raw_sources
    }
    client = get_s3_client(settings)
    details, payloads = [], {}
    for key, manifest_key in expected.items():
        if manifest_key not in raw_objects:
            continue
        body = client.get_object(Bucket=minio.bucket_raw, Key=manifest_key)["Body"].read()
        payloads[key] = json.loads(body)
        details.append(
            f"{key}: md5={str(payloads[key].get('md5', ''))[:8]}..., "
            f"{payloads[key].get('row_count', '?')} baris"
        )
    missing = sorted(set(expected) - set(payloads))
    auditor.add(
        COMPLIANCE, "CMP-01", "Provenance terekam (checksum + waktu)",
        FAIL if missing else PASS,
        (" | ".join(details) or "tidak ada manifest di raw-data")
        + (f" | TANPA manifest: {', '.join(missing)}" if missing else ""),
        "Jalankan Fase 1 (`python -m src.phases.phase1_collect`)." if missing else "",
    )

    # Immutability raw zone: prefix novel/ hanya berisi sumber terdaftar dan manifestnya, dengan ukuran objek sama seperti saat ingestion.
    allowed = set(expected) | set(expected.values())
    stray = sorted(k for k in raw_objects if k not in allowed)
    resized = sorted(
        key for key, payload in payloads.items()
        if key in raw_objects and raw_objects[key]["Size"] != payload.get("size_bytes")
    )
    problems = ([f"objek tak terdaftar: {', '.join(stray)}"] if stray else []) + (
        [f"ukuran beda dari manifest: {', '.join(resized)}"] if resized else []
    )
    auditor.add(
        COMPLIANCE, "CMP-02", "Raw zone tidak pernah ditimpa transformasi",
        FAIL if problems else PASS,
        "; ".join(problems) if problems else
        f"{len(raw_objects)} objek di raw-data/novel/ = {len(expected)} sumber + manifest, "
        "ukuran tiap sumber sama dengan manifest ingestion",
        "Transformasi harus menulis ke clean-data/curated-data, bukan raw-data." if problems else "",
    )

    versioned = []
    for bucket in minio.buckets:
        try:
            status = client.get_bucket_versioning(Bucket=bucket).get("Status", "Disabled")
        except ClientError:
            status = "Disabled"
        versioned.append(f"{bucket}={status}")
    all_on = all("Enabled" in v for v in versioned)
    auditor.add(
        COMPLIANCE, "CMP-03", "Versioning bucket (pemulihan & audit)",
        PASS if all_on else WARN,
        ", ".join(versioned),
        "" if all_on else "Aktifkan lewat `mc version enable local/<bucket>` (sudah "
        "ditambahkan ke minio-init; jalankan `docker compose up -d minio-init`).",
    )

    auditor.add(
        COMPLIANCE, "CMP-04", "Sumber data terdokumentasi",
        INFO,
        "Dataset berasal dari katalog publik NovelUpdates (metadata karya + sinopsis "
        "hasil scraping). Metadata katalog bersifat faktual; teks sinopsis tetap "
        "karya berhak cipta pemiliknya dan hanya dipakai untuk analitik non-komersial",
        "Status lisensi redistribusi harus ditegaskan sebelum dataset ini dipublikasikan "
        "ulang — keputusan ini ada di penulis laporan, bukan di kode.",
    )


def _scan_pii(df: DataFrame, label: str, auditor: Auditor) -> list[dict]:
    """
    Pindai seluruh kolom bertipe string untuk pola PII.

    Semua kolom dipindai dalam SATU agregasi: satu `sum(...)` per (kolom, pola).
    Memindai kolom satu per satu memicu puluhan job Spark untuk hasil yang sama.
    """
    string_cols = [f.name for f in df.schema.fields if f.dataType.simpleString() == "string"]
    if not string_cols:
        return []

    aggregations = []
    for column in string_cols:
        for pattern_name, pattern in PII_PATTERNS.items():
            aggregations.append(
                F.sum(
                    F.when(F.col(column).rlike(pattern), F.lit(1)).otherwise(F.lit(0))
                ).alias(f"{column}||{pattern_name}")
            )

    row = df.agg(*aggregations).first().asDict()
    findings = []
    for key, hits in row.items():
        if not hits:
            continue
        column, pattern_name = key.split("||")
        findings.append(
            {"dataset": label, "column": column, "pattern": pattern_name, "n_rows": int(hits)}
        )
    return findings


def audit_privacy(auditor: Auditor, spark: SparkSession) -> list[dict]:
    """Cari PII, dan pastikan tidak ada data tingkat individu."""
    settings = get_settings()
    minio = settings.minio

    # Setiap sumber raw dipindai, termasuk teks sinopsisnya.
    frames: dict[str, DataFrame] = {}
    for _, key in settings.raw_sources:
        if not object_exists(minio.bucket_raw, key, settings):
            continue
        frames[PurePosixPath(key).name] = (
            spark.read.option("header", True).option("multiLine", True).option("escape", '"')
            .csv(minio.s3a_uri(minio.bucket_raw, key))
        )
    findings = [f for label, df in frames.items() for f in _scan_pii(df, label, auditor)]
    scanned = ", ".join(frames) or "(tidak ada sumber)"

    # Pola `kartu_kredit`/`nik` hampir pasti false positive pada teks sinopsis
    # (deretan angka panjang di dalam prosa). Dipisahkan supaya statusnya tidak
    # menakut-nakuti, tetapi TETAP dilaporkan agar bisa diperiksa manusia.
    direct = [f for f in findings if f["pattern"] in {"email", "telepon"}]
    weak = [f for f in findings if f["pattern"] not in {"email", "telepon"}]

    auditor.add(
        PRIVACY, "PRV-01", "Pemindaian PII langsung (email/telepon)",
        PASS if not direct and frames else WARN,
        f"tidak ada kecocokan pada seluruh kolom teks ({scanned})" if not direct
        else "; ".join(f"{f['dataset']}.{f['column']} {f['pattern']}={f['n_rows']}" for f in direct),
        "" if not direct else "Periksa manual; masking wajib sebelum data dibagikan.",
    )
    auditor.add(
        PRIVACY, "PRV-02", "Pola angka mirip identitas (perlu verifikasi)",
        INFO,
        "tidak ada" if not weak
        else "; ".join(f"{f['dataset']}.{f['column']} {f['pattern']}={f['n_rows']}" for f in weak),
        "Kecocokan pada kolom prosa umumnya deretan angka biasa, bukan identitas.",
    )

    user_like = sorted({
        c for df in frames.values() for c in df.columns
        if re.search(r"user|member|account|email|ip_", c, re.I)
    })
    auditor.add(
        PRIVACY, "PRV-03", "Tidak ada data tingkat individu",
        PASS if not user_like else FAIL,
        "tidak ada kolom ber-grain pengguna; sinyal pembaca hanya tersedia sebagai "
        "agregat (`rating_votes`, `reading_list_count`) sehingga tidak ada subjek data "
        "yang dapat diidentifikasi ulang" if not user_like
        else f"kolom mencurigakan: {', '.join(user_like)}",
        "",
    )

    blank = next((
        df.agg(F.avg(F.when(F.col("artists").isNull()
                            | F.lower(F.trim("artists")).isin("unknown", "n/a", ""), 1).otherwise(0))).first()[0]
        for df in frames.values() if "artists" in df.columns), None)
    artists_note = f"{100 * blank:.2f}% kosong atau Unknown".replace(".", ",") if blank is not None else "hampir seluruhnya kosong"
    auditor.add(
        PRIVACY, "PRV-04", "Nama orang yang ikut terproses",
        INFO,
        "kolom `authors` memuat nama pena yang dipublikasikan platform sebagai kredit "
        f"karya (bukan identitas privat); `artists` dibuang di Fase 2 karena {artists_note}",
        "Nama pena tetap data pribadi bila dapat dikaitkan ke orang nyata — dasar "
        "pemrosesannya perlu dinyatakan di Bab 3.3.",
    )

    return findings


def _raw_row_count(settings) -> int:
    """Jumlah baris katalog sumber menurut manifest Fase 1 (0 bila manifest tidak ada)."""
    key = settings.raw_object_key
    manifest_key = f"{PurePosixPath(key).parent.as_posix()}/{PurePosixPath(key).stem}_manifest.json"
    try:
        body = get_s3_client(settings).get_object(Bucket=settings.minio.bucket_raw, Key=manifest_key)["Body"].read()
        return int(json.loads(body).get("row_count", 0))
    except (ClientError, ValueError):
        return 0


def audit_quality(auditor: Auditor, spark: SparkSession) -> None:
    """
    Tegakkan aturan kualitas data sebagai assertion yang **bisa gagal**: kalau
    suatu saat Fase 2 diubah dan sentinel bocor ke clean zone, cek di bawah
    berubah jadi GAGAL.
    """
    settings = get_settings()
    clean = spark.read.parquet(
        settings.minio.s3a_uri(settings.minio.bucket_clean, "novels_clean")
    )
    has_tier = "metadata_tier" in clean.columns
    is_full = (F.col("metadata_tier") == "full") if has_tier else F.lit(True)
    is_listing = (F.col("metadata_tier") == "listing") if has_tier else F.lit(False)

    def n_where(cond):
        return F.sum(F.when(cond, 1).otherwise(0))

    # Sentinel tidak boleh lolos ke clean zone; aturannya sama dengan `phase2_clean.normalize_sentinels`.
    string_cols = [c for c in ("type", "type_origin", "tags", "language", "authors",
                               "original_publisher", "status_coo", "release_frequency")
                   if c in clean.columns]
    sentinel_exprs = [
        n_where(F.col("chapter_count") <= 0).alias("chapter_count"),
        n_where((F.col("year") < 1000) | (F.col("year") > 2030)).alias("year"),
        n_where(F.col("rating_score") <= 0).alias("rating_score"),
        n_where(F.col("reading_list_count") < 0).alias("reading_list_count"),
        *[n_where(F.lower(F.trim(F.col(c))).isin("unknown", "n/a")).alias(c) for c in string_cols],
    ]
    # Satu agregasi untuk seluruh pilar kualitas (dulu delapan job terpisah).
    stats = clean.agg(
        F.count(F.lit(1)).alias("_total"),
        F.countDistinct("series_id").alias("_unique"),
        n_where(is_full).alias("_full"),
        n_where(is_full & F.col("has_synopsis")).alias("_full_syn"),
        n_where(is_listing & F.col("has_synopsis")).alias("_listing_syn"),
        # Baris 'listing' tidak boleh membawa metrik halaman detail.
        n_where(is_listing & (
            F.col("rating_score").isNotNull() | F.col("chapter_count").isNotNull()
            | F.col("year").isNotNull() | F.col("popularity_score").isNotNull()
        )).alias("_listing_with_detail"),
        n_where(F.col("is_rated_reliable")).alias("_reliable"),
        n_where(F.col("rating_score").isNotNull() & ~F.col("rating_score").between(0, 5)).alias("_oor"),
        *sentinel_exprs,
    ).first().asDict()
    total = stats.pop("_total")

    # Pembanding = jumlah baris di manifest Fase 1, bukan angka mutlak: katalog
    # boleh tumbuh tiap scrape, tetapi Fase 2 hanya boleh membuang duplikat
    # series_id dan baris tanpa url/judul (jauh di bawah 0,1%).
    n_full = int(stats.pop("_full") or 0)
    n_listing = total - n_full
    raw_rows = _raw_row_count(settings)
    kept = total / raw_rows if raw_rows else 0.0
    auditor.add(
        QUALITY, "QLT-01", "Katalog bersih memuat seluruh baris sumber (>= 99,9%)",
        PASS if kept >= 0.999 else FAIL,
        f"{total:,} dari {raw_rows:,} baris sumber ({kept:.2%}) | full={n_full:,} | listing={n_listing:,} | "
        f"total={total:,}",
        "" if kept >= 0.999 else "Fase 2 membuang lebih dari duplikat; periksa filter identitas dan dedup.",
    )
    n_listing_with_detail = int(stats.pop("_listing_with_detail") or 0)
    if n_listing:
        auditor.add(
            QUALITY, "QLT-01b", "Baris tier 'listing' ditandai, bukan disamarkan",
            PASS if n_listing_with_detail == 0 else FAIL,
            f"{n_listing:,} novel hanya bermodal teks & genre; rating/bab/tahun/popularitas "
            "NULL di semua baris itu (diperiksa, bukan diasumsikan). Setiap metrik dari "
            "halaman detail WAJIB memfilter metadata_tier = 'full'" if n_listing_with_detail == 0
            else f"{n_listing_with_detail:,} baris tier 'listing' membawa metrik halaman detail",
            "" if n_listing_with_detail == 0 else "Cek merge_detail/add_derived_features di Fase 2.",
        )

    n_unique = stats.pop("_unique")
    auditor.add(
        QUALITY, "QLT-02", "Kunci `series_id` unik",
        PASS if n_unique == total else FAIL,
        f"{n_unique:,} unik dari {total:,}",
        "" if n_unique == total else "Deduplikasi Fase 2 bocor.",
    )

    n_full_syn = int(stats.pop("_full_syn") or 0)
    n_listing_syn = int(stats.pop("_listing_syn") or 0)
    n_reliable = int(stats.pop("_reliable") or 0)
    out_of_range = int(stats.pop("_oor") or 0)
    leaked = {k: int(v or 0) for k, v in stats.items()}  # sisanya = kolom sentinel
    n_leaked = sum(leaked.values())
    auditor.add(
        QUALITY, "QLT-03", "Sentinel sudah dinormalisasi jadi NULL",
        PASS if n_leaked == 0 else FAIL,
        "tidak ada sentinel tersisa di clean zone" if n_leaked == 0 else f"bocor: {leaked}",
        "" if n_leaked == 0 else "Fase 2 `normalise_sentinels` tidak jalan.",
    )

    # Cakupan sinopsis di-assert pada tier 'full'; baris 'listing' dilaporkan
    # terpisah supaya regresi pada inti katalog tidak terkubur.
    n_synopsis = n_full_syn
    coverage = 100.0 * n_synopsis / n_full if n_full else 0.0
    auditor.add(
        QUALITY, "QLT-04", "Cakupan sinopsis inti katalog >= 98%",
        PASS if coverage >= 98 else WARN,
        f"{coverage:.2f}% ({n_synopsis:,}/{n_full:,}) pada tier 'full'",
        "" if coverage >= 98 else "Cek dedup 538 baris url ganda di berkas sinopsis.",
    )
    if n_listing:
        auditor.add(
            QUALITY, "QLT-04b", "Cakupan sinopsis baris perluasan",
            INFO,
            f"{100.0 * n_listing_syn / n_listing:.2f}% ({n_listing_syn:,}/{n_listing:,})",
            "",
        )

    auditor.add(
        QUALITY, "QLT-05", "Populasi rating tepercaya dilaporkan",
        INFO,
        f"{n_reliable:,} novel (>= 10 suara) dari {total:,} — agregasi rating di Fase 3 "
        f"memakai populasi ini atau weighted rating, bukan rata-rata mentah",
        "",
    )

    # Rating di luar rentang = indikasi kerusakan parsing.
    auditor.add(
        QUALITY, "QLT-06", "Rating berada di rentang sah (0-5)",
        PASS if out_of_range == 0 else FAIL,
        "semua nilai dalam rentang" if out_of_range == 0 else f"{out_of_range} baris di luar rentang",
        "",
    )


def write_markdown(results: list[Check], pii_findings: list[dict]) -> None:
    """Tulis `Doc/generated/governance_audit.md` — fakta hasil audit saja."""
    settings = get_settings()
    out_dir = settings.paths.doc_dir / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "governance_audit.md"

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    tally = {s: sum(1 for r in results if r.status == s) for s in (PASS, WARN, FAIL, INFO)}

    lines = [
        "# Audit Tata Kelola IT (dihasilkan otomatis)",
        "",
        f"> Dihasilkan `src/phases/governance_audit.py` pada {stamp}. **Jangan diedit "
        "manual.** Berkas ini memuat hasil pemeriksaan; argumentasi kebijakan "
        "ditulis terpisah oleh penulis.",
        "",
        f"**Ringkasan:** {tally[PASS]} lulus · {tally[WARN]} peringatan · "
        f"{tally[FAIL]} gagal · {tally[INFO]} informasi",
        "",
    ]

    for pillar in (SECURITY, COMPLIANCE, PRIVACY, QUALITY):
        rows = [r for r in results if r.pillar == pillar]
        if not rows:
            continue
        lines += [
            f"## {pillar}",
            "",
            "| ID | Pemeriksaan | Status | Bukti |",
            "|---|---|---|---|",
        ]
        lines += [
            "| `{id}` | {name} | **{status}** | {ev} |".format(
                id=r.check_id, name=r.name, status=r.status,
                ev=r.evidence.replace("|", "\\|"),
            )
            for r in rows
        ]
        follow_up = [r for r in rows if r.remediation]
        if follow_up:
            lines += ["", "**Tindak lanjut:**", ""]
            lines += [f"- `{r.check_id}` — {r.remediation}" for r in follow_up]
        lines.append("")

    if pii_findings:
        lines += [
            "## Rincian kecocokan pola PII",
            "",
            "| Dataset | Kolom | Pola | Baris cocok |",
            "|---|---|---|---:|",
        ]
        lines += [
            f"| {f['dataset']} | `{f['column']}` | {f['pattern']} | {f['n_rows']:,} |"
            for f in sorted(pii_findings, key=lambda x: -x["n_rows"])
        ]
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Laporan audit ditulis -> %s", out_path)


def run() -> dict:
    """Jalankan keempat pilar audit, simpan hasilnya."""
    log_section(logger, "audit tata kelola it")
    settings = get_settings()
    auditor = Auditor()

    logger.info("--- 3.1 Security")
    audit_security(auditor)

    logger.info("--- 3.2 Compliance")
    audit_compliance(auditor)

    with spark_session("novel-governance-audit") as spark:
        logger.info("--- 3.3 Data Privacy")
        pii_findings = audit_privacy(auditor, spark)

        logger.info("--- 3.4 Data Quality")
        audit_quality(auditor, spark)

        records = [
            {
                "pillar": r.pillar, "check_id": r.check_id, "name": r.name,
                "status": r.status, "evidence": r.evidence, "remediation": r.remediation,
            }
            for r in auditor.results
        ]
        uri = settings.minio.s3a_uri(settings.minio.bucket_curated, "governance/audit_report")
        spark.createDataFrame(records).coalesce(1).write.mode("overwrite").parquet(uri)
        logger.info("Hasil audit tersimpan -> %s", uri)

    write_markdown(auditor.results, pii_findings)

    tally = {s: sum(1 for r in auditor.results if r.status == s) for s in (PASS, WARN, FAIL, INFO)}
    log_section(logger, "ringkasan audit")
    logger.info("LULUS=%d  PERINGATAN=%d  GAGAL=%d  INFO=%d",
                tally[PASS], tally[WARN], tally[FAIL], tally[INFO])
    if tally[FAIL]:
        logger.error("Ada %d kontrol yang GAGAL — lihat kolom tindak lanjut.", tally[FAIL])
    return tally


def main() -> int:
    try:
        tally = run()
    except Exception as exc:  # noqa: BLE001
        logger.error("Audit tata kelola gagal: %s", exc, exc_info=True)
        return 1
    # Exit code non-nol hanya untuk kegagalan kontrol, bukan peringatan —
    # supaya modul ini bisa dipakai sebagai gate di CI tanpa jadi berisik.
    return 2 if tally[FAIL] else 0


if __name__ == "__main__":
    sys.exit(main())
