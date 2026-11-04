"""
SCRAPER NOVELUPDATES — akuisisi data katalog
NovelUpdates berada di balik Cloudflare managed challenge, jadi scraper berbasis
notebook lama tidak lagi jalan. `requests`, `cloudscraper`, impersonasi TLS (`curl_cffi`), sampai
Chrome asli headful di Xvfb semuanya dibalas 403 "Just a moment...".

Yang berubah dari notebook lama
1. **Autentikasi.** Sesi memakai cookie `cf_clearance` dari browser sungguhan
   (lihat README di bawah). Cookie itu terikat pada IP **dan** User-Agent, jadi
   keduanya harus dipasok berpasangan; salah satu tidak cocok -> 403 lagi.

2. **Filter exclude dibuang.** Notebook lama memakai
   `ge=168,324,18,157,851,12,1692,281,560,922`. `ge=` adalah genre-**exclude**
   (pasangannya `gi=` untuk include), jadi query itu sengaja **membuang** Gender
   Bender, Josei, Seinen, Shoujo, Shoujo Ai, Shounen, Shounen Ai, Smut, Yaoi,
   dan Yuri. Itulah sebabnya kesepuluh genre itu nyaris nol di dataset lama —
   `Yuri` dan `Shoujo Ai` benar-benar 0, sisanya 1-29 novel yang lolos karena
   genre di halaman detail berbeda dari yang dipakai filter listing. Sembilan
   di antaranya lalu tersingkir dari model karena di bawah ambang 50 novel,
   padahal di katalog aslinya sama sekali tidak langka.

3. **Endpoint yang benar.** `series-finder` meng-clamp di 100 halaman (2.500
   hasil) — `pg=800` mengembalikan isi halaman ke-100 lagi byte-per-byte.
   `series-ranking` **tidak** meng-clamp: teruji sampai `pg=1200` (30.000 novel)
   dan tiap halaman masih berbeda. Jadi tulang punggungnya `series-ranking`.
   Notebook lama berhenti di 5.000 baris karena `max_page=200`-nya sendiri,
   bukan karena dibatasi situs.

   Urutan kerja: 35 genre langka lebih dulu (`gi=<id>`, tuntas cepat, menjamin
   cakupan), lalu penyapuan katalog penuh. Varian urutan lanjutan hanya
   dijalankan bila varian pertama benar-benar kehabisan halaman — untuk genre
   yang lebih kecil dari kedalaman halaman, semua urutan mengembalikan himpunan
   yang sama, dan menjalankan kelimanya membuang empat penelusuran penuh demi
   nol baris baru.

4. **Idempoten & bisa dilanjutkan.** Setiap baris ditulis inkremental; menjalankan
   ulang akan melewati URL yang sudah terkumpul, jadi scrape berjam-jam boleh
   diputus kapan saja.

Cara memakai
Sekali saja, ambil kredensial sesi dari browser yang bisa membuka situsnya:

    1. Buka https://www.novelupdates.com sampai halamannya tampil normal
    2. F12 -> Application -> Cookies -> https://www.novelupdates.com
       salin nilai cookie `cf_clearance`
    3. F12 -> Console -> ketik `navigator.userAgent`, salin hasilnya
    4. Isikan ke .env:
           NU_CF_CLEARANCE=<nilai cookie>
           NU_USER_AGENT=<user agent persis>

Lalu:

    docker compose exec -T pipeline python scripts/scraper/novelupdates.py listing
    docker compose exec -T pipeline python scripts/scraper/novelupdates.py detail

Catatan etika & kepatuhan: jeda antar-permintaan tidak pernah di bawah
`MIN_DELAY` detik dan ada cooldown berkala — beban ke server tetap jauh di bawah
satu pengunjung aktif. Yang diambil hanya metadata katalog yang tampil publik.
Cookie sesi disimpan di `.env` yang sudah gitignored; jangan pernah di-commit.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString

sys.path.insert(0, "/opt/project")

from src.common.logging_utils import get_logger, log_section  # noqa: E402

logger = get_logger("scraper.novelupdates")

BASE = "https://www.novelupdates.com"

#: Jeda antar-permintaan (detik). Jeda tetap 2,5-5 detik memicu HTTP 429 setelah ~40 halaman,
#: jadi jedanya adaptif; lihat `Session._throttle`.
MIN_DELAY = 6.0
MAX_DELAY = 10.0

#: Batas atas jeda adaptif. Di atas ini scraping tidak lagi masuk akal dan
#: lebih baik berhenti daripada menahan sesi berjam-jam tanpa hasil.
MAX_ADAPTIVE_DELAY = 45.0

#: Faktor pengali saat kena 429, dan peluruhan saat permintaan lancar kembali.
THROTTLE_UP = 1.6
THROTTLE_DOWN = 0.93

#: Jeda tambahan khusus 429 sebelum mencoba lagi.
RATE_LIMIT_BACKOFF = (45.0, 90.0)

#: Cooldown lebih panjang tiap N permintaan, meniru jeda alami pembaca.
COOLDOWN_EVERY = 40
COOLDOWN_RANGE = (30.0, 60.0)

#: `series-finder` meng-clamp di 100 halaman (`pg=800` mengembalikan isi halaman ke-100 lagi); `series-ranking` tidak,
#: sehingga menjadi tulang punggung akuisisi. Per 2026-10-07 `rank=popular` berujung di hal 3.062 (~76.500 seri).
DEEP_MAX_PAGES = 3500
GENRE_MAX_PAGES = 250

#: Genre di atas ambang ini tidak diberi penelusuran bertarget karena penyapuan katalog sudah mencakupnya;
#: tanpa batas ini Romance dan Fantasy menghabiskan ~3.700 permintaan mubazir.
RARE_GENRE_CEILING = 1500

#: Tipe ranking yang TERBUKTI menghasilkan urutan berbeda. `all` dan `alltime`
#: sengaja tidak dipakai: keduanya nilai tidak sah dan diam-diam jatuh ke
#: `week`, sehingga hanya menambah dua query duplikat.
RANK_TYPES: tuple[str, ...] = ("popular", "popmonth", "sixmonths", "month", "week")

#: Taksonomi `series-finder` (atribut `genreid`); `gi=` genre include, `ge=` genre exclude.
#: Notebook lama memakai `ge=168,324,18,157,851,12,1692,281,560,922`, membuang sepuluh genre (Yuri, Shoujo Ai, Seinen, dll.) dari dataset lama.
GENRE_IDS: dict[str, int] = {
    "Action": 8, "Adult": 280, "Adventure": 13, "Comedy": 17, "Drama": 9,
    "Ecchi": 292, "Fantasy": 5, "Gender Bender": 168, "Harem": 3,
    "Historical": 330, "Horror": 343, "Josei": 324, "Martial Arts": 14,
    "Mature": 4, "Mecha": 10, "Mystery": 245, "Psychological": 486,
    "Romance": 15, "School Life": 6, "Sci-fi": 11, "Seinen": 18,
    "Shoujo": 157, "Shoujo Ai": 851, "Shounen": 12, "Shounen Ai": 1692,
    "Slice of Life": 7, "Smut": 281, "Sports": 1357, "Supernatural": 16,
    "Tragedy": 132, "Wuxia": 479, "Xianxia": 480, "Xuanhuan": 3954,
    "Yaoi": 560, "Yuri": 922,
}

#: Satu kelompok kerja: nama, batas halaman, dan daftar varian query.
#: Varian ke-2 dan seterusnya HANYA dijalankan bila varian pertama kehabisan
#: halaman — lihat `run_listing`.
SourceGroup = tuple[str, int, list[tuple[str, str]]]


def build_source_groups(rank_by_rarity: dict[str, int] | None = None) -> list[SourceGroup]:
    """
    Rakit rencana akuisisi: genre langka dulu, lalu penelusuran katalog penuh.

    **Kenapa genre langka didahulukan.** Cookie `cf_clearance` berumur pendek
    dan sesi bisa putus kapan saja, jadi urutan kerja harus menaruh perolehan
    marjinal terbesar di depan. Genre langka juga tuntas cepat — beberapa ratus
    novel, selesai dalam puluhan halaman — sehingga jaminan cakupannya didapat
    lebih dulu, bukan digantungkan pada penelusuran panjang yang mungkin
    terputus.

    **Kenapa `series-ranking`, bukan `series-finder`.** `series-finder`
    meng-clamp di 100 halaman; `series-ranking` tidak (teruji sampai `pg=1200`).
    Jadi kelompok terakhir — `rank=popular` tanpa filter — sanggup menyapu
    seluruh katalog sendirian, terurut dari novel yang paling banyak dibaca.

    **Kenapa varian tambahan bersyarat.** Satu genre yang jumlah novelnya lebih
    kecil dari kedalaman halaman akan mengembalikan himpunan yang SAMA berapa
    pun urutannya. Menjalankan lima tipe ranking untuk genre semacam itu berarti
    empat penelusuran penuh demi nol baris baru — pada laju yang dibatasi 429,
    kira-kira satu jam terbuang per genre. Karena itu varian lanjutan hanya
    dijalankan bila varian pertama benar-benar kehabisan halaman.
    """
    def rank_url(rank: str, genre_filter: str = "") -> str:
        parts = [f"rank={rank}"]
        if genre_filter:
            parts.append(genre_filter)
        return BASE + "/series-ranking/?" + "&".join(parts) + "&pg={page}"

    names = list(GENRE_IDS)
    if rank_by_rarity:
        names = [g for g in names if rank_by_rarity.get(g, 0) < RARE_GENRE_CEILING]
        names.sort(key=lambda g: rank_by_rarity.get(g, 0))
        logger.info("Genre yang diberi pass bertarget: %d dari %d (sisanya sudah "
                    ">= %d novel dan tercakup penyapuan katalog)",
                    len(names), len(GENRE_IDS), RARE_GENRE_CEILING)

    groups: list[SourceGroup] = []
    for name in names:
        gid = GENRE_IDS[name]
        slug = name.lower().replace(" ", "_").replace("-", "_")
        groups.append(
            (
                f"g_{slug}",
                GENRE_MAX_PAGES,
                [(rank, rank_url(rank, f"gi={gid}")) for rank in RANK_TYPES],
            )
        )

    # Penyapuan katalog penuh, ditaruh terakhir: ia paling lama, tetapi juga
    # yang paling toleran diputus — apa pun yang sudah terkumpul tetap terpakai,
    # dan menjalankan ulang melanjutkan dari URL yang belum ada.
    groups.append(
        ("catalog", DEEP_MAX_PAGES, [(rank, rank_url(rank)) for rank in RANK_TYPES])
    )
    return groups

LISTING_COLUMNS = ("title", "summary", "genres", "is_completed", "url", "source")

DETAIL_COLUMNS = (
    "series_id", "title", "url",
    "type", "type_origin", "genres", "tags", "language",
    "authors", "artists", "original_publisher", "english_publisher",
    "year", "status_coo", "chapter_count", "is_completed",
    "licensed", "completely_translated", "release_frequency",
    "rating_score", "rating_votes",
    "activity_week_rank", "activity_month_rank", "activity_all_rank",
    "reading_list_count", "reading_list_month_rank", "reading_list_all_rank",
)


class ChallengeError(RuntimeError):
    """Cloudflare menolak sesi — cookie kedaluwarsa atau UA/IP tidak cocok."""


def pick_impersonation(user_agent: str) -> str:
    """
    Pilih profil TLS yang cocok dengan User-Agent.

    Ini bukan detail kosmetik. Cloudflare mencocokkan TLS/HTTP2 fingerprint
    dengan User-Agent yang diklaim: mengirim UA Firefox lewat fingerprint Chrome
    justru **lebih** mencurigakan daripada UA Chrome yang jujur, dan cookie
    `cf_clearance` yang sah pun ikut ditolak. Versi persis tidak harus sama —
    yang menentukan adalah keluarga browsernya.
    """
    ua = user_agent.lower()
    if "firefox" in ua:
        return "firefox147"
    if "safari" in ua and "chrome" not in ua:
        return "safari17_0"
    return "chrome"


@dataclass
class Session:
    """
    Klien HTTP bercookie sesi browser.

    Memakai `curl_cffi` alih-alih `requests` biasa: Cloudflare memeriksa TLS
    fingerprint di samping cookie, dan `requests` ditolak walaupun cookie-nya
    benar. Profil impersonasinya diselaraskan dengan User-Agent lewat
    `pick_impersonation`.
    """

    clearance: str
    user_agent: str
    n_requests: int = 0
    delay: float = MIN_DELAY

    def __post_init__(self) -> None:
        from curl_cffi import requests as cr

        self.impersonate = pick_impersonation(self.user_agent)
        logger.info("Profil TLS: %s (diselaraskan dengan User-Agent)", self.impersonate)
        self._session = cr.Session(impersonate=self.impersonate)
        self._session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": BASE + "/",
                "Upgrade-Insecure-Requests": "1",
            }
        )
        self._session.cookies.set("cf_clearance", self.clearance, domain=".novelupdates.com")

    def _throttle(self, *, rate_limited: bool) -> None:
        """
        Sesuaikan jeda dasar berdasarkan respons server.

        Jeda tetap tidak cukup: batas laju situs ini tidak diketahui dari luar
        dan berubah menurut beban. Percobaan dengan jeda tetap 2,5-5 detik
        lancar selama ~40 halaman lalu dibalas 429 terus-menerus. Jadi jedanya
        naik tajam saat ditolak dan meluruh perlahan saat lancar — mencari
        sendiri laju tertinggi yang masih diterima, tanpa perlu ditebak.
        """
        if rate_limited:
            self.delay = min(self.delay * THROTTLE_UP, MAX_ADAPTIVE_DELAY)
            logger.warning("   429 -> jeda dasar dinaikkan ke %.1f detik", self.delay)
        else:
            self.delay = max(self.delay * THROTTLE_DOWN, MIN_DELAY)

    def _sleep(self) -> None:
        time.sleep(random.uniform(self.delay, self.delay * (MAX_DELAY / MIN_DELAY)))
        if self.n_requests and self.n_requests % COOLDOWN_EVERY == 0:
            pause = random.uniform(*COOLDOWN_RANGE)
            logger.info("Cooldown %.0f detik setelah %d permintaan (jeda dasar %.1fs)...",
                        pause, self.n_requests, self.delay)
            time.sleep(pause)

    def get(self, url: str, max_try: int = 4) -> str | None:
        """Ambil satu halaman; None bila gagal permanen, raise bila diblokir."""
        for attempt in range(max_try):
            self._sleep()
            self.n_requests += 1
            try:
                # `timeout` berupa pasangan (connect, read): dengan satu angka, permintaan bisa menggantung selama server masih meneteskan byte.
                response = self._session.get(url, timeout=(15, 40))
            except Exception as exc:  # noqa: BLE001 - jaringan apa pun -> coba lagi
                logger.warning("  [%d/%d] error jaringan: %s", attempt + 1, max_try, exc)
                time.sleep(5 * (attempt + 1))
                continue

            if response.status_code == 200:
                if "Just a moment" in response.text:
                    raise ChallengeError(
                        "Cloudflare menyajikan challenge — cookie cf_clearance sudah "
                        "kedaluwarsa. Ambil ulang dari browser lalu perbarui .env."
                    )
                self._throttle(rate_limited=False)
                return response.text

            if response.status_code == 403:
                raise ChallengeError(
                    f"403 dari {url} — cookie/User-Agent tidak cocok dengan sesi browser, "
                    "atau cookie sudah kedaluwarsa. Perbarui NU_CF_CLEARANCE & NU_USER_AGENT."
                )

            if response.status_code == 429:
                # Rate limit: naikkan jeda dasar untuk seluruh sesi lalu tunggu lebih lama dari backoff biasa,
                # atau halaman berikutnya kena 429 lagi.
                self._throttle(rate_limited=True)
                wait = random.uniform(*RATE_LIMIT_BACKOFF) * (attempt + 1)
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = max(wait, float(retry_after))
                logger.warning("  [%d/%d] HTTP 429 — tunggu %.0f detik",
                               attempt + 1, max_try, wait)
                time.sleep(wait)
                continue

            logger.warning("  [%d/%d] HTTP %s", attempt + 1, max_try, response.status_code)
            time.sleep(5 * (attempt + 1))
        return None


def build_session() -> Session:
    """Rakit sesi dari .env, dengan pesan yang actionable bila belum diisi."""
    clearance = os.getenv("NU_CF_CLEARANCE", "").strip()
    user_agent = os.getenv("NU_USER_AGENT", "").strip()
    if not clearance or not user_agent:
        raise SystemExit(
            "NU_CF_CLEARANCE dan NU_USER_AGENT belum diisi di .env.\n"
            "  1. Buka https://www.novelupdates.com di browser sampai tampil normal\n"
            "  2. F12 > Application > Cookies > salin nilai `cf_clearance`\n"
            "  3. F12 > Console > `navigator.userAgent` > salin hasilnya\n"
            "  4. Isikan keduanya ke .env, lalu jalankan ulang perintah ini.\n"
            "Keduanya HARUS berpasangan: cookie terikat pada User-Agent dan IP."
        )
    return Session(clearance=clearance, user_agent=user_agent)


def clean_text(text: str | None) -> str | None:
    if not text:
        return None
    text = re.sub(r"\s+", " ", text)
    text = text.replace("<<less", "").replace("more>>", "")
    return text.strip() or None


def read_existing(path: Path, key: str = "url") -> set[str]:
    """Kunci yang sudah terkumpul, supaya scrape bisa dilanjutkan."""
    if not path.is_file():
        return set()
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            value = (row.get(key) or "").strip()
            if value:
                seen.add(value)
    return seen


class Writer:
    """
    Penulis CSV inkremental.

    Menulis per baris, bukan menumpuk di memori lalu `to_csv` di akhir seperti
    notebook lama: scrape berjam-jam yang mati di tengah tidak boleh kehilangan
    semuanya. `QUOTE_ALL` dipakai karena sinopsis memuat koma dan newline.
    """

    def __init__(self, path: Path, columns: tuple[str, ...]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not path.is_file()
        self._handle = path.open("a", encoding="utf-8-sig", newline="")
        self._writer = csv.DictWriter(
            self._handle, fieldnames=list(columns), quoting=csv.QUOTE_ALL,
            extrasaction="ignore",
        )
        if is_new:
            self._writer.writeheader()
            self._handle.flush()
        self.n = 0

    def write(self, row: dict) -> None:
        self._writer.writerow(row)
        self.n += 1
        if self.n % 25 == 0:
            self._handle.flush()

    def close(self) -> None:
        self._handle.flush()
        self._handle.close()


def extract_summary(item) -> str | None:
    """Sinopsis listing terpecah antara teks langsung dan blok `.testhide`."""
    parts = []
    for content in item.contents:
        if isinstance(content, NavigableString):
            text = content.strip()
            if text:
                parts.append(text)
        elif getattr(content, "attrs", None) and "testhide" in (content.attrs.get("class") or []):
            parts.append(content.get_text(" ", strip=True))
    return clean_text(" ".join(parts))


def parse_listing(html: str, source: str) -> list[dict]:
    """
    Urai satu halaman listing.

    **Genre disaring lewat href, bukan diambil semua `.search_genre a`.**
    Notebook lama mengambil seluruh tautan di blok itu, padahal blok yang sama
    memuat badge status:

        <a class="gennew complete" href=".../novelslisting/?...&sta=">Completed</a>
        <a class="gennew search"   href=".../genre/romance/">Romance</a>

    Akibatnya `"Completed"` ikut tercatat sebagai genre — 66 dari 425 baris pada
    percobaan pertama. Kalau lolos, ia menjadi label ke-34 yang dipelajari model
    Head A padahal ia status penerjemahan, bukan genre.

    Badge itu justru berguna kalau dibaca sebagai dirinya sendiri: ia memberi
    `is_completed` langsung dari halaman listing, tanpa perlu membuka halaman
    detail satu per satu.
    """
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for item in soup.select(".search_body_nu"):
        link = item.select_one(".search_title a")
        if not link:
            continue
        genre_block = item.select(".search_genre a")
        genres = [
            a.get_text(strip=True)
            for a in genre_block
            if "/genre/" in (a.get("href") or "")
        ]
        completed = any("complete" in (a.get("class") or []) for a in genre_block)
        rows.append(
            {
                "title": clean_text(link.get_text()),
                "summary": extract_summary(item),
                "genres": ", ".join(genres),
                "is_completed": int(completed),
                "url": (link.get("href") or "").strip(),
                "source": source,
            }
        )
    return rows


def genre_rarity(listing_path: Path) -> dict[str, int]:
    """
    Hitung frekuensi genre pada data yang SUDAH dimiliki (lama + hasil scrape).

    Dipakai untuk mengurutkan query dari genre paling langka. Kalau berkasnya
    belum ada, kembalikan dict kosong — urutan default (alfabetis) tetap jalan.
    """
    import collections

    counter: collections.Counter = collections.Counter()
    for path in (Path("/opt/project/Dataset/data_novel.csv"), listing_path):
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    for genre in (row.get("genres") or "").split(","):
                        genre = genre.strip()
                        if genre:
                            counter[genre] += 1
        except OSError:
            continue
    return dict(counter)


def _walk(
    session: Session,
    label: str,
    template: str,
    max_pages: int,
    seen: set[str],
    writer: "Writer",
) -> tuple[int, bool]:
    """
    Telusuri satu query sampai habis.

    Kembalikan `(jumlah_baris_baru, kehabisan_halaman)`. Nilai kedua itulah
    yang menentukan apakah varian berikutnya perlu dijalankan: bila query ini
    berhenti karena datanya memang habis, mengurutkannya dengan cara lain hanya
    akan mengembalikan himpunan yang sama.
    """
    previous_keys: frozenset[str] | None = None
    blocked_streak = 0
    new_rows = 0
    exhausted = False

    for page in range(1, max_pages + 1):
        html = session.get(template.format(page=page))
        if html is None:
            logger.warning("   hal %d gagal total, dilewati", page)
            continue

        rows = parse_listing(html, label)
        if not rows:
            # Halaman kosong bisa berarti data habis atau soft-block sementara; beri tiga kesempatan dengan jeda menaik sebelum berhenti.
            blocked_streak += 1
            logger.warning("   hal %d kosong (%d berturut)", page, blocked_streak)
            if blocked_streak >= 3:
                logger.info("   3 halaman kosong berturut — dianggap habis")
                exhausted = True
                break
            time.sleep(20 * blocked_streak)
            continue
        blocked_streak = 0

        # Endpoint `series-finder` meng-clamp: di luar batas ia mengembalikan
        # halaman terakhir berulang-ulang. Itu ujung data, bukan halaman baru.
        keys = frozenset(r["url"] for r in rows)
        if previous_keys is not None and keys == previous_keys:
            logger.info("   hal %d mengulang isi hal %d — ujung data", page, page - 1)
            exhausted = True
            break
        previous_keys = keys

        fresh = [r for r in rows if r["url"] and r["url"] not in seen]
        for row in fresh:
            seen.add(row["url"])
            writer.write(row)
        new_rows += len(fresh)

        if page % 25 == 0:
            logger.info("   hal %4d | +%s baru | total unik %s",
                        page, f"{new_rows:,}", f"{len(seen):,}")

    return new_rows, exhausted


def run_listing(session: Session, out_path: Path, max_pages: int | None, only: list[str]) -> int:
    log_section(logger, "tahap 1 — akuisisi katalog (series-ranking, tanpa ge=)")

    seen = read_existing(out_path)
    logger.info("URL sudah terkumpul sebelumnya: %s", f"{len(seen):,}")

    groups = build_source_groups(genre_rarity(out_path))
    if only:
        groups = [g for g in groups if g[0] in only]
    logger.info("Kelompok kerja: %d | varian lanjutan hanya bila query pertama "
                "kehabisan halaman", len(groups))

    writer = Writer(out_path, LISTING_COLUMNS)
    added = 0
    try:
        for index, (group, group_max, variants) in enumerate(groups, start=1):
            limit = min(group_max, max_pages) if max_pages else group_max
            logger.info("")
            logger.info(">> [%d/%d] %s (maks %d halaman/varian)",
                        index, len(groups), group, limit)

            group_new = 0
            for position, (variant, template) in enumerate(variants):
                label = f"{group}_{variant}"
                logger.info("   -- varian '%s'", variant)
                fresh, exhausted = _walk(session, label, template, limit, seen, writer)
                added += fresh
                group_new += fresh

                if position == 0 and exhausted:
                    # Query pertama sudah menyentuh dasar: koleksinya lebih
                    # kecil daripada kedalaman halaman, jadi urutan lain akan
                    # mengembalikan himpunan yang persis sama.
                    logger.info("   '%s' habis pada varian pertama — %d varian lain "
                                "dilewati (akan mengembalikan data yang sama)",
                                group, len(variants) - 1)
                    break

            logger.info("   selesai '%s': +%s baru | total unik %s",
                        group, f"{group_new:,}", f"{len(seen):,}")
    except ChallengeError as exc:
        logger.error("DIHENTIKAN: %s", exc)
        logger.info("Data yang sudah terkumpul TETAP tersimpan di %s", out_path)
    except KeyboardInterrupt:
        logger.warning("Dihentikan pengguna — data yang terkumpul tetap tersimpan.")
    finally:
        writer.close()

    logger.info("")
    logger.info("Listing selesai: +%s baris baru, total unik %s -> %s",
                f"{added:,}", f"{len(seen):,}", out_path)
    return added


def _list_field(soup, div_id: str) -> str | None:
    tags = soup.select(f"div#{div_id} a")
    if tags:
        return ", ".join(t.get_text(strip=True) for t in tags)
    na = soup.select_one(f"div#{div_id} .seriesna")
    return clean_text(na.get_text()) if na else None


def _text_field(soup, div_id: str) -> str | None:
    tag = soup.find("div", id=div_id)
    return clean_text(tag.get_text(" ", strip=True)) if tag else None


def parse_detail(html: str, url: str) -> dict:
    """Ekstrak 27 kolom skema dari satu halaman seri."""
    soup = BeautifulSoup(html, "html.parser")
    out: dict = {"url": url}

    title = soup.select_one("div.seriestitlenu")
    out["title"] = clean_text(title.get_text()) if title else None

    pid = re.search(r"pid=(\d+)", html)
    out["series_id"] = pid.group(1) if pid else None

    out["type"] = _list_field(soup, "showtype")
    origin = soup.select_one("#showtype span")
    out["type_origin"] = (
        origin.get_text(strip=True).replace("(", "").replace(")", "") if origin else None
    )

    for key, div in (
        ("genres", "seriesgenre"), ("tags", "showtags"), ("language", "showlang"),
        ("authors", "showauthors"), ("artists", "showartists"),
        ("original_publisher", "showopublisher"), ("english_publisher", "showepublisher"),
    ):
        out[key] = _list_field(soup, div)

    out["year"] = _text_field(soup, "edityear")
    status = _text_field(soup, "editstatus")
    out["status_coo"] = status

    # Jumlah bab tertanam di teks status ("182 Chapters (Completed)"). Pola
    # "<angka> chapters" dicoba lebih dulu; angka telanjang hanya cadangan.
    match = re.search(r"([\d,]+)\s*chapters?", status or "", re.IGNORECASE) or re.search(
        r"([\d,]+)", status or ""
    )
    out["chapter_count"] = match.group(1).replace(",", "") if match else None
    out["is_completed"] = bool(re.search(r"completed", status or "", re.IGNORECASE))

    out["licensed"] = _text_field(soup, "showlicensed")
    out["completely_translated"] = _text_field(soup, "showtranslated")

    rating = soup.select_one("span.uvotes")
    if rating:
        text = rating.get_text(strip=True)
        score = re.search(r"\(([\d.]+)\s*/", text)
        votes = re.search(r"([\d,]+)\s*votes", text)
        out["rating_score"] = score.group(1) if score else None
        out["rating_votes"] = votes.group(1).replace(",", "") if votes else None
    else:
        out["rating_score"] = out["rating_votes"] = None

    freq = re.search(r"Release Frequency.*?</h5>\s*([^<]+)", html, re.DOTALL)
    out["release_frequency"] = clean_text(freq.group(1)) if freq else None

    # Situs memotong peringkat aktivitas mingguan dan bulanan di #2000 ("di luar 2000 besar"); `data_novel.csv` lama tanpa plafon ini.
    # Nilai mentah ditulis apa adanya (zona raw tetap raw) dan 2000 diperlakukan sebagai sentinel di Fase 2.
    for key, label in (
        ("activity_week_rank", "Weekly Rank"),
        ("activity_month_rank", "Monthly Rank"),
        ("activity_all_rank", "All Time Rank"),
    ):
        m = re.search(
            rf'{label}:\s*<span class="userrate rank">#([\d,]+)</span>', html, re.I | re.S
        )
        out[key] = m.group(1).replace(",", "") if m else None

    rlist = soup.select_one("b.rlist")
    out["reading_list_count"] = rlist.get_text(strip=True).replace(",", "") if rlist else None

    reading = re.search(
        r'Reading List.*?On\s*<b class="rlist">.*?</b>.*?'
        r'Monthly Rank:\s*<span class="userrate rank">#([\d,]+)</span>.*?'
        r'All Time Rank:\s*<span class="userrate rank">#([\d,]+)</span>',
        html, re.I | re.S,
    )
    out["reading_list_month_rank"] = reading.group(1).replace(",", "") if reading else None
    out["reading_list_all_rank"] = reading.group(2).replace(",", "") if reading else None
    return out


def run_detail(session: Session, listing_path: Path, out_path: Path, limit: int | None) -> int:
    log_section(logger, "tahap 2 — detail per seri (27 kolom)")
    if not listing_path.is_file():
        raise SystemExit(f"Berkas listing belum ada: {listing_path} — jalankan tahap 'listing' dulu.")

    with listing_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [r for r in csv.DictReader(handle)
                if (r.get("url") or "").strip().startswith("http")]

    # Novel dari papan peringkat populer lebih dulu: pada lari yang bisa terputus, merekalah yang punya cukup suara (>= 10).
    # `sorted` stabil, jadi urutan peringkat di dalam kelompok terjaga.
    rows.sort(key=lambda r: 0 if "popular" in (r.get("source") or "") else 1)
    urls = list(dict.fromkeys((r["url"] or "").strip() for r in rows))

    done = read_existing(out_path)
    todo = [u for u in urls if u not in done]
    if limit:
        todo = todo[:limit]

    logger.info("Kandidat %s | sudah ada %s | akan diambil %s",
                f"{len(urls):,}", f"{len(done):,}", f"{len(todo):,}")
    if todo:
        estimate = len(todo) * (MIN_DELAY + MAX_DELAY) / 2 / 3600
        logger.info("Perkiraan durasi: %.1f jam bila lancar (jeda %.1f-%.1f detik). "
                    "Rate limit akan memperpanjangnya — proses ini dirancang untuk "
                    "diputus-sambung.", estimate, MIN_DELAY, MAX_DELAY)

    writer = Writer(out_path, DETAIL_COLUMNS)
    ok = 0
    started = time.time()
    try:
        for index, url in enumerate(todo, start=1):
            html = session.get(url)
            if html is None:
                logger.warning("[%d/%d] GAGAL %s", index, len(todo), url)
                continue
            try:
                writer.write(parse_detail(html, url))
                ok += 1
            except Exception as exc:  # noqa: BLE001 - satu halaman rusak tak boleh menghentikan
                logger.warning("[%d/%d] gagal mengurai %s: %s", index, len(todo), url, exc)
            if index % 25 == 0:
                elapsed = time.time() - started
                rate = index / elapsed
                remaining = (len(todo) - index) / rate / 3600 if rate else 0
                logger.info("[%d/%d] berhasil %s | %.1f dtk/novel | sisa ~%.1f jam "
                            "| jeda dasar %.1fs",
                            index, len(todo), f"{ok:,}", 1 / rate if rate else 0,
                            remaining, session.delay)
    except ChallengeError as exc:
        logger.error("DIHENTIKAN: %s", exc)
        logger.info("%s baris sudah tersimpan; jalankan ulang untuk melanjutkan.", f"{ok:,}")
    except KeyboardInterrupt:
        logger.warning("Dihentikan pengguna — %s baris tersimpan.", f"{ok:,}")
    finally:
        writer.close()

    logger.info("Detail selesai: %s baris baru -> %s", f"{ok:,}", out_path)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Scraper katalog NovelUpdates")
    parser.add_argument("stage", choices=("listing", "detail"), help="Tahap yang dijalankan")
    parser.add_argument("--out-dir", default="/opt/project/Dataset/scraped",
                        help="Direktori keluaran (default: Dataset/scraped)")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="Batasi halaman per varian (default: batas per kelompok)")
    parser.add_argument("--sources", nargs="*", default=[],
                        help="Batasi kelompok, mis. --sources g_yuri catalog")
    parser.add_argument("--limit", type=int, help="Batasi jumlah seri pada tahap detail")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    listing_path = out_dir / "listing.csv"
    detail_path = out_dir / "detail.csv"

    session = build_session()
    logger.info("Sesi siap | UA: %s...", session.user_agent[:60])

    if args.stage == "listing":
        run_listing(session, listing_path, args.max_pages, args.sources)
    else:
        run_detail(session, listing_path, detail_path, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
