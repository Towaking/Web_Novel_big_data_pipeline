# NovelLens: pipeline Big Data untuk katalog web novel

Pipeline end-to-end yang mengumpulkan katalog web novel dari NovelUpdates, membersihkan dan memodelkannya, melatih model machine learning, menganalisis graf, lalu menyajikannya di dashboard Streamlit. Seluruh stack berjalan lokal di Docker.

| Lapisan                   | Teknologi                                                                                               |
| ------------------------- | ------------------------------------------------------------------------------------------------------- |
| Data lake                 | MinIO (S3 API, HTTPS, versioning), tiga zona: `raw-data`, `clean-data`, `curated-data`            |
| Penyimpanan terdistribusi | HDFS (1 NameNode, 2 DataNode, blok 4 MB, replikasi 2)                                                   |
| Pemrosesan                | Apache Spark 3.5 (batch, SQL, Structured Streaming, MLlib), Hadoop MapReduce di YARN                    |
| Model data                | Star schema dengan bridge table, property graph                                                         |
| Machine learning          | scikit-learn (Logistic Regression, Linear SVM, Random Forest, ensemble), Spark MLlib sebagai pembanding |
| Graf                      | NetworkX (PageRank, Louvain, homofili)                                                                  |
| Penyajian                 | Streamlit, 15 halaman                                                                                   |

## Isi repositori

```
docker-compose.yml     14 service: MinIO, Spark, Hadoop, pipeline, dashboard, notebook
Makefile               pintasan perintah (opsional, lihat bagian Perintah)
.env.example           contoh konfigurasi, salin menjadi .env
docker/                Dockerfile, requirements.txt, konfigurasi Hadoop, pembuat sertifikat TLS
Dataset/
  novel_catalogue.csv  katalog gabungan 39.670 baris, satu-satunya masukan pipeline
  scraped/             hasil scraping halaman daftar dan halaman detail (snapshot 7 Oktober 2026)
  archive/             dua berkas CSV awal
src/
  common/              konfigurasi, sesi Spark, storage, registri model, pra-pemrosesan teks
  phases/              fase 1 sampai 5, model data, graf, audit, streaming
  mapreduce/           mapper dan reducer Hadoop Streaming (Python murni)
Model/
  pipeline/            fitur, model, evaluasi, protokol pelatihan Head A sampai D
  train.py             melatih ulang model tanpa menjalankan seluruh pipeline
scripts/
  run_pipeline.py      orkestrator (retry, resume)
  scraper/             scraper NovelUpdates dan penyatu katalog
  hdfs/ mapreduce/     pemuatan HDFS, job MapReduce, verifikasi silang dengan Spark
  streaming/           simulasi kedatangan batch baru
  verify_env.py        uji lingkungan sebelum pipeline dijalankan
  verify_app.py        uji dashboard tanpa peramban
app/                   dashboard Streamlit
```

Keluaran pipeline (tidak ikut repositori, dibuat saat dijalankan): data di volume MinIO, `Dataset/variety/`, `Model/artifacts/`, `Doc/generated/`, `logs/`.

## Prasyarat

- Docker Desktop (atau Docker Engine) dengan Docker Compose v2.
- Alokasi memori ke Docker yang cukup. Konfigurasi bawaan memakai worker Spark 3 GB dan driver 2 GB, di luar MinIO, HDFS, YARN, dan dashboard. Sebagai perkiraan, sediakan 12 GB atau lebih.
- Ruang disk beberapa GB untuk image dan volume.
- `make` bersifat opsional. Di Windows pakai Git Bash atau WSL, atau jalankan perintah `docker compose` yang tertera di tabel bagian Perintah.

## Replikasi langkah demi langkah

### 1. Konfigurasi

```bash
cp .env.example .env
```

Ganti `MINIO_ROOT_PASSWORD` dan `MINIO_READONLY_PASSWORD` dengan kata sandi acak minimal 16 karakter. Audit tata kelola menandai kredensial bawaan sebagai peringatan. Port dan memori Spark bisa disesuaikan di berkas yang sama. `HOST_BIND=127.0.0.1` membatasi semua antarmuka ke mesin ini.

### 2. Bangun image dan jalankan stack

```bash
docker compose build
docker compose up -d
docker compose ps
```

Service `certs-init` membuat CA lokal dan sertifikat MinIO, lalu `minio-init` membuat tiga bucket, versioning, aturan lifecycle, dan akun baca-saja untuk dashboard. Build pertama mengunduh jar konektor S3A (sekitar 200 MB) dan memerlukan koneksi yang stabil.

### 3. Uji lingkungan

```bash
docker compose exec pipeline python scripts/verify_env.py
```

Memeriksa MinIO dan ketiga bucket, berkas dataset, job Spark terdistribusi, dan round-trip Parquet lewat S3A. Lanjut hanya bila semua lulus.

### 4. Jalankan pipeline Fase 1 sampai 4 dan lapisan tata kelola

```bash
docker compose exec pipeline python scripts/run_pipeline.py
```

Urutannya: Fase 1 (unggah ke zona raw dengan manifest MD5), Fase 2 (pembersihan), Fase 3 (analitik Spark SQL), Fase 4 (pelatihan Head A sampai D), pemisahan dataset per format ke `Dataset/variety/`, inventaris elemen data, model data (star schema dan graf), analisis graf, dan audit tata kelola. Pipeline berhenti di langkah pertama yang gagal. Opsi: `--from N`, `--to N`, `--only N`, `--retry N`, `--resume`, `--force`, `--skip-governance`, `--only-governance`.

### 5. Fase 5 dan graf kemiripan

```bash
docker compose exec pipeline python -m src.phases.phase5_mllib
docker compose exec pipeline python -m src.phases.similarity_graph
```

`phase5_mllib` membandingkan scikit-learn dan Spark MLlib pada protokol validasi yang sama (waktu latih, log loss, kurva objective). `similarity_graph` membangun graf kNN 10 tetangga di ruang SVD, lalu menghitung komponen, PageRank, komunitas Louvain, dan homofili per genre.

### 6. HDFS dan MapReduce

```bash
docker compose exec -T hadoop-client bash scripts/hdfs/load_hdfs.sh
docker compose exec -T hadoop-client bash scripts/mapreduce/run_jobs.sh
docker compose exec pipeline python scripts/mapreduce/verify_mapreduce.py
```

Langkah pertama memuat enam berkas hasil pemisahan format ke HDFS (blok 4 MB, replikasi 2) dan menjalankan `fsck`. Langkah kedua menjalankan empat job Hadoop Streaming (wordcount, genre_stats, origin_stats, genre_rating). Langkah ketiga mencocokkan hasil MapReduce baris demi baris dengan Spark SQL. Job bisa dicoba tanpa Hadoop lewat `bash scripts/mapreduce/local_smoke.sh`.

### 7. Ingest inkremental (opsional)

```bash
docker compose exec pipeline python scripts/streaming/simulate_landing.py --rows 500
docker compose exec pipeline python -m src.phases.streaming_ingest
docker compose exec pipeline python scripts/streaming/simulate_landing.py --offset 500 --rows 500
docker compose exec pipeline python -m src.phases.streaming_ingest
```

Batch kedua hanya memproses berkas baru karena checkpoint disimpan di MinIO. `python -m src.phases.streaming_ingest --reset` mengulang dari nol.

### 8. Audit dan dashboard

```bash
docker compose exec pipeline python -m src.phases.bigdata_audit
docker compose restart streamlit
docker compose exec streamlit python scripts/verify_app.py
```

Dashboard ada di http://localhost:8501. Restart `streamlit` diperlukan setiap kali artefak model baru ditulis.

## Hasil yang diharapkan

Angka berikut berasal dari snapshot 7 Oktober 2026. Hasil pelatihan bisa bergeser tipis antar mesin dan versi pustaka.

| Tahap                       | Hasil                                                                                         |
| --------------------------- | --------------------------------------------------------------------------------------------- |
| Katalog setelah pembersihan | 39.668 judul (39.652 tier `full`, 16 tier `listing`)                                      |
| Populasi rating tepercaya   | 23.256 judul dengan minimal 10 suara, rata-rata global C = 3,6519                             |
| Model data                  | 67.799 simpul dan 858.901 sisi pada property graph                                            |
| Head A (genre, 35 label)    | ensemble SVM + Random Forest, F1 makro 0,6121, Precision@1 0,904                              |
| Head D (rating)             | MAE 0,3250 dibanding baseline 0,4151                                                          |
| Pembanding MLlib            | F1 makro selisih kurang dari 0,004, waktu latih 18 sampai 43 kali lebih lama pada satu worker |
| Graf kemiripan              | 39.660 simpul, 314.786 sisi, 27 komunitas, modularity 0,7192                                  |
| MapReduce                   | 22 dari 22 task map data-local, empat job identik dengan Spark SQL                            |
| Audit tata kelola           | 23 pemeriksaan: 17 lulus, 6 informasi, 0 gagal                                                |

## Perintah

Semua target `make` memanggil `docker compose exec` di container `pipeline` kecuali disebut lain.

| `make`                                                                  | Fungsi                                            |
| ------------------------------------------------------------------------- | ------------------------------------------------- |
| `build`, `up`, `down`, `ps`, `logs`                             | kelola stack                                      |
| `verify`                                                                | uji lingkungan                                    |
| `pipeline`                                                              | Fase 1 sampai 4 dan tata kelola                   |
| `phase1` sampai `phase5`                                              | satu fase saja                                    |
| `similarity-graph`                                                      | graf kemiripan                                    |
| `catalog`, `datamodel`, `graph-analysis`, `audit`, `variety`    | langkah tata kelola satu per satu                 |
| `hdfs-load`, `mapreduce`, `mapreduce-verify`, `mapreduce-local`   | HDFS dan MapReduce                                |
| `streaming-simulate`, `streaming-run`, `streaming-reset`            | ingest inkremental                                |
| `bigdata-audit`                                                         | audit konsep Big Data                             |
| `consolidate`, `scrape-listing`, `scrape-detail`, `scrape-status` | akuisisi data                                     |
| `clean`, `nuke`                                                       | hentikan stack, `nuke` juga menghapus data lake |

Melatih ulang model tanpa menjalankan pipeline: `docker compose exec pipeline python Model/train.py` (semua head), `--only genre` (hanya Head A), atau `--models svm rf ensemble` (arsitektur tertentu).

## Data

`Dataset/novel_catalogue.csv` adalah satu-satunya masukan pipeline. Berkas itu dibangun dari empat sumber: `archive/data_novel.csv`, `archive/novel_dataset_scrapped.csv`, `scraped/listing.csv`, dan `scraped/detail.csv`. Untuk membuktikan reproduksinya:

```bash
docker compose exec pipeline python scripts/scraper/consolidate_catalogue.py
```

Hasilnya identik byte demi byte dengan berkas di repositori (39.670 baris, 28 kolom).

Berkas di `scraped/` adalah snapshot 7 Oktober 2026 (36.679 baris daftar, 36.674 halaman detail). Katalog ini baru sekitar separuh NovelUpdates (sekitar 76 ribu judul) dan mengutamakan judul populer, jadi persentase dan tren yang dihitung darinya tidak mewakili seluruh situs.

### Scrap ulang dari NovelUpdates

NovelUpdates berada di balik Cloudflare, jadi scraper memakai cookie sesi dari peramban sungguhan:

1. Buka https://www.novelupdates.com sampai halaman tampil normal.
2. Salin cookie `cf_clearance` (F12, Application, Cookies) dan User-Agent (`navigator.userAgent` di Console).
3. Isi `NU_CF_CLEARANCE` dan `NU_USER_AGENT` di `.env`, lalu `docker compose up -d pipeline`.
4. Jalankan `make scrape-listing`, `make scrape-detail`, lalu `make consolidate`.

Scraper tidak pernah mengirim permintaan paralel, jedanya adaptif (6 sampai 45 detik) dengan cooldown berkala, dan berhenti sendiri bila cookie kedaluwarsa. Proses detail memakan puluhan jam dan dapat dilanjutkan kapan saja. Cookie tersimpan di `.env` yang tidak ikut repositori.

### Catatan hak cipta

Metadata katalog bersifat faktual, tetapi sinopsis tetap ciptaan pemiliknya. Data di repositori ini dipakai untuk keperluan akademik nonkomersial dan tidak dimaksudkan untuk didistribusikan ulang. Periksa ketentuan NovelUpdates dan hak cipta sinopsis sebelum membagikan data di luar konteks itu.

## Port bawaan

Semua antarmuka hanya terbuka di `127.0.0.1`.

| Antarmuka            | Alamat                                                                             |
| -------------------- | ---------------------------------------------------------------------------------- |
| Dashboard            | http://localhost:8501                                                              |
| MinIO Console        | port 9001 (HTTPS dengan CA lokal, peramban bisa menampilkan peringatan sertifikat) |
| Spark Master UI      | http://localhost:7080                                                              |
| HDFS NameNode        | http://localhost:9870                                                              |
| YARN ResourceManager | http://localhost:18088                                                             |
| JupyterLab           | http://localhost:8888                                                              |

Port Spark UI dan YARN sengaja bukan 8080 dan 8088: di Windows kedua port itu sering jatuh di rentang yang dicadangkan Hyper-V. Cek rentang itu dengan `netsh interface ipv4 show excludedportrange protocol=tcp`.

## Beberapa masalah yang mungkin didapat

- **Job Spark berhenti di status WAITING**: memori worker (`SPARK_WORKER_MEMORY`) atau alokasi Docker kurang. Notebook dan pipeline berbagi satu worker, jadi hentikan kernel notebook yang tidak dipakai.
- **`docker compose up` gagal bind port**: ubah port terkait di `.env`.
- **Pipeline berhenti di tengah**: jalankan ulang dengan `--resume` agar langkah yang sudah sukses dilewati. Penulisan Parquet memakai committer v2, jadi tulisan yang gagal di tengah bisa meninggalkan sebagian berkas tanpa `_SUCCESS`; menjalankan ulang menimpa path yang sama.
- **Dashboard menampilkan model lama**: `docker compose restart streamlit`.
- **Scraper dibalas 403 atau "Just a moment"**: cookie atau User-Agent tidak cocok dengan IP saat ini. Ambil ulang dari peramban yang sama.

## Mode publik dashboard

`NOVELLENS_PUBLIC=1` di `.env` menyembunyikan halaman Dataset dan Lake Operations serta teks galat mentah. Gunakan mode ini bila dashboard dipublikasikan, dan ganti seluruh kredensial MinIO sebelumnya.
