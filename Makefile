# Shortcut operasional pipeline.
# Windows: pakai `make` dari Git Bash / WSL, atau salin perintah
# di kolom kanan langsung ke PowerShell.

COMPOSE := docker compose
EXEC    := $(COMPOSE) exec -T pipeline

.DEFAULT_GOAL := help
.PHONY: help build up down restart ps logs verify pipeline governance scraper-deps scrape-detail scrape-status catalog datamodel graph-analysis audit variety hdfs-load mapreduce mapreduce-local mapreduce-verify bigdata-audit streaming-simulate streaming-run streaming-reset hdfs-ui yarn-ui phase1 phase2 phase3 phase4 phase5 similarity-graph scrape-listing consolidate shell spark-ui minio-ui dashboard clean nuke

help:  ## Tampilkan daftar perintah
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

build:  ## Bangun image dasar (java + pyspark + konektor S3A)
	$(COMPOSE) build

up:  ## Jalankan seluruh stack di background
	$(COMPOSE) up -d

down:  ## Hentikan stack (volume data tetap aman)
	$(COMPOSE) down

restart: down up  ## Restart stack

ps:  ## Status seluruh service
	$(COMPOSE) ps

logs:  ## Ikuti log semua service
	$(COMPOSE) logs -f --tail=100

verify:  ## Smoke test environment (MinIO + Spark + S3A)
	$(EXEC) python scripts/verify_env.py

pipeline:  ## Jalankan seluruh pipeline Fase 1-4 + lapisan tata kelola
	$(EXEC) python scripts/run_pipeline.py

governance:  ## Hanya lapisan tata kelola (inventaris + model data + audit)
	$(EXEC) python scripts/run_pipeline.py --only-governance

scraper-deps:  ## Pasang ulang dependensi scraper (perlu bila container dibuat ulang)
	$(EXEC) pip install --quiet --retries 10 --timeout 120 curl_cffi
	$(EXEC) python -c "import curl_cffi; print('curl_cffi', curl_cffi.__version__, 'siap')"

scrape-listing: scraper-deps  ## Scrape daftar seri (resumable)
	$(COMPOSE) exec pipeline python scripts/scraper/novelupdates.py listing

scrape-detail: scraper-deps  ## Lanjutkan scrape detail 27 kolom (resumable)
	$(COMPOSE) exec pipeline python scripts/scraper/novelupdates.py detail

consolidate:  ## Satukan scrape + arsip menjadi Dataset/novel_catalogue.csv
	$(EXEC) python scripts/scraper/consolidate_catalogue.py

scrape-status:  ## Berapa yang sudah terkumpul
	@echo "listing: $$(( $$(wc -l < Dataset/scraped/listing.csv) - 1 )) novel"
	@echo "detail : $$(( $$(wc -l < Dataset/scraped/detail.csv) - 1 )) novel"

catalog:  ## Inventaris elemen data & klasifikasi format
	$(EXEC) python -m src.phases.data_catalog

datamodel:  ## Model data - star schema + proyeksi graf
	$(EXEC) python -m src.phases.data_model

graph-analysis:  ## PageRank, deteksi komunitas, traversal (NetworkX di atas property graph)
	$(EXEC) python -m src.phases.graph_analysis

audit:  ## Audit tata kelola IT (security/compliance/privacy/quality)
	$(EXEC) python -m src.phases.governance_audit

variety:  ## Pisahkan dataset per format (CSV / JSONL / TXT) ke Dataset/variety/
	$(EXEC) python -m src.phases.data_variety

hdfs-load:  ## Muat Dataset/variety/ ke HDFS (blok 4 MB, replikasi 2) + fsck
	$(COMPOSE) exec -T hadoop-client bash scripts/hdfs/load_hdfs.sh

mapreduce:  ## Jalankan 4 job MapReduce (Hadoop Streaming) di YARN
	$(COMPOSE) exec -T hadoop-client bash scripts/mapreduce/run_jobs.sh

mapreduce-local:  ## Job yang sama tanpa Hadoop (cat | mapper | sort | reducer)
	$(EXEC) bash scripts/mapreduce/local_smoke.sh

mapreduce-verify:  ## Cocokkan hasil MapReduce (HDFS) dengan Spark SQL (zona clean)
	$(EXEC) python scripts/mapreduce/verify_mapreduce.py

bigdata-audit:  ## Audit konsep Big Data (5V, DFS, MapReduce, keandalan)
	$(EXEC) python -m src.phases.bigdata_audit

OFFSET ?= 0
ROWS ?= 500
streaming-simulate:  ## Kirim satu batch landing baru (uji Velocity) - args: make streaming-simulate OFFSET=500 ROWS=500
	$(EXEC) python scripts/streaming/simulate_landing.py --offset $(OFFSET) --rows $(ROWS)

streaming-run:  ## Proses berkas landing baru via Structured Streaming (berhenti sendiri)
	$(EXEC) python -m src.phases.streaming_ingest

streaming-reset:  ## Hapus checkpoint & tabel streaming (ulangi demo dari nol)
	$(EXEC) python -m src.phases.streaming_ingest --reset

hdfs-ui:  ## URL NameNode UI (HDFS)
	@echo "http://localhost:9870"

yarn-ui:  ## URL ResourceManager UI (YARN / job MapReduce)
	@echo "http://localhost:18088"

phase1:  ## Fase 1 - ingestion CSV ke MinIO
	$(EXEC) python -m src.phases.phase1_collect

phase2:  ## Fase 2 - cleaning & text preprocessing
	$(EXEC) python -m src.phases.phase2_clean

phase3:  ## Fase 3 - EDA Spark SQL
	$(EXEC) python -m src.phases.phase3_explore

phase4:  ## Fase 4 - training model multi-head
	$(EXEC) python -m src.phases.phase4_model

phase5:  ## Fase 5 - perbandingan scikit-learn vs Spark MLlib (Head A dan Head D)
	$(EXEC) python -m src.phases.phase5_mllib

similarity-graph:  ## Graf kemiripan novel: kNN, Louvain, PageRank, homofili
	$(EXEC) python -m src.phases.similarity_graph

dashboard:  ## URL dashboard Streamlit
	@echo "http://localhost:8501"

shell:  ## Masuk ke container driver
	$(COMPOSE) exec pipeline bash

spark-ui:  ## URL Spark Master UI (port dipindah dari 8080: rentang port dinamis Hyper-V)
	@echo "http://localhost:7080"

minio-ui:  ## URL MinIO Console
	@echo "http://localhost:9001"

clean:  ## Hapus container + network (volume MinIO tetap)
	$(COMPOSE) down --remove-orphans

nuke:  ## Hapus SEMUANYA termasuk data lake di volume MinIO
	$(COMPOSE) down -v --remove-orphans
