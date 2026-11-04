# Base image tunggal untuk seluruh compute layer: wheel PySpark membawa distribusi Spark lengkap, sehingga satu image
# berperan sebagai master, worker, driver, dan Streamlit dengan versi Python dan PySpark yang identik di semua node.
FROM python:3.11-slim-bookworm

ARG HADOOP_AWS_VERSION=3.3.4
ARG AWS_SDK_VERSION=1.12.262

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    SPARK_HOME=/usr/local/lib/python3.11/site-packages/pyspark \
    PYTHONPATH=/opt/project

# Layer 1: system runtime (JVM untuk Spark)
RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
        curl \
        procps \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Layer 2: PySpark (jarang berubah, dipisah agar cache-nya awet)
# Dipasang lebih dulu dan terpisah dari requirements lain supaya perubahan
# dependency ML (yang sering) tidak memaksa unduh ulang jar 200 MB di bawah.
RUN pip install --upgrade pip && pip install pyspark==3.5.3

# Layer 3: konektor S3A (MinIO), di-bake ke image agar startup tidak bergantung pada Maven. Unduhan jar ~200 MB sering terputus
# (curl exit 18), jadi dibungkus retry + resume dan divalidasi sebagai ZIP utuh; jar terpotong lolos cek ukuran tetapi gagal dengan ClassNotFoundException.
RUN set -eux; \
    fetch_jar() { \
        url="$1"; out="$2"; \
        for attempt in 1 2 3 4 5; do \
            curl -fL --retry 5 --retry-delay 5 --retry-all-errors \
                 --connect-timeout 30 --max-time 1800 -C - -o "$out" "$url" || true; \
            if python -c "import sys, zipfile; sys.exit(0 if zipfile.is_zipfile(sys.argv[1]) else 1)" "$out"; then \
                echo "[jar] OK: $out"; return 0; \
            fi; \
            echo "[jar] percobaan $attempt gagal/terpotong: $out - mengulang"; \
            sleep 5; \
        done; \
        echo "[jar] GAGAL permanen: $url" >&2; return 1; \
    }; \
    fetch_jar "https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/${HADOOP_AWS_VERSION}/hadoop-aws-${HADOOP_AWS_VERSION}.jar" \
              "${SPARK_HOME}/jars/hadoop-aws-${HADOOP_AWS_VERSION}.jar"; \
    fetch_jar "https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/${AWS_SDK_VERSION}/aws-java-sdk-bundle-${AWS_SDK_VERSION}.jar" \
              "${SPARK_HOME}/jars/aws-java-sdk-bundle-${AWS_SDK_VERSION}.jar"

ENV PATH="${SPARK_HOME}/bin:${SPARK_HOME}/sbin:${JAVA_HOME}/bin:${PATH}"

# Layer 4: dependency aplikasi (sering berubah, ditaruh paling akhir)
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Layer 5: POS tagger NLTK untuk ide judul AI Studio (src/common/title_templates.py):
# membedakan kata benda/sifat/kerja di cerita pengguna dan di judul katalog.
# Tanpa data ini generator tetap jalan, hanya tanpa pemeriksaan jenis kata.
ENV NLTK_DATA=/opt/nltk_data
RUN python -c "import nltk, sys; sys.exit(0 if nltk.download('averaged_perceptron_tagger_eng', download_dir='/opt/nltk_data', quiet=True) else 1)" \
    && chmod -R a+rX /opt/nltk_data

# Layer 6: workspace
WORKDIR /opt/project
RUN useradd --create-home --uid 1000 spark \
    && mkdir -p /opt/project /home/spark/.ivy2 \
    && chown -R spark:spark /opt/project /home/spark

USER spark

CMD ["bash"]

# Layer 7: batasi thread BLAS/OpenMP. OpenBLAS membuka thread-pool sebesar jumlah core host dan di container berisi JVM Spark
# ini bisa deadlock (semua thread di futex_wait, job menggantung tanpa error); pada dataset ini batas eksplisit tidak memperlambat.
ENV OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2
