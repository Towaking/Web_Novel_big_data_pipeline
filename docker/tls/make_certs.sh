#!/bin/sh
# Private CA + MinIO server certificate for TLS inside the stack.
# Idempotent: an existing certificate is kept while it is valid for 30+ more days.
# Output (volume tls-certs, mounted read-only at /certs elsewhere):
#   ca.crt               trust anchor for Python (AWS_CA_BUNDLE) and mc
#   truststore.p12       JVM default CAs + ca.crt, for Spark/Hadoop S3A
#   minio/public.crt     server certificate, SAN minio, localhost, 127.0.0.1
#   minio/private.key    server key, readable by root (MinIO) only
set -eu
OUT=/certs
mkdir -p "$OUT/minio/CAs"

if [ -f "$OUT/minio/public.crt" ] && openssl x509 -checkend 2592000 -noout -in "$OUT/minio/public.crt"; then
  echo "[certs] existing certificate still valid, nothing to do"
  exit 0
fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

openssl req -x509 -new -newkey rsa:3072 -nodes -sha256 -days 825 \
  -subj "/CN=NovelLens local CA" -keyout "$WORK/ca.key" -out "$OUT/ca.crt"

cat > "$WORK/server.ext" <<EOF
basicConstraints=CA:FALSE
keyUsage=digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
subjectAltName=DNS:minio,DNS:localhost,IP:127.0.0.1
EOF
openssl req -new -newkey rsa:2048 -nodes -sha256 -subj "/CN=minio" \
  -keyout "$OUT/minio/private.key" -out "$WORK/server.csr"
openssl x509 -req -in "$WORK/server.csr" -CA "$OUT/ca.crt" -CAkey "$WORK/ca.key" -CAcreateserial \
  -days 397 -sha256 -extfile "$WORK/server.ext" -out "$OUT/minio/public.crt"
cp "$OUT/ca.crt" "$OUT/minio/CAs/ca.crt"

# The CA key is not kept: a new server certificate means a new CA, generated here.
rm -f "$OUT/truststore.p12"
keytool -importkeystore -noprompt -srckeystore "$JAVA_HOME/lib/security/cacerts" -srcstorepass changeit \
  -destkeystore "$OUT/truststore.p12" -deststoretype PKCS12 -deststorepass changeit >/dev/null 2>&1
keytool -importcert -noprompt -alias novellens-ca -file "$OUT/ca.crt" \
  -keystore "$OUT/truststore.p12" -storetype PKCS12 -storepass changeit >/dev/null

chmod 0644 "$OUT/ca.crt" "$OUT/minio/public.crt" "$OUT/minio/CAs/ca.crt" "$OUT/truststore.p12"
chmod 0600 "$OUT/minio/private.key"
echo "[certs] new CA and MinIO certificate written"
openssl x509 -noout -subject -enddate -ext subjectAltName -in "$OUT/minio/public.crt"
