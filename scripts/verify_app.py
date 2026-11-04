"""
Smoke test dashboard (Fase 5)
Menjalankan `app/streamlit_app.py` secara headless memakai `AppTest` bawaan
Streamlit, lalu memeriksa setiap halaman tidak melempar exception dan form
AI Studio menghasilkan genre untuk setiap model yang bisa dipilih.

Kenapa perlu: kesalahan pada dashboard (nama kolom salah, tabel belum ada,
argumen widget tidak valid) hanya muncul saat halaman dirender. Tanpa tes ini,
kegagalan baru ketahuan ketika seseorang membukanya di browser.

Catatan: tes ini memvalidasi LOGIKA halaman, bukan tampilannya.

Jalankan:
    docker compose exec streamlit python scripts/verify_app.py
"""

from __future__ import annotations

import sys

from streamlit.testing.v1 import AppTest

from src.common.logging_utils import get_logger, log_section

logger = get_logger("verify_app")

APP_PATH = "/opt/project/app/streamlit_app.py"
#: Nama tampilan -> fungsi halaman di streamlit_app.py.
PAGES = {
    "Overview": "page_overview",
    "Title Lookup": "page_lookup",
    "AI Studio": "page_studio",
    "Catalogue Analytics": "page_analytics",
    "Market Trends": "page_trends",
    "Genre Explorer": "page_genre",
    "Story Elements": "page_tags",
    "Authors & Publishers": "page_creators",
    "Data Warehouse": "page_warehouse",
    "Relationship Graph": "page_graph",
    "Content Graph": "page_content_graph",
    "Dataset": "page_dataset",
    "Lake Operations": "page_lake",
    "Quality & Governance": "page_governance",
    "Methodology": "page_method",
}
TIMEOUT_SECONDS = 180


def check_page(page: str) -> bool:
    """Render satu halaman dan laporkan exception yang muncul."""
    # Router `st.navigation` tidak bisa dipindah halamannya dari AppTest pada
    # Streamlit 1.39, jadi fungsi halamannya dipanggil langsung: seluruh kode
    # modul (tema, loader, helper) tetap dieksekusi, hanya `main()` yang diganti.
    with open(APP_PATH, encoding="utf-8") as handle:
        source = handle.read().rsplit("\nmain()", 1)[0]
    app = AppTest.from_string(f"{source}\n{PAGES[page]}()\n", default_timeout=TIMEOUT_SECONDS)
    app.run()

    if app.exception:
        for exc in app.exception:
            logger.error("[%s] exception: %s", page, exc.value)
        return False

    logger.info(
        "[%s] OK | %d judul, %d chart, %d tabel, %d metrik",
        page,
        len(app.title) + len(app.subheader),
        len(app.get("plotly_chart")),
        len(app.dataframe),
        len(app.metric),
    )
    return True


def check_studio_models() -> list[tuple[str, bool]]:
    """Tekan "Analyse story" sekali per model genre; merender halaman saja tidak menjalankan inferensi."""
    from src.common.model_registry import load_registry

    with open(APP_PATH, encoding="utf-8") as handle:
        script = handle.read().rsplit("\nmain()", 1)[0] + "\npage_studio()\n"
    # Pilihan radio di AppTest berupa label terformat ("Linear SVM · F1 0.61"), jadi nama model dari artefak.
    return [_submit_story(script, name) for name in load_registry().genre_model_names]


def _submit_story(script: str, model: str) -> tuple[str, bool]:
    app = AppTest.from_string(script, default_timeout=TIMEOUT_SECONDS)
    app.run()
    radio = next((r for r in app.radio if r.label == "Genre model"), None)
    submit = next((b for b in app.button if "Analyse story" in str(b.label)), None)
    if radio is None or submit is None:
        logger.error("[AI Studio / %s] pemilih model atau tombol Analyse story tidak ditemukan", model)
        return f"AI Studio: {model}", False
    radio.set_value(model)
    submit.click()
    app.run()
    rendered = " ".join(m.value for m in app.markdown)
    ok = not app.exception and "predicted" in rendered
    if app.exception:
        for exc in app.exception:
            logger.error("[AI Studio / %s] exception: %s", model, exc.value)
    logger.info("[AI Studio / %s] %s", model, "OK" if ok else "tidak ada genre terprediksi")
    return f"AI Studio: {model}", ok


def main() -> int:
    log_section(logger, "smoke test dashboard streamlit")
    results = [(page, check_page(page)) for page in PAGES]
    results += check_studio_models()

    log_section(logger, "ringkasan")
    for page, ok in results:
        logger.info("%-26s : %s", page, "LULUS" if ok else "GAGAL")

    if all(ok for _, ok in results):
        logger.info("Dashboard siap dibuka di http://localhost:8501")
        return 0
    logger.error("Ada halaman yang gagal dirender.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
