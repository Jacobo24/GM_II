import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader  # pip install pypdf

try:
    from config import DATA_RAW, DATA_PROCESSED
except ImportError:
    BASE_DIR = Path(__file__).resolve().parents[1]
    DATA_RAW = BASE_DIR / "data" / "raw"
    DATA_PROCESSED = BASE_DIR / "data" / "processed"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_pdf_text(pdf_path: Path, max_pages: int = 0) -> list[str]:
    reader = PdfReader(str(pdf_path))
    pages = []
    n = len(reader.pages)

    if max_pages and max_pages > 0:
        n = min(n, max_pages)

    for i in range(n):
        page = reader.pages[i]
        try:
            txt = (page.extract_text() or "").strip()
        except Exception as e:
            print(f"[WARN] {pdf_path.name} página {i+1}: fallo extract_text -> {type(e).__name__}")
            txt = ""
        pages.append(txt)

    return pages



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", default=str(DATA_RAW / "pdfs"))
    parser.add_argument("--out", default=str(DATA_PROCESSED / "pdf_docs.jsonl"))
    parser.add_argument("--min-page-chars", type=int, default=200, help="Descarta páginas casi vacías")
    parser.add_argument("--group-pages", type=int, default=5, help="Agrupa N páginas por doc (reduce docs)")
    parser.add_argument("--max-pages", type=int, default=0, help="0 = todas; si quieres probar pon 50")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pdfs = sorted([p for p in pdf_dir.glob("*.pdf") if p.is_file()])
    if not pdfs:
        raise SystemExit(f"No hay PDFs en: {pdf_dir.resolve()}")

    total_docs = 0

    with out_path.open("w", encoding="utf-8") as f_out:
        for pdf in pdfs:
            pages = extract_pdf_text(pdf, max_pages=args.max_pages)

            try:
                pages = extract_pdf_text(pdf, max_pages=args.max_pages)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"[ERROR] PDF {pdf.name}: {e}")
                continue

            # filtra páginas vacías
            keep = [(i, t) for i, t in enumerate(pages) if t and len(t) >= args.min_page_chars]
            if not keep:
                continue

            # agrupar páginas para que cada doc no sea ENORME ni haya demasiados docs
            group = args.group_pages
            for start in range(0, len(keep), group):
                chunk = keep[start : start + group]
                page_ids = [i for i, _ in chunk]
                text = "\n\n".join(t for _, t in chunk).strip()
                if not text:
                    continue

                doc_id = f"pdf_{pdf.stem}_p{page_ids[0]+1}_p{page_ids[-1]+1}"
                doc = {
                    "id": doc_id,
                    "texto": text,
                    "fuente": "pdf",
                    "metadata": {
                        "title": pdf.stem,
                        "path": str(pdf.resolve()),
                        "page_start": int(page_ids[0] + 1),
                        "page_end": int(page_ids[-1] + 1),
                        "retrieved_at": now_utc_iso(),
                    },
                }
                f_out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                total_docs += 1

    print(f"[DONE] PDFs: {len(pdfs)}")
    print(f"[DONE] Wrote docs: {total_docs}")
    print(f"[DONE] Output: {out_path.resolve()}")


if __name__ == "__main__":
    main()