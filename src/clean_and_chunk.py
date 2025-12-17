import argparse
import json
import re
from pathlib import Path

try:
    from config import DATA_PROCESSED
except ImportError:
    BASE_DIR = Path(__file__).resolve().parents[1]
    DATA_PROCESSED = BASE_DIR / "data" / "processed"


REF_RE = re.compile(r"\[\d+\]")          # [1], [23]...
MULTISPACE_RE = re.compile(r"[ \t]+")
MANY_NEWLINES_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    if not text:
        return ""
    t = text

    # Quita referencias tipo [1]
    t = REF_RE.sub("", t)

    # Normaliza espacios
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = MULTISPACE_RE.sub(" ", t)

    # Reduce saltos de línea excesivos
    t = MANY_NEWLINES_RE.sub("\n\n", t)

    # Recorta espacios por línea
    t = "\n".join(line.strip() for line in t.split("\n"))

    # Recorta bordes
    t = t.strip()

    return t


def split_paragraphs(text: str) -> list[str]:
    """
    Divide por párrafos usando líneas en blanco.
    Filtra párrafos demasiado cortos (ruido) pero conserva headings.
    """
    if not text:
        return []

    parts = [p.strip() for p in text.split("\n\n") if p.strip()]

    paras = []
    for p in parts:
        # Mantén headings tipo "== Cronología ==" aunque sean cortos
        is_heading = (p.startswith("==") and p.endswith("==")) or p.startswith("#")
        if is_heading:
            paras.append(p)
            continue

        # Filtra párrafos ultra cortos (ruido)
        if len(p) < 40:
            continue

        paras.append(p)

    return paras


def chunk_with_overlap(paragraphs: list[str], max_chars: int, overlap: int) -> list[str]:
    """
    Une párrafos hasta max_chars.
    Overlap: añade al siguiente chunk los últimos `overlap` caracteres del anterior.
    """
    chunks: list[str] = []
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""

    for p in paragraphs:
        candidate = (buf + "\n\n" + p) if buf else p
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            flush()
            # Si un solo párrafo es más largo que max_chars, lo troceamos “a pelo”
            if len(p) > max_chars:
                start = 0
                while start < len(p):
                    piece = p[start : start + max_chars]
                    chunks.append(piece.strip())
                    start += max_chars
                buf = ""
            else:
                buf = p

    flush()

    if overlap > 0 and len(chunks) >= 2:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = overlapped[-1]
            tail = prev[-overlap:] if len(prev) > overlap else prev
            # intenta empezar el tail en un corte "limpio"
            cut_candidates = [
                tail.rfind("\n\n"),  # fin de párrafo
                tail.rfind(". "),    # fin de frase
                tail.rfind("! "),
                tail.rfind("? "),
                tail.rfind("\n"),    # fin de línea
            ]
            cut = max(cut_candidates)
            if cut != -1 and cut + 2 < len(tail):
                tail = tail[cut + 2 :].lstrip()  # +2 para saltar el separador (". ", "\n\n", etc.)
            merged = (tail + "\n\n" + chunks[i]).strip()
            overlapped.append(merged)
        chunks = overlapped

    return chunks


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON inválido en {path} línea {line_no}: {e}") from e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default=str(DATA_PROCESSED / "wiki_docs.jsonl"))
    parser.add_argument("--out", dest="out_path", default=str(DATA_PROCESSED / "wiki_chunks.jsonl"))
    parser.add_argument("--max-chars", type=int, default=1400)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--min-chars", type=int, default=200, help="Descarta chunks demasiado cortos (ruido)")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_docs = 0
    total_chunks = 0
    skipped_empty = 0

    with out_path.open("w", encoding="utf-8") as f_out:
        for doc in iter_jsonl(in_path):
            total_docs += 1

            doc_id = doc.get("id")
            fuente = doc.get("fuente", "unknown")
            meta = doc.get("metadata", {}) or {}

            raw = doc.get("texto", "")
            cleaned = clean_text(raw)
            if not cleaned:
                skipped_empty += 1
                continue

            paras = split_paragraphs(cleaned)
            chunks = chunk_with_overlap(paras, max_chars=args.max_chars, overlap=args.overlap)

            # filtra chunks cortos
            chunks = [c for c in chunks if len(c) >= args.min_chars]
            if not chunks:
                skipped_empty += 1
                continue

            for idx, chunk_text in enumerate(chunks):
                chunk = {
                    "id": f"{doc_id}_chunk_{idx}",
                    "texto": chunk_text,
                    "fuente": fuente,
                    "metadata": {
                        **meta,
                        "source_id": doc_id,
                        "chunk_index": idx,
                        "chunk_chars": len(chunk_text),
                    },
                }
                f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"[DONE] Input docs: {total_docs}")
    print(f"[DONE] Output chunks: {total_chunks}")
    print(f"[DONE] Skipped docs (empty/too short): {skipped_empty}")
    print(f"[DONE] Wrote: {out_path.resolve()}")


if __name__ == "__main__":
    main()
