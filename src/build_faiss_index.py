import argparse
import json
from pathlib import Path

import numpy as np

try:
    import faiss  # faiss-cpu
except ImportError as e:
    raise SystemExit("Falta faiss. Instala: pip install faiss-cpu") from e

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise SystemExit("Falta sentence-transformers. Instala: pip install sentence-transformers") from e

try:
    from config import DATA_PROCESSED
except ImportError:
    BASE_DIR = Path(__file__).resolve().parents[1]
    DATA_PROCESSED = BASE_DIR / "data" / "processed"


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON inválido en {path} línea {i}: {e}") from e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default=str(DATA_PROCESSED / "wiki_chunks.jsonl"))
    ap.add_argument("--out-index", default=str(DATA_PROCESSED / "faiss.index"))
    ap.add_argument("--out-meta", default=str(DATA_PROCESSED / "faiss_meta.jsonl"))
    ap.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-items", type=int, default=0, help="0 = todos, si quieres probar pon 2000")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_index = Path(args.out_index)
    out_meta = Path(args.out_meta)
    out_index.parent.mkdir(parents=True, exist_ok=True)

    # 1) Cargar docs y metadata (guardamos metadata en el mismo orden que los vectores)
    texts = []
    metas = []

    for doc in iter_jsonl(in_path):
        text = doc.get("texto", "")
        if not text:
            continue
        texts.append(text)
        metas.append({
            "id": doc.get("id"),
            "fuente": doc.get("fuente"),
            "metadata": doc.get("metadata", {}) or {},
        })
        if args.max_items and len(texts) >= args.max_items:
            break

    if not texts:
        raise SystemExit(f"No se han encontrado textos en {in_path}")

    print(f"[INFO] Chunks a indexar: {len(texts)}")
    print(f"[INFO] Modelo: {args.model}")

    # 2) Embeddings
    model = SentenceTransformer(args.model)
    emb = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # importante: para cosine similarity con Inner Product
    ).astype(np.float32)

    # 3) FAISS (cosine) usando IndexFlatIP con vectores normalizados
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    # 4) Guardar
    faiss.write_index(index, str(out_index))

    with out_meta.open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"[DONE] Índice guardado en: {out_index.resolve()}")
    print(f"[DONE] Metadata guardada en: {out_meta.resolve()}")
    print(f"[DONE] Vectores: {index.ntotal} | Dim: {dim}")


if __name__ == "__main__":
    main()
