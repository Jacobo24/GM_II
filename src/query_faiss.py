import argparse
import json
from pathlib import Path

import numpy as np

import faiss
from sentence_transformers import SentenceTransformer

try:
    from config import DATA_PROCESSED
except ImportError:
    BASE_DIR = Path(__file__).resolve().parents[1]
    DATA_PROCESSED = BASE_DIR / "data" / "processed"


def load_jsonl(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(DATA_PROCESSED / "faiss.index"))
    ap.add_argument("--meta", default=str(DATA_PROCESSED / "faiss_meta.jsonl"))
    ap.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--query", required=True)
    args = ap.parse_args()

    index_path = Path(args.index)
    meta_path = Path(args.meta)

    index = faiss.read_index(str(index_path))
    metas = load_jsonl(meta_path)

    model = SentenceTransformer(args.model)
    q = model.encode([args.query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    scores, idxs = index.search(q, args.k)

    print(f"\n[Q] {args.query}\n")
    for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), start=1):
        m = metas[int(idx)]
        md = m.get("metadata", {})
        title = md.get("title")
        url = md.get("url")
        chunk_index = md.get("chunk_index")
        source_id = md.get("source_id")

        print(f"#{rank}  score={float(score):.4f}")
        print(f"    title: {title}")
        print(f"    url:   {url}")
        print(f"    id:    {m.get('id')} (source_id={source_id}, chunk_index={chunk_index})")
        print()

if __name__ == "__main__":
    main()
