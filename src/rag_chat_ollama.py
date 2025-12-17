import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import requests
import faiss
from sentence_transformers import SentenceTransformer

try:
    from config import DATA_PROCESSED
except ImportError:
    BASE_DIR = Path(__file__).resolve().parents[1]
    DATA_PROCESSED = BASE_DIR / "data" / "processed"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def build_context(hits: list[dict[str, Any]], max_context_chars: int) -> tuple[str, list[dict[str, Any]]]:
    """
    Devuelve:
      - context_str: texto para el prompt
      - sources: lista de fuentes (title/url/id/score)
    """
    chunks_out = []
    sources = []
    used = 0

    for h in hits:
        text = h["texto"]
        md = h.get("metadata", {}) or {}
        title = md.get("title")
        url = md.get("url")
        chunk_id = h.get("id")
        score = h.get("_score")

        block = (
            f"[CHUNK]\n"
            f"source_id: {md.get('source_id')}\n"
            f"chunk_id: {chunk_id}\n"
            f"title: {title}\n"
            f"url: {url}\n"
            f"text:\n{text}\n"
        )

        if used + len(block) > max_context_chars:
            break

        chunks_out.append(block)
        used += len(block)

        sources.append({
            "title": title,
            "url": url,
            "chunk_id": chunk_id,
            "source_id": md.get("source_id"),
            "chunk_index": md.get("chunk_index"),
            "score": float(score) if score is not None else None,
        })

    return "\n---\n".join(chunks_out), sources


def ollama_chat(host: str, model: str, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
    """
    Usa /api/chat (Ollama).
    """
    url = host.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=6)

    ap.add_argument("--index", default=str(DATA_PROCESSED / "faiss.index"))
    ap.add_argument("--meta", default=str(DATA_PROCESSED / "faiss_meta.jsonl"))
    ap.add_argument("--chunks", default=str(DATA_PROCESSED / "wiki_chunks.jsonl"))

    ap.add_argument("--embed-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    ap.add_argument("--ollama-host", default="http://localhost:11434")
    ap.add_argument("--ollama-model", default="llama3.1")  # cámbialo por el tuyo: mistral, qwen2.5, llama3.2...
    ap.add_argument("--temperature", type=float, default=0.2)

    ap.add_argument("--max-context-chars", type=int, default=12000)
    args = ap.parse_args()

    index_path = Path(args.index)
    meta_path = Path(args.meta)
    chunks_path = Path(args.chunks)

    if not index_path.exists():
        raise SystemExit(f"No existe el índice: {index_path}")
    if not meta_path.exists():
        raise SystemExit(f"No existe la metadata: {meta_path}")
    if not chunks_path.exists():
        raise SystemExit(f"No existe el fichero de chunks: {chunks_path}")

    # Load index + metadata + chunks
    print("[INFO] Cargando FAISS...")
    index = faiss.read_index(str(index_path))
    metas = load_jsonl(meta_path)
    chunks = load_jsonl(chunks_path)

    if len(metas) != len(chunks):
        print(f"[WARN] metas ({len(metas)}) != chunks ({len(chunks)}). "
              f"Esto suele pasar si reindexaste con un subset. Aun así intentamos seguir.")

    # Embed query
    print("[INFO] Cargando modelo de embeddings...")
    embedder = SentenceTransformer(args.embed_model)

    q = embedder.encode([args.query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    # Search
    scores, idxs = index.search(q, args.k)

    # Build hits list
    hits = []
    for score, idx in zip(scores[0], idxs[0]):
        idx = int(idx)
        if idx < 0:
            continue
        # En principio, chunks y metas están alineados por posición
        doc = chunks[idx].copy() if idx < len(chunks) else {}
        md_wrap = metas[idx] if idx < len(metas) else {}

        # Preferimos metadata del chunk (tiene chunk_index/source_id)
        doc.setdefault("metadata", {})
        if isinstance(md_wrap, dict):
            # md_wrap guarda {"id","fuente","metadata"} en el index builder
            # nos aseguramos de no pisar el texto
            if "metadata" in md_wrap and isinstance(md_wrap["metadata"], dict):
                doc["metadata"] = {**md_wrap["metadata"], **(doc.get("metadata") or {})}
            if "id" in md_wrap and not doc.get("id"):
                doc["id"] = md_wrap["id"]
            if "fuente" in md_wrap and not doc.get("fuente"):
                doc["fuente"] = md_wrap["fuente"]

        doc["_score"] = float(score)
        hits.append(doc)

    context_str, sources = build_context(hits, max_context_chars=args.max_context_chars)

    system = (
        "Eres un asistente de historia especializado en la Segunda Guerra Mundial.\n"
        "Responde SIEMPRE en español.\n"
        "Usa SOLO la información del CONTEXTO proporcionado.\n"
        "Si el contexto no contiene la respuesta, di claramente que no aparece en las fuentes.\n"
        "Al final, incluye una sección 'Fuentes' con una lista de title + url (sin inventar).\n"
    )

    user = (
        f"Pregunta: {args.query}\n\n"
        f"CONTEXTO (fragmentos recuperados):\n{context_str}\n"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    print("[INFO] Llamando a Ollama...")
    answer = ollama_chat(
        host=args.ollama_host,
        model=args.ollama_model,
        messages=messages,
        temperature=args.temperature,
    )

    print("\n" + "=" * 80)
    print(answer.strip())
    print("=" * 80)

    # Fuentes (las imprimimos aparte para que siempre estén, incluso si el modelo se despista)
    print("\nFuentes (chunks recuperados):")
    for i, s in enumerate(sources, start=1):
        print(f"{i}. {s.get('title')} — {s.get('url')}  (score={s.get('score'):.4f}, chunk={s.get('chunk_id')})")


if __name__ == "__main__":
    main()