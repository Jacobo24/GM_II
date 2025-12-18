import json
from pathlib import Path
from typing import Any

import numpy as np
import requests
import streamlit as st

import faiss
from sentence_transformers import SentenceTransformer


# -------------------------
# Config
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

DEFAULT_INDEX = DATA_DIR / "faiss.index"
DEFAULT_META = DATA_DIR / "faiss_meta.jsonl"
DEFAULT_CHUNKS = DATA_DIR / "all_chunks.jsonl"

DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1:latest"


# -------------------------
# Helpers
# -------------------------
def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


@st.cache_resource
def load_embedder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


@st.cache_resource
def load_faiss_index(index_path: str):
    return faiss.read_index(index_path)


@st.cache_resource
def load_store(meta_path: str, chunks_path: str):
    metas = load_jsonl(Path(meta_path))
    chunks = load_jsonl(Path(chunks_path))
    return metas, chunks


def build_context(hits: list[dict[str, Any]], max_context_chars: int) -> tuple[str, list[dict[str, Any]]]:
    blocks = []
    sources = []
    used = 0

    for h in hits:
        text = h.get("texto", "")
        md = h.get("metadata", {}) or {}
        title = md.get("title")
        url = md.get("url")

        block = (
            f"[CHUNK]\n"
            f"chunk_id: {h.get('id')}\n"
            f"title: {title}\n"
            f"url: {url}\n"
            f"text:\n{text}\n"
        )

        if used + len(block) > max_context_chars:
            break

        blocks.append(block)
        used += len(block)

        sources.append(
            {
                "title": title,
                "url": url,
                "chunk_id": h.get("id"),
                "source_id": md.get("source_id"),
                "chunk_index": md.get("chunk_index"),
                "score": float(h.get("_score", 0.0)),
            }
        )

    return "\n---\n".join(blocks), sources


def ollama_respond(host: str, model: str, system: str, user: str, temperature: float = 0.2) -> str:
    host = host.rstrip("/")

    # 1) intento con /api/chat
    chat_url = host + "/api/chat"
    chat_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }

    try:
        r = requests.post(chat_url, json=chat_payload, timeout=180)
        if r.status_code == 200:
            data = r.json()
            return data["message"]["content"]
        # si es 500 u otro, probamos fallback
    except Exception:
        pass

    # 2) fallback con /api/generate
    gen_url = host + "/api/generate"
    prompt = f"SISTEMA:\n{system}\n\nUSUARIO:\n{user}\n"
    gen_payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    r2 = requests.post(gen_url, json=gen_payload, timeout=180)
    r2.raise_for_status()
    data2 = r2.json()
    return data2.get("response", "")

def expand_query_for_retrieval(query: str) -> str:
    q = query.lower()

    extras = []

    # Heurística: si detectamos Frente Oriental / Barbarroja / URSS
    if any(x in q for x in ["frente oriental", "barbarroja", "unión soviética", "urss", "eastern front", "soviet"]):
        extras += [
            "logística", "suministro", "abastecimiento", "transporte", "ferrocarril",
            "combustible", "munición", "invierno", "equipamiento", "líneas de suministro",
            "railway", "rail gauge", "supply lines", "fuel", "winter", "logistics"
        ]

    # Si el usuario ya menciona logística, reforzamos también
    if "logístic" in q or "suministr" in q or "abastec" in q:
        extras += [
            "fuel", "ammunition", "transport", "railway", "winter equipment",
            "partisan", "roads", "mud", "rasputitsa"
        ]

    if not extras:
        return query

    # Importante: esto se usa SOLO para retrieval
    return query + " " + " ".join(extras)


def retrieve(query: str, embedder: SentenceTransformer, index, metas, chunks, k: int) -> list[dict[str, Any]]:
    retrieval_query = expand_query_for_retrieval(query)
    qv = embedder.encode([retrieval_query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    scores, idxs = index.search(qv, k)

    hits = []
    for score, idx in zip(scores[0], idxs[0]):
        idx = int(idx)
        if idx < 0:
            continue

        doc = (chunks[idx].copy() if idx < len(chunks) else {})
        md_wrap = (metas[idx] if idx < len(metas) else {})

        # Alineación: en tu builder metas tiene {"id","fuente","metadata"}
        if isinstance(md_wrap, dict) and "metadata" in md_wrap and isinstance(md_wrap["metadata"], dict):
            doc.setdefault("metadata", {})
            doc["metadata"] = {**md_wrap["metadata"], **(doc.get("metadata") or {})}
            if "id" in md_wrap and not doc.get("id"):
                doc["id"] = md_wrap["id"]
            if "fuente" in md_wrap and not doc.get("fuente"):
                doc["fuente"] = md_wrap["fuente"]

        doc["_score"] = float(score)
        hits.append(doc)

    return hits


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="RAG WWII (Ollama + FAISS)", layout="wide")

st.title("RAG Segunda Guerra Mundial — Ollama + FAISS")

with st.sidebar:
    st.header("Configuración")

    index_path = st.text_input("FAISS index", str(DEFAULT_INDEX))
    meta_path = st.text_input("FAISS meta (jsonl)", str(DEFAULT_META))
    chunks_path = st.text_input("Chunks (jsonl)", str(DEFAULT_CHUNKS))

    embed_model = st.text_input("Embedding model", DEFAULT_EMBED_MODEL)

    st.divider()

    ollama_host = st.text_input("Ollama host", DEFAULT_OLLAMA_HOST)
    ollama_model = st.text_input("Ollama model", DEFAULT_OLLAMA_MODEL)

    st.divider()
    k = st.slider("Top-k (retrieval)", 3, 15, 6)
    max_context_chars = st.slider("Máx caracteres de contexto", 4000, 20000, 12000, step=1000)
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.2, step=0.05)

    show_retrieved = st.checkbox("Mostrar chunks recuperados", value=False)

# Carga recursos
try:
    embedder = load_embedder(embed_model)
    index = load_faiss_index(index_path)
    metas, chunks = load_store(meta_path, chunks_path)
except Exception as e:
    st.error(f"Error cargando recursos: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input de chat
prompt = st.chat_input("Escribe tu pregunta…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieval
    with st.spinner("Buscando contexto (FAISS)…"):
        hits = retrieve(prompt, embedder, index, metas, chunks, k=k)
        context_str, sources = build_context(hits, max_context_chars=max_context_chars)

    system = (
        "Eres un asistente de historia especializado en la Segunda Guerra Mundial.\n"
        "Responde SIEMPRE en español.\n"
        "Usa SOLO la información del CONTEXTO proporcionado.\n"
        "Si el contexto no contiene la respuesta, di claramente que no aparece en las fuentes.\n"
        "Si el contexto contiene principalmente referencias bibliográficas sin explicar el contenido, indícalo.\n"
        "No inventes detalles de libros; limita tu respuesta a lo que aparece en el contexto.\n"
        "Al final, incluye una sección 'Fuentes' con una lista de title + url.\n"
        "Responde de forma explicativa y desarrollada.\n"
        "No respondas con sí/no.\n\n"
    )


    user_msg = (
        "Responde de forma explicativa y desarrollada.\n"
        "No respondas con sí/no.\n\n"
        f"Pregunta: {prompt}\n\n"
        f"CONTEXTO:\n{context_str}\n"
    )


    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]

    # LLM (Ollama)
    with st.spinner("Generando respuesta (Ollama)…"):
        try:
            answer = ollama_respond(ollama_host, ollama_model, system, user_msg, temperature=temperature)
        except Exception as e:
            st.error(f"Error llamando a Ollama: {e}")
            st.stop()

    # Output
    with st.chat_message("assistant"):
        st.markdown(answer)

        st.subheader("Fuentes (recuperadas)")
        for i, s in enumerate(sources, start=1):
            title = s.get("title") or "(sin título)"
            url = s.get("url") or ""
            st.markdown(f"{i}. **{title}** — {url}  \nscore={s.get('score'):.4f} | chunk={s.get('chunk_id')}")

        if show_retrieved:
            st.subheader("Chunks recuperados (debug)")
            for i, h in enumerate(hits, start=1):
                md = h.get("metadata", {}) or {}
                st.markdown(f"### #{i} score={h.get('_score'):.4f} — {md.get('title')}")
                st.code(h.get("texto", "")[:2000])

    st.session_state.messages.append({"role": "assistant", "content": answer})
