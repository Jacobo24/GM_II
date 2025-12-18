import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

try:
    from config import DATA_PROCESSED
except ImportError:
    BASE_DIR = Path(__file__).resolve().parents[1]
    DATA_PROCESSED = BASE_DIR / "data" / "processed"


WIKI_DEFAULT_URLS = [
    "https://en.wikipedia.org/wiki/Bibliography_of_World_War_II",
    # opcionales (puedes activarlos con --also-related)
    "https://en.wikipedia.org/wiki/Bibliography_of_World_War_II_memoirs_and_autobiographies",
    "https://en.wikipedia.org/wiki/Bibliography_of_World_War_II_military_units_and_formations",
]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_id(prefix: str, s: str) -> str:
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{h}"


def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s


def fetch_html(url: str, headers: dict, timeout: int = 30) -> str:
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text


def parse_bibliography_page(html: str, source_url: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")

    # En Wikipedia, el contenido suele estar en #mw-content-text
    content = soup.select_one("#mw-content-text")
    if not content:
        content = soup

    docs: list[dict[str, Any]] = []

    current_section = "Uncategorized"
    section_stack = []

    # Recorremos en orden: headings + listas
    for el in content.find_all(["h2", "h3", "h4", "ul", "ol"], recursive=True):
        if el.name in ("h2", "h3", "h4"):
            # título limpio de sección
            headline = el.get_text(" ", strip=True)
            headline = headline.replace("[edit]", "").strip()

            # mantenemos jerarquía simple
            level = {"h2": 2, "h3": 3, "h4": 4}[el.name]
            while section_stack and section_stack[-1][0] >= level:
                section_stack.pop()
            section_stack.append((level, headline))
            current_section = " > ".join(x[1] for x in section_stack)

        elif el.name in ("ul", "ol"):
            # muchas listas NO son bibliografía (menús, navboxes). Filtramos:
            # nos quedamos solo con listas que estén cerca de un heading (ya tenemos current_section)
            lis = el.find_all("li", recursive=False)
            if not lis:
                continue

            for li in lis:
                text = clean_text(li.get_text(" ", strip=True))
                if not text:
                    continue

                # Filtra “See also” y cosas que no parecen bibliografía
                # (heurística: bibliografía suele tener puntos, comillas, año, editorial, etc.)
                if len(text) < 25:
                    continue

                doc_key = f"{source_url}||{current_section}||{text}"
                doc_id = stable_id("biblio_ww2_en", doc_key)

                docs.append(
                    {
                        "id": doc_id,
                        "texto": text,
                        "fuente": "wikipedia_bibliography",
                        "metadata": {
                            "title": "Bibliography of World War II",
                            "lang": "en",
                            "source_url": source_url,
                            "section": current_section,
                            "retrieved_at": now_utc_iso(),
                            "type": "bibliography_entry",
                        },
                    }
                )

    # Dedup por id
    uniq = {}
    for d in docs:
        uniq[d["id"]] = d
    return list(uniq.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA_PROCESSED / "biblio_docs.jsonl"))
    ap.add_argument("--url", default=WIKI_DEFAULT_URLS[0], help="Página principal de bibliografía WW2")
    ap.add_argument("--also-related", action="store_true", help="Incluye páginas hermanas (memoirs, units, etc.)")
    ap.add_argument("--sleep", type=float, default=0.4)
    ap.add_argument("--user-agent", default="GM_II-RAG/0.1 (contact: you@example.com)")
    args = ap.parse_args()

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)

    urls = [args.url]
    if args.also_related:
        for u in WIKI_DEFAULT_URLS[1:]:
            if u not in urls:
                urls.append(u)

    headers = {"User-Agent": args.user_agent}

    total = 0
    with out_path.open("w", encoding="utf-8") as f_out:
        for url in urls:
            print(f"[INFO] Descargando: {url}")
            html = fetch_html(url, headers=headers)
            docs = parse_bibliography_page(html, source_url=url)
            print(f"[INFO] Docs extraídos de esta página: {len(docs)}")

            for d in docs:
                f_out.write(json.dumps(d, ensure_ascii=False) + "\n")
            total += len(docs)
            time.sleep(args.sleep)

    print(f"[DONE] Guardado: {out_path.resolve()}")
    print(f"[DONE] Total docs: {total}")


if __name__ == "__main__":
    main()
