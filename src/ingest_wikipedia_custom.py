import argparse
import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import requests

try:
    from config import DATA_PROCESSED, DATA_RAW
except ImportError:
    BASE_DIR = Path(__file__).resolve().parents[1]
    DATA_PROCESSED = BASE_DIR / "data" / "processed"
    DATA_RAW = BASE_DIR / "data" / "raw"


# -------------------------
# Utils
# -------------------------
def read_keywords(path: Path) -> list[str]:
    kws = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            kws.append(s)
    return kws


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def wiki_api(lang: str) -> str:
    return f"https://{lang}.wikipedia.org/w/api.php"


def normalize_category_title(cat: str) -> str:
    """
    Acepta:
      - 'Category:World War II'
      - 'Categoría:Segunda Guerra Mundial'
      - 'World War II'  -> lo convierte en 'Category:World War II'
    """
    s = cat.strip()
    if s.lower().startswith("category:") or s.lower().startswith("categoría:"):
        return s
    # si no trae prefijo, asumimos inglés "Category:" como convención interna
    return f"Category:{s}"


# -------------------------
# Wikipedia: search + page fetch
# -------------------------
def search_wiki_title(query: str, lang: str, headers: dict, timeout: int = 20) -> str | None:
    url = wiki_api(lang)
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srlimit": 5,
    }
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    hits = data.get("query", {}).get("search", [])
    if not hits:
        return None

    q = query.lower()

    def score(title: str) -> int:
        t = (title or "").lower()
        s = 0

        # preferimos que el título contenga palabras del query
        for w in q.replace("(", " ").replace(")", " ").split():
            if len(w) >= 4 and w in t:
                s += 3

        # preferir "Lista de" / "List of" cuando la query sugiere listados
        wants_list = any(k in q for k in ["batallas", "armamento", "aeronaves", "carros", "operaciones", "equipment", "battles", "operations", "tanks", "aircraft"])
        if wants_list and (t.startswith("lista de") or t.startswith("list of") or "lista" in t or "list" in t):
            s += 5

        # evitar desvíos muy genéricos
        if "aliados" in t and "batallas" in q:
            s -= 3

        return s

    best = max(hits, key=lambda h: score(h.get("title", "")))
    return best.get("title")



def fetch_wiki_page(title: str, lang: str, headers: dict, timeout: int = 20, allow_search_fallback: bool = True) -> dict | None:
    """
    Descarga una página por título.
    Si no existe y allow_search_fallback=True, hace una búsqueda y reintenta con el título sugerido.
    """
    url = wiki_api(lang)
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "titles": title,
    }

    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None

    page = next(iter(pages.values()))
    if "missing" in page:
        if allow_search_fallback:
            fixed = search_wiki_title(title, lang=lang, headers=headers, timeout=timeout)
            if fixed and fixed != title:
                print(f"[FIX] ({lang}) '{title}' -> '{fixed}'")
                return fetch_wiki_page(
                    fixed, lang=lang, headers=headers, timeout=timeout, allow_search_fallback=False
                )
        print(f"[WARN] ({lang}) No encontrada: {title}")
        return None

    extract = (page.get("extract") or "").strip()
    if not extract:
        print(f"[WARN] ({lang}) Sin extracto: {title}")
        return None

    normalized_title = page.get("title", title)
    pageid = page.get("pageid")

    doc_id = f"wiki_{lang}_{pageid}" if pageid is not None else f"wiki_{lang}_{normalized_title.replace(' ', '_')}"
    retrieved_at = now_utc_iso()

    return {
        "id": doc_id,
        "texto": extract,
        "fuente": "wikipedia",
        "metadata": {
            "title": normalized_title,
            "lang": lang,
            "pageid": pageid,
            "url": (f"https://{lang}.wikipedia.org/?curid={pageid}" if pageid is not None else None),
            "original_query": title,
            "retrieved_at": retrieved_at,
            "source": "title",
        },
    }


def fetch_wiki_pages_by_pageids(pageids: list[int], lang: str, headers: dict, timeout: int = 20) -> list[dict]:
    """
    Descarga extractos por lote usando pageids=... (más eficiente que uno a uno).
    Devuelve una lista de docs con el mismo formato.
    """
    if not pageids:
        return []

    url = wiki_api(lang)
    # MediaWiki suele aceptar bastante, pero por seguridad troceamos en lotes.
    BATCH = 50
    out = []

    for i in range(0, len(pageids), BATCH):
        batch = pageids[i : i + BATCH]
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": 1,
            "redirects": 1,
            "pageids": "|".join(str(x) for x in batch),
        }
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            if "missing" in page:
                continue
            extract = (page.get("extract") or "").strip()
            if not extract:
                continue
            pageid = page.get("pageid")
            title = page.get("title")

            doc_id = f"wiki_{lang}_{pageid}" if pageid is not None else f"wiki_{lang}_{(title or 'unknown').replace(' ', '_')}"
            out.append(
                {
                    "id": doc_id,
                    "texto": extract,
                    "fuente": "wikipedia",
                    "metadata": {
                        "title": title,
                        "lang": lang,
                        "pageid": pageid,
                        "url": (f"https://{lang}.wikipedia.org/?curid={pageid}" if pageid is not None else None),
                        "original_query": None,
                        "retrieved_at": now_utc_iso(),
                        "source": "pageids",
                    },
                }
            )
    return out


# -------------------------
# Wikipedia: categories
# -------------------------
def get_category_members(
    category_title: str,
    lang: str,
    headers: dict,
    timeout: int = 20,
    include_subcats: bool = True,
) -> tuple[list[int], list[str]]:
    """
    Devuelve:
      - pageids de artículos (ns=0)
      - títulos de subcategorías (ns=14)
    """
    url = wiki_api(lang)
    cmcontinue = None
    pageids: list[int] = []
    subcats: list[str] = []

    while True:
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": category_title,
            "cmlimit": 500,
            "cmtype": "page|subcat" if include_subcats else "page",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        members = data.get("query", {}).get("categorymembers", [])
        for m in members:
            m_ns = m.get("ns")
            if m_ns == 14:  # subcat
                title = m.get("title")
                if title:
                    subcats.append(title)
            elif m_ns == 0:  # artículos
                pid = m.get("pageid")
                if pid is not None:
                    pageids.append(int(pid))

        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break

    return pageids, subcats



def expand_categories(
    seed_categories: list[str],
    lang: str,
    headers: dict,
    timeout: int,
    ns: int,
    depth: int,
    sleep_s: float,
    max_pages: int,
) -> tuple[dict[int, set[str]], set[str]]:
    """
    Recorre categorías (BFS) hasta 'depth' y acumula pageids.
    Devuelve:
      - mapping pageid -> set(seed_categories que lo trajeron)
      - set de categorías visitadas (para debug)
    """
    visited_cats: set[str] = set()
    pageid_sources: dict[int, set[str]] = {}

    queue = deque()
    for seed in seed_categories:
        queue.append((seed, 0, seed))  # (cat_title, level, seed_origin)

    while queue:
        cat_title, level, seed_origin = queue.popleft()
        if cat_title in visited_cats:
            continue
        visited_cats.add(cat_title)

        # Si la categoría no existe exactamente, intentamos arreglarla con search
        fixed = cat_title
        # Solo intentamos fix si la API devuelve vacío; lo detectaremos abajo

        try:
            print(f"[CAT] ({lang}) level={level}/{depth} -> {cat_title}")
            pageids, subcats = get_category_members(
                category_title=fixed,
                lang=lang,
                headers=headers,
                timeout=timeout,
                include_subcats=True,
            )


            # si devuelve 0 y estamos sospechando de nombre mal puesto, intentamos search
            if not pageids and not subcats:
                # search sobre el nombre sin prefijo para aumentar acierto
                guess_query = cat_title.replace("Category:", "").replace("Categoría:", "")
                hit = search_wiki_title(guess_query, lang=lang, headers=headers, timeout=timeout)
                if hit and (hit.lower().startswith("category:") or hit.lower().startswith("categoría:")) and hit != cat_title:
                    print(f"[CAT-FIX] ({lang}) '{cat_title}' -> '{hit}'")
                    fixed = hit
                    pageids, subcats = get_category_members(
                        category_title=fixed,
                        lang=lang,
                        headers=headers,
                        timeout=timeout,
                        include_subcats=True,
                    )


            for pid in pageids:
                pageid_sources.setdefault(pid, set()).add(seed_origin)
                if len(pageid_sources) >= max_pages:
                    print(f"[CAT] ({lang}) level={level}/{depth} -> {cat_title}")
                    return pageid_sources, visited_cats

            if level < depth:
                for sc in subcats:
                    if sc not in visited_cats:
                        queue.append((sc, level + 1, seed_origin))

            time.sleep(sleep_s)

        except requests.HTTPError as e:
            print(f"[HTTP ERROR] ({lang}) categoría '{cat_title}': {e}")
        except Exception as e:
            print(f"[ERROR] ({lang}) categoría '{cat_title}': {e}")

    return pageid_sources, visited_cats


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", default="es,en", help="Idiomas separados por coma. Ej: es,en")
    parser.add_argument("--sleep", type=float, default=0.5, help="Pausa entre requests")
    parser.add_argument("--out", default=str(DATA_PROCESSED / "wiki_docs.jsonl"))
    parser.add_argument("--append", action="store_true", help="Si se indica, no sobreescribe el JSONL; añade al final")
    parser.add_argument("--user-agent", default="TuRAG-WW2/0.1 (contact: tu_email@dominio.com)")

    # Keywords
    parser.add_argument("--use-keywords", action="store_true", help="Ingresa desde wiki_keywords_{lang}.txt")
    parser.add_argument("--keywords-file-suffix", default="wiki_keywords_", help="Prefijo del fichero de keywords")

    # Categories
    parser.add_argument("--use-categories", action="store_true", help="Ingresa desde wiki_categories_{lang}.txt")
    parser.add_argument("--categories-file-suffix", default="wiki_categories_", help="Prefijo del fichero de categorías")
    parser.add_argument("--cat-depth", type=int, default=1, help="Profundidad de subcategorías (0=solo seeds)")
    parser.add_argument("--max-pages", type=int, default=5000, help="Máximo de artículos a descargar desde categorías")
    parser.add_argument("--ns", type=int, default=0, help="Namespace a incluir (0=artículos). Recomendado dejar 0.")

    args = parser.parse_args()

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)

    headers = {"User-Agent": args.user_agent}
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    # Si el usuario no especifica nada, por defecto usamos ambos (keywords + categorías)
    if not args.use_keywords and not args.use_categories:
        args.use_keywords = True
        args.use_categories = True

    # Carga keywords/categorías
    keywords_by_lang: dict[str, list[str]] = {}
    categories_by_lang: dict[str, list[str]] = {}

    for lang in langs:
        if args.use_keywords:
            kw_file = DATA_RAW / f"{args.keywords_file_suffix}{lang}.txt"
            if kw_file.exists():
                keywords_by_lang[lang] = read_keywords(kw_file)
            else:
                print(f"[WARN] No existe {kw_file}, se omite keywords para {lang}")

        if args.use_categories:
            cat_file = DATA_RAW / f"{args.categories_file_suffix}{lang}.txt"
            if cat_file.exists():
                raw_cats = read_keywords(cat_file)
                categories_by_lang[lang] = [normalize_category_title(c) for c in raw_cats]
            else:
                print(f"[WARN] No existe {cat_file}, se omite categorías para {lang}")

    # dedupe global
    seen: set[tuple[str, int | str]] = set()

    mode = "a" if args.append else "w"
    saved = 0

    with out_path.open(mode, encoding="utf-8") as f_out:
        # 1) Keywords
        for lang, keywords in keywords_by_lang.items():
            for kw in keywords:
                try:
                    print(f"[INFO] ({lang}) Descargando (keyword): {kw}")
                    doc = fetch_wiki_page(kw, lang=lang, headers=headers)
                    if not doc:
                        continue

                    key = (doc["metadata"]["lang"], doc["metadata"]["pageid"] or doc["metadata"]["title"])
                    if key in seen:
                        print(f"[SKIP] Duplicado: {key}")
                        continue
                    seen.add(key)

                    f_out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    saved += 1
                    time.sleep(args.sleep)

                except requests.HTTPError as e:
                    print(f"[HTTP ERROR] ({lang}) keyword '{kw}': {e}")
                except Exception as e:
                    print(f"[ERROR] ({lang}) keyword '{kw}': {e}")

        # 2) Categories
        for lang, seed_categories in categories_by_lang.items():
            if not seed_categories:
                continue

            pageid_sources, visited_cats = expand_categories(
                seed_categories=seed_categories,
                lang=lang,
                headers=headers,
                timeout=20,
                ns=args.ns,
                depth=args.cat_depth,
                sleep_s=args.sleep,
                max_pages=args.max_pages,
            )

            all_pageids = list(pageid_sources.keys())
            print(f"[CAT] ({lang}) Categorías visitadas: {len(visited_cats)}")
            print(f"[CAT] ({lang}) PageIDs únicos recolectados: {len(all_pageids)}")

            # Descargar por lotes
            try:
                docs = fetch_wiki_pages_by_pageids(all_pageids, lang=lang, headers=headers, timeout=20)
            except requests.HTTPError as e:
                print(f"[HTTP ERROR] ({lang}) batch pageids: {e}")
                docs = []

            # Añadir metadatos de origen (qué categoría seed lo trajo)
            # (Si un pageid venía de varias seeds, guardamos lista)
            for doc in docs:
                pid = doc["metadata"].get("pageid")
                if pid is None:
                    continue

                key = (doc["metadata"]["lang"], pid)
                if key in seen:
                    continue
                seen.add(key)

                seeds = sorted(pageid_sources.get(int(pid), set()))
                doc["metadata"]["source"] = "category"
                doc["metadata"]["seed_categories"] = seeds

                f_out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                saved += 1

    print(f"[DONE] Guardado: {out_path.resolve()}")
    print(f"[DONE] Total docs escritos en esta ejecución: {saved}")


if __name__ == "__main__":
    main()
