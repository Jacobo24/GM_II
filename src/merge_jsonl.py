import argparse
import json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    written = 0

    with out.open("w", encoding="utf-8") as w:
        for inp in args.inputs:
            p = Path(inp)
            if not p.exists():
                print(f"[WARN] No existe: {p}")
                continue
            with p.open("r", encoding="utf-8") as r:
                for line in r:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    _id = obj.get("id")
                    if not _id or _id in seen:
                        continue
                    seen.add(_id)
                    w.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    written += 1

    print(f"[DONE] Wrote merged docs: {written}")
    print(f"[DONE] Output: {out.resolve()}")

if __name__ == "__main__":
    main()
