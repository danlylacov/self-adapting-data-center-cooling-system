#!/usr/bin/env python3
"""
Сохранить конфигурацию запущенного API двойника (GET /config) в JSON.
Использование: TWIN_BASE=http://127.0.0.1:8000 python scripts/snapshot_twin_config.py [--out path.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


def main() -> None:
    p = argparse.ArgumentParser(description="Snapshot digital twin GET /config")
    p.add_argument("--base", default=os.environ.get("TWIN_BASE", "http://127.0.0.1:8000"))
    p.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: models/_training_data/twin_config_snapshot_<ts>.json)",
    )
    args = p.parse_args()

    base = args.base.rstrip("/")
    url = f"{base}/config"
    req = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
    except Exception as e:
        print(f"Failed to GET {url}: {e}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(raw)
    out: dict = {
        "snapshot_at_utc": datetime.now(timezone.utc).isoformat(),
        "twin_base": base,
        "config": data,
    }

    out_path = Path(args.out) if args.out else None
    if out_path is None:
        root = Path(__file__).resolve().parents[1]
        d = root / "models" / "_training_data"
        d.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = d / f"twin_config_snapshot_{ts}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
