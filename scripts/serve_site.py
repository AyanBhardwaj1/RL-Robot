"""Local preview of the built Vercel site: static files with clean URLs plus the Python API.

    python scripts/build_site.py && python scripts/serve_site.py   # http://127.0.0.1:3000
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import uvicorn
from fastapi.responses import FileResponse, HTMLResponse

DIST = Path(__file__).resolve().parent.parent / "site_dist"


def load_app():
    spec = importlib.util.spec_from_file_location("site_api", DIST / "api" / "index.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["site_api"] = mod
    spec.loader.exec_module(mod)
    return mod.app


app = load_app()


@app.get("/{path:path}", include_in_schema=False)
def static(path: str):
    root = DIST.resolve()
    candidates = [root / path, root / f"{path}.html", root / path / "index.html"] if path else [root / "index.html"]
    for c in candidates:
        c = c.resolve()
        if root in c.parents or c == root / "index.html":
            if c.is_file():
                return FileResponse(c)
    missing = root / "404.html"
    if missing.is_file():
        return HTMLResponse(missing.read_text(), status_code=404)
    return HTMLResponse("Not found", status_code=404)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]) if len(sys.argv) > 1 else 3000)
