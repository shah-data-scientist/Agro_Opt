"""
Uvicorn entrypoint for the AgroOpt FastAPI application.

Usage
-----
From the project root (Poetry shell or with the .venv activated):

    python -m src.api.main
    # or
    uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

The interactive docs are then available at:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

from __future__ import annotations

import uvicorn


def main() -> None:
    """Start the AgroOpt API server."""
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
