"""Run the FastAPI server."""
from __future__ import annotations

import os

import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("API_PORT", "8001"))
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
