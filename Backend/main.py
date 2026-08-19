"""Run the BBB prediction API.

The service is mounted behind the Replit ``/api`` proxy prefix.  The FastAPI
application itself accepts both direct routes and prefixed routes so it also
works when run locally.
"""

from __future__ import annotations

import os

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "bbb_api:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )