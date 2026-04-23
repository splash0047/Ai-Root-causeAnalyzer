"""
AI Root Cause Analyzer — Request Tracer / Logger
Middleware that logs every request with timing, status, and structured traces.
"""
import time
import logging
import uuid
from datetime import datetime
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# Configure structured logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rca.tracer")


class TracerMiddleware(BaseHTTPMiddleware):
    """Middleware that traces every request with timing and structured logs."""

    async def dispatch(self, request: Request, call_next):
        trace_id = str(uuid.uuid4())[:8]
        start = time.time()
        method = request.method
        path = request.url.path
        query = str(request.url.query) if request.url.query else ""

        logger.info(f"[{trace_id}] ▶ {method} {path}{'?' + query if query else ''}")

        try:
            response = await call_next(request)
            elapsed = round((time.time() - start) * 1000, 1)
            status = response.status_code
            status_icon = "✅" if status < 400 else "⚠️" if status < 500 else "❌"

            logger.info(
                f"[{trace_id}] {status_icon} {status} │ {elapsed}ms │ {method} {path}"
            )

            # Add trace headers to response
            response.headers["X-Trace-ID"] = trace_id
            response.headers["X-Response-Time"] = f"{elapsed}ms"
            return response

        except Exception as e:
            elapsed = round((time.time() - start) * 1000, 1)
            logger.error(
                f"[{trace_id}] ❌ ERROR │ {elapsed}ms │ {method} {path} │ {str(e)}"
            )
            raise
