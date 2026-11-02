"""ASGI middleware loaded through the ``middleware`` option of ``vllm serve``."""

import logging
import os
import time

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from vllm.entrypoints.serve.middleware.authenticate import AuthenticationMiddleware

# A child of "vllm", so records go through the handler and PrivacyFilter.
audit_logger = logging.getLogger("vllm.privacy_audit")

# Kubelet probes carry no token.
PUBLIC_PATHS = frozenset({"/health"})


class RequireTokenMiddleware(AuthenticationMiddleware):
    """Requires the API token on every route except PUBLIC_PATHS.

    vLLM's own middleware only guards ``/v1``, ``/v2``, ``/inference`` and
    ``/cohere``, which leaves routes such as ``/invocations`` (full inference),
    ``/tokenize`` and ``/pooling`` open to anyone who can reach the service.
    """

    def __init__(self, app: ASGIApp) -> None:
        token = os.environ.get("VLLM_API_KEY")
        if not token:
            raise RuntimeError("VLLM_API_KEY must be set")
        super().__init__(app, tokens=[token])

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] not in ("http", "websocket")
            or scope.get("method") == "OPTIONS"
            or scope["path"] in PUBLIC_PATHS
            or self.verify_token(Headers(scope=scope))
        ):
            await self.app(scope, receive, send)
            return

        response = JSONResponse({"error": "Unauthorized"}, status_code=401)
        await response(scope, receive, send)


class AuditLogMiddleware:
    """Records that a request happened, without its content.

    This replaces the uvicorn access log, which prints the query string. The
    path of an unmatched route is client-controlled text, so it is not logged.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        status = 500

        async def send_with_status(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_with_status)
        finally:
            audit_logger.info(
                "API request: method=%s path=%s status=%d duration_ms=%.1f",
                scope["method"],
                "<unmatched>" if status == 404 else scope["path"],
                status,
                (time.perf_counter() - started) * 1000,
            )
