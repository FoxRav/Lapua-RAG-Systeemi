"""Minimal MCP server stub exposing /ingest and /query as MCP tools.

Left intentionally thin: the FastMCP integration will be wired in once the
core pipeline is validated. Run via ``python -m lapua_rag.mcp.server``.
"""

from __future__ import annotations

from lapua_rag.api.routes.query import _answer_service
from lapua_rag.observability import configure_logging, get_logger

_log = get_logger(__name__)


def main() -> None:
    configure_logging()
    _log.info("mcp.starting")
    try:
        from fastmcp import FastMCP
    except ImportError as exc:
        msg = "fastmcp not installed; add extras via `pip install -e .[mcp]`"
        raise RuntimeError(msg) from exc

    mcp = FastMCP("lapua-rag")

    @mcp.tool()
    def query_lapua(question: str, tenant: str = "lapua") -> dict[str, object]:
        """Ask a question against the Lapua municipal document corpus."""
        answer = _answer_service().answer(query=question, tenant=tenant)
        return answer.model_dump()

    mcp.run()


if __name__ == "__main__":  # pragma: no cover
    main()
