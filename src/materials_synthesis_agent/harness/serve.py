"""Run the Discovery Harness API.

    python -m materials_synthesis_agent.harness.serve            # workspace ~/.materials-agent-harness
    python -m materials_synthesis_agent.harness.serve --port 8000 --workspace ./ws

The React UI (frontend/) proxies /api to this server. The chat endpoint needs ANTHROPIC_API_KEY;
every other endpoint (campaigns, state, log-result, export) works without a model.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    # Hosting platforms (Fly/Railway/Render) inject PORT and require binding 0.0.0.0; honor those as
    # defaults so a deploy needs no flags, while local runs still default to loopback.
    default_workspace = os.environ.get("HARNESS_WORKSPACE", str(Path.home() / ".materials-agent-harness"))
    default_host = os.environ.get("HARNESS_HOST", "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1")
    default_port = int(os.environ.get("PORT", "8000"))

    parser = argparse.ArgumentParser(description="Serve the Discovery Harness API")
    parser.add_argument("--workspace", default=default_workspace,
                        help="Directory for per-tenant campaign databases and prior/index stores.")
    parser.add_argument("--host", default=default_host)
    parser.add_argument("--port", type=int, default=default_port)
    args = parser.parse_args()

    import uvicorn

    from materials_synthesis_agent.harness.api import create_app

    app = create_app(args.workspace)
    print(f"Discovery Harness API on http://{args.host}:{args.port}  (workspace: {args.workspace})")
    print("Chat needs ANTHROPIC_API_KEY; all other endpoints work without it.")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
