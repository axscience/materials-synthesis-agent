"""Run the Discovery Harness API.

    python -m materials_synthesis_agent.harness.serve            # workspace ~/.materials-agent-harness
    python -m materials_synthesis_agent.harness.serve --port 8000 --workspace ./ws

The React UI (frontend/) proxies /api to this server. The chat endpoint needs ANTHROPIC_API_KEY;
every other endpoint (campaigns, state, log-result, export) works without a model.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Discovery Harness API")
    parser.add_argument("--workspace", default=str(Path.home() / ".materials-agent-harness"),
                        help="Directory for campaign databases and the shared prior/index stores.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    from materials_synthesis_agent.harness.api import create_app

    app = create_app(args.workspace)
    print(f"Discovery Harness API on http://{args.host}:{args.port}  (workspace: {args.workspace})")
    print("Chat needs ANTHROPIC_API_KEY; all other endpoints work without it.")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
