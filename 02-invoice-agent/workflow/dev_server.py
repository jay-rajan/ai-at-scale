"""Starts a local Temporal dev server (downloaded automatically by the SDK) with the web UI.
State is kept in temporal-dev.db, so paused workflows survive a restart.
    python -m workflow.dev_server        UI: http://localhost:8233
"""
import asyncio
from pathlib import Path

from temporalio.testing import WorkflowEnvironment

DB = Path(__file__).resolve().parents[1] / "temporal-dev.db"


async def main() -> None:
    env = await WorkflowEnvironment.start_local(port=7233, ui=True, ui_port=8233,
                                                dev_server_database_filename=str(DB))
    print("Temporal dev server on localhost:7233, UI on http://localhost:8233  (Ctrl+C to stop)")
    try:
        await asyncio.Event().wait()
    finally:
        await env.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
