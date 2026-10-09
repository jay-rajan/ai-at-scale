"""Runs the Temporal worker: executes workflow code and agent activities.
    python -m workflow.worker
"""
import asyncio
import logging
import os

from temporalio.client import Client
from temporalio.worker import Worker

from agent.env import load_env
from workflow.activities import agent_turn, notify_reviewer, record_event
from workflow.workflows import TASK_QUEUE, InvoiceInvestigationWorkflow

load_env()


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    client = await Client.connect(os.environ.get("TEMPORAL_ADDRESS", "localhost:7233"))
    worker = Worker(client, task_queue=TASK_QUEUE, workflows=[InvoiceInvestigationWorkflow],
                    activities=[agent_turn, record_event, notify_reviewer])
    logging.info("Worker listening on task queue %r", TASK_QUEUE)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
