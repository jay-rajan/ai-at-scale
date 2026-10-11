"""Worker for build C1.    python -m graph_in_temporal.worker"""
import asyncio
import logging
import os

from temporalio.client import Client
from temporalio.worker import Worker

from agent.env import load_env
from graph_in_temporal.activities import run_graph
from graph_in_temporal.workflows import TASK_QUEUE, GraphInvestigationWorkflow

load_env()


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    client = await Client.connect(os.environ.get("TEMPORAL_ADDRESS", "localhost:7233"))
    worker = Worker(client, task_queue=TASK_QUEUE, workflows=[GraphInvestigationWorkflow], activities=[run_graph])
    logging.info("Worker listening on task queue %r (pid %d)", TASK_QUEUE, os.getpid())
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
