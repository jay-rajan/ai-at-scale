"""Worker for build 6b.    python -m graph_plugin_in_temporal.worker"""
import asyncio
import logging
import os

from temporalio.client import Client
from temporalio.langgraph import LangGraphPlugin
from temporalio.worker import Worker

from agent.env import load_env
from graph_plugin_in_temporal.graph import GRAPH, build_plugin_graph
from graph_plugin_in_temporal.workflows import TASK_QUEUE, PluginInvestigationWorkflow

load_env()


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    client = await Client.connect(os.environ.get("TEMPORAL_ADDRESS", "localhost:7233"))
    plugin = LangGraphPlugin(graphs={GRAPH: build_plugin_graph()})        # registers one activity per node
    worker = Worker(client, task_queue=TASK_QUEUE, workflows=[PluginInvestigationWorkflow], plugins=[plugin])
    logging.info("Worker listening on task queue %r (pid %d)", TASK_QUEUE, os.getpid())
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
