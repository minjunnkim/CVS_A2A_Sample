# Monitor Agent Entrypoint (skeleton)

import logging
import os
import click
import uvicorn
from dotenv import load_dotenv
from adk_agent import create_agent
from adk_agent_executor import MonitorAgentExecutor
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

load_dotenv()
logging.basicConfig()

@click.command()
@click.option("--host", "host", default="localhost")
@click.option("--port", "port", default=11002)
def main(host: str, port: int):
    skill = AgentSkill(
        id="monitor_code",
        name="Monitor Code Execution",
        description="Monitors code execution and reports status.",
        tags=["monitoring", "status"],
        examples=["Monitor example.py execution"],
    )
    agent_card = AgentCard(
        name="Monitor Agent",
        description="Monitors code execution and reports status.",
        url=f"http://{host}:{port}/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )
    adk_agent = create_agent()
    runner = Runner(
        app_name=agent_card.name,
        agent=adk_agent,
        artifact_service=InMemoryArtifactService(),
        session_service=InMemorySessionService(),
        memory_service=InMemoryMemoryService(),
    )
    agent_executor = MonitorAgentExecutor(runner, agent_card)
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor, task_store=InMemoryTaskStore()
    )
    a2a_app = A2AStarletteApplication(
        agent_card=agent_card, http_handler=request_handler
    )
    uvicorn.run(a2a_app.build(), host=host, port=port)

if __name__ == "__main__":
    main()
