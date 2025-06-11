# Drift Agent Entrypoint (skeleton)

import logging
import os
import click
import uvicorn
from dotenv import load_dotenv
from adk_agent import create_agent
from adk_agent_executor import DriftAgentExecutor
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
@click.option("--port", "port", default=11003)
def main(host: str, port: int):
    skill = AgentSkill(
        id="drift_analysis",
        name="Drift Analysis",
        description="Analyzes drift and trends across code runs.",
        tags=["drift", "trend", "analysis"],
        examples=["Analyze drift for last 10 runs"],
    )
    agent_card = AgentCard(
        name="Drift Agent",
        description="Analyzes drift and trends across code runs.",
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
    agent_executor = DriftAgentExecutor(runner, agent_card)
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor, task_store=InMemoryTaskStore()
    )
    a2a_app = A2AStarletteApplication(
        agent_card=agent_card, http_handler=request_handler
    )
    uvicorn.run(a2a_app.build(), host=host, port=port)

if __name__ == "__main__":
    main()
