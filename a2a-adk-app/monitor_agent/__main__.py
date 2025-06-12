# Monitor Agent Entrypoint (skeleton)

import logging
import os
import click
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
import json

from .adk_agent_executor import MonitorAgentExecutor # Relative import
from .adk_agent import MonitorLangchainAgent # Relative import for agent card details

from a2a.types import (
    AgentCard, AgentSkill, AgentCapabilities, 
    SendMessageRequest, SendMessageResponse, 
    ErrorResponse, ErrorResponseBody, 
    Message, Task
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MONITOR_AGENT_EXECUTOR: MonitorAgentExecutor = None
MONITOR_AGENT_CARD: AgentCard = None

def create_monitor_agent_card(host: str, port: int) -> AgentCard:
    monitor_status_skill = AgentSkill(
        id="monitor_status",
        name="Check System Status",
        description="Checks and reports the current operational status of the system or a specific component.",
        tags=["monitoring", "status", "health check"]
    )
    report_issue_skill = AgentSkill(
        id="report_issue",
        name="Report Issue",
        description="Reports a detected issue or anomaly in the system. Allows for manual issue reporting.",
        tags=["monitoring", "issues", "errors"]
    )
    return AgentCard(
        name="MonitorLangchainAgent",
        description="Monitors system status and allows reporting of issues using Langchain.",
        url=f"http://{host}:{port}/",
        version="1.1.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[monitor_status_skill, report_issue_skill]
    )

@click.command()
@click.option("--host", "host", default=os.getenv("MONITOR_AGENT_HOST", "localhost"), help="Host for the agent server.")
@click.option("--port", "port", default=int(os.getenv("MONITOR_AGENT_PORT", 11002)), help="Port for the agent server.")
def main(host: str, port: int):
    global MONITOR_AGENT_EXECUTOR, MONITOR_AGENT_CARD

    logger.info(f"Initializing MonitorLangchainAgent on {host}:{port}")

    MONITOR_AGENT_CARD = create_monitor_agent_card(host, port)
    logger.info(f"Monitor Agent Card created: {MONITOR_AGENT_CARD.name}")

    try:
        MONITOR_AGENT_EXECUTOR = MonitorAgentExecutor(agent_card=MONITOR_AGENT_CARD)
        logger.info("MonitorAgentExecutor initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize MonitorAgentExecutor: {e}", exc_info=True)
        return

    app = FastAPI(
        title=MONITOR_AGENT_CARD.name,
        version=MONITOR_AGENT_CARD.version,
        description=MONITOR_AGENT_CARD.description
    )

    @app.get("/a2a/agent-card", response_model=AgentCard)
    async def get_agent_card():
        return MONITOR_AGENT_CARD

    @app.post("/", response_model=SendMessageResponse)
    async def handle_message(request: SendMessageRequest = Body(...)):
        logger.info(f"Received message request: {request.id}, params: {request.params.message.messageId}")
        if not MONITOR_AGENT_EXECUTOR:
            logger.error("MonitorAgentExecutor not initialized.")
            error_response_body = ErrorResponseBody(message="Agent executor not available")
            return JSONResponse(
                status_code=500, 
                content=ErrorResponse(id=request.id, error=error_response_body).model_dump(exclude_none=True)
            )
        try:
            response_data = await MONITOR_AGENT_EXECUTOR.execute(request)
            if isinstance(response_data, SendMessageResponse):
                logger.info(f"Sending success response for request {request.id}")
                return response_data
            elif isinstance(response_data, dict):
                logger.info(f"Sending success response (from dict) for request {request.id}")
                return SendMessageResponse.model_validate(response_data)
            else:
                logger.error(f"Unexpected response type from executor: {type(response_data)}")
                error_response_body = ErrorResponseBody(message="Invalid response type from agent executor")
                return JSONResponse(
                    status_code=500, 
                    content=ErrorResponse(id=request.id, error=error_response_body).model_dump(exclude_none=True)
                )
        except Exception as e:
            logger.error(f"Error processing message {request.id}: {e}", exc_info=True)
            error_response_body = ErrorResponseBody(message=f"Server error: {str(e)}")
            return JSONResponse(
                status_code=500, 
                content=ErrorResponse(id=request.id, error=error_response_body).model_dump(exclude_none=True)
            )

    logger.info(f"Starting Uvicorn server for {MONITOR_AGENT_CARD.name} on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        print("CRITICAL: HUGGINGFACEHUB_API_TOKEN environment variable not set. Agent will likely fail.")
    main()
