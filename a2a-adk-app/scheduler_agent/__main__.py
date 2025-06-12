import logging
import os
import click
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
import json

from .adk_agent_executor import SchedulerAgentExecutor # Relative import
from .adk_agent import SchedulerLangchainAgent # Relative import for agent card details

from a2a.types import (
    AgentCard, AgentSkill, AgentCapabilities, 
    SendMessageRequest, SendMessageResponse, 
    ErrorResponse, ErrorResponseBody, 
    Message, Task # Assuming Task is the expected result type in SendMessageSuccessResponse
)

load_dotenv()
logging.basicConfig(level=logging.INFO) # Set default logging level
logger = logging.getLogger(__name__)

# Global instance of the agent executor and card
SCHEDULER_AGENT_EXECUTOR: SchedulerAgentExecutor = None
SCHEDULER_AGENT_CARD: AgentCard = None

# Helper function to create the agent card (similar to original but simplified)
def create_scheduler_agent_card(host: str, port: int) -> AgentCard:
    # These details should ideally match what the Langchain agent is designed for.
    # The skills and descriptions should align with the tools provided to SchedulerLangchainAgent.
    schedule_event_skill = AgentSkill(
        id="schedule_event",
        name="Schedule Event",
        description="Schedules a new event, appointment, or task. Examples: schedule a meeting, book a time slot.",
        tags=["scheduling", "calendar", "events"]
    )
    cancel_event_skill = AgentSkill(
        id="cancel_event",
        name="Cancel Event",
        description="Cancels a previously scheduled event. Example: cancel my meeting tomorrow.",
        tags=["scheduling", "calendar", "events"]
    )
    list_events_skill = AgentSkill(
        id="list_scheduled_events",
        name="List Scheduled Events",
        description="Lists all currently scheduled events or tasks. Example: what are my scheduled events?",
        tags=["scheduling", "calendar", "events"]
    )
    return AgentCard(
        name="SchedulerLangchainAgent", # Match the Langchain agent's conceptual name
        description="Manages scheduling, canceling, and listing of events and tasks using Langchain.",
        url=f"http://{host}:{port}/", # A2A endpoint
        version="1.1.0", # Updated version
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True), # Matches executor's streaming capability
        skills=[schedule_event_skill, cancel_event_skill, list_events_skill]
    )

@click.command()
@click.option("--host", "host", default=os.getenv("SCHEDULER_AGENT_HOST", "localhost"), help="Host for the agent server.")
@click.option("--port", "port", default=int(os.getenv("SCHEDULER_AGENT_PORT", 11001)), help="Port for the agent server.")
def main(host: str, port: int):
    """Main function to start the Scheduler Langchain Agent server."""
    global SCHEDULER_AGENT_EXECUTOR, SCHEDULER_AGENT_CARD

    logger.info(f"Initializing SchedulerLangchainAgent on {host}:{port}")

    # 1. Create Agent Card
    SCHEDULER_AGENT_CARD = create_scheduler_agent_card(host, port)
    logger.info(f"Scheduler Agent Card created: {SCHEDULER_AGENT_CARD.name}")

    # 2. Initialize the Langchain Agent and its Executor
    # The SchedulerLangchainAgent itself is instantiated within SchedulerAgentExecutor
    try:
        # SchedulerAgentExecutor now directly uses SchedulerLangchainAgent
        SCHEDULER_AGENT_EXECUTOR = SchedulerAgentExecutor(agent_card=SCHEDULER_AGENT_CARD)
        logger.info("SchedulerAgentExecutor initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize SchedulerAgentExecutor: {e}", exc_info=True)
        return # Exit if executor fails to initialize

    # 3. Setup FastAPI app
    app = FastAPI(
        title=SCHEDULER_AGENT_CARD.name,
        version=SCHEDULER_AGENT_CARD.version,
        description=SCHEDULER_AGENT_CARD.description
    )

    @app.get("/a2a/agent-card", response_model=AgentCard)
    async def get_agent_card():
        return SCHEDULER_AGENT_CARD

    @app.post("/", response_model=SendMessageResponse) # Or specific A2A path like /invoke
    async def handle_message(request: SendMessageRequest = Body(...)):
        logger.info(f"Received message request: {request.id}, params: {request.params.message.messageId}")
        if not SCHEDULER_AGENT_EXECUTOR:
            logger.error("SchedulerAgentExecutor not initialized.")
            error_response_body = ErrorResponseBody(message="Agent executor not available")
            return JSONResponse(
                status_code=500, 
                content=ErrorResponse(id=request.id, error=error_response_body).model_dump(exclude_none=True)
            )
        try:
            # The execute method of SchedulerAgentExecutor is expected to handle the SendMessageRequest
            # and return a SendMessageResponse (either success with Task or error)
            response_data = await SCHEDULER_AGENT_EXECUTOR.execute(request)
            
            # Ensure response_data is a dict suitable for SendMessageResponse model validation
            # The execute method should already return a SendMessageResponse object or a compatible dict.
            # If it returns a Pydantic model, convert to dict for FastAPI response_model validation.
            if isinstance(response_data, SendMessageResponse):
                logger.info(f"Sending success response for request {request.id}")
                # FastAPI will automatically call .model_dump_json() if response_model is set and object is returned
                return response_data 
            elif isinstance(response_data, dict): # If execute returns a dict
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
            # Ensure the error response is structured according to A2A SendMessageResponse with an error field
            return JSONResponse(
                status_code=500, 
                content=ErrorResponse(id=request.id, error=error_response_body).model_dump(exclude_none=True)
            )

    logger.info(f"Starting Uvicorn server for {SCHEDULER_AGENT_CARD.name} on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    # Ensure HUGGINGFACEHUB_API_TOKEN is set (checked by agent, but good for early warning)
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        print("CRITICAL: HUGGINGFACEHUB_API_TOKEN environment variable not set. Agent will likely fail.")
    main()
