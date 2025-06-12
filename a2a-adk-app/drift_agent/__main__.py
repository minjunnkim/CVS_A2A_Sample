# Drift Agent Entrypoint - Refactored for Langchain with FastAPI

import logging
import os
import asyncio
import click
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

from a2a.types import (
    AgentCard,
    AgentSkill,
    AgentCapabilities,
    A2AMessage,
    Task, # Added Task
    TaskStatus, # Added TaskStatus
    TaskState, # Added TaskState
    TaskStatusUpdateEvent, # Added TaskStatusUpdateEvent
)
from a2a.server.agent_execution.context import RequestContext # Added RequestContext
from a2a.server.events.event_queue import EventQueue # Added EventQueue
from a2a.utils import new_task, new_agent_text_message # Added new_task, new_agent_text_message

# Langchain specific imports
from .adk_agent_executor import DriftAgentExecutor
from .drift_tool import DriftAnalysisTool # Import the Langchain tool

load_dotenv()
logging.basicConfig(level=logging.INFO) # Changed to INFO for better visibility
logger = logging.getLogger(__name__)

# Global variable for the agent executor
agent_executor: DriftAgentExecutor = None
agent_card_global: AgentCard = None

@click.command()
@click.option("--host", "host", default=os.getenv("DRIFT_AGENT_HOST", "localhost"))
@click.option("--port", "port", default=int(os.getenv("DRIFT_AGENT_PORT", 11003)))
@click.option("--huggingfacehub_api_token", default=os.getenv("HUGGINGFACEHUB_API_TOKEN"), help="Hugging Face Hub API Token")
def main(host: str, port: int, huggingfacehub_api_token: str):
    global agent_executor, agent_card_global

    if not huggingfacehub_api_token:
        logger.error("HUGGINGFACEHUB_API_TOKEN is not set. Please set it in your .env file or as an environment variable.")
        return

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingfacehub_api_token

    skill = AgentSkill(
        id="drift_analysis_langchain", # Updated ID for clarity
        name="Drift Analysis (Langchain)",
        description="Analyzes drift and trends across code runs using Langchain.",
        tags=["drift", "trend", "analysis", "langchain"],
        examples=["Analyze drift for the last 10 runs"],
    )
    agent_card_global = AgentCard(
        name="Drift Agent (Langchain)",
        description="Analyzes drift and trends across code runs using Langchain and Hugging Face models.",
        url=f"http://{host}:{port}/",
        version="2.0.0", # Updated version
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
        a2aVersion="1.0", # Added A2A version
        publisher="CVS Health A2A Sample Refactor", # Added publisher
        iconUrl=None, # Placeholder for icon
        tags=["refactor", "langchain", "huggingface", "drift-analysis"] # Added tags
    )

    # Instantiate the Langchain tool and executor
    drift_analysis_tool = DriftAnalysisTool()
    agent_executor = DriftAgentExecutor(mcp_tools=[drift_analysis_tool])
    
    logger.info(f"Drift Agent (Langchain) initialized with card: {agent_card_global.name}")
    logger.info(f"Starting Drift Agent server at http://{host}:{port}")

    app = FastAPI(
        title=agent_card_global.name,
        version=agent_card_global.version,
        description=agent_card_global.description,
    )

    @app.post("/", response_model=None) # Using StreamingResponse, so no specific response_model here
    async def handle_message(request: Request):
        try:
            message_data = await request.json()
            a2a_message = A2AMessage(**message_data)
            logger.info(f"Received A2A message: {a2a_message.id} for task: {a2a_message.taskId}")

            # Create a task if it's a new interaction
            task = None
            if a2a_message.taskId:
                 # In a real scenario, you'd fetch an existing task. For this example, we'll assume new task per message if not provided.
                 # This part might need adjustment based on how tasks are managed across multiple interactions.
                 # For now, let's assume the message carries enough context or implies a new task if taskId is new.
                 task = Task(id=a2a_message.taskId, contextId=a2a_message.contextId, status=TaskStatus(state=TaskState.processing))
            else:
                task = new_task(a2a_message) # Create a new task based on the message

            request_context = RequestContext(
                message=a2a_message,
                current_task=task, # Pass the created/retrieved task
                # http_request=request # If needed by executor/tools
            )
            event_queue = EventQueue()

            # Non-blocking call to the executor
            asyncio.create_task(agent_executor.execute(request_context, event_queue))

            async def event_generator():
                try:
                    while True:
                        event = await event_queue.dequeue_event()
                        if event:
                            logger.debug(f"Sending event: {event.event_type()} for task {event.taskId if hasattr(event, 'taskId') else 'N/A'}")
                            yield {"event": event.event_type(), "data": event.model_dump_json()}
                            if isinstance(event, TaskStatusUpdateEvent) and event.final:
                                logger.info(f"Task {event.taskId} marked final. Closing event stream.")
                                break
                        else:
                            # Small sleep to prevent tight loop if queue is temporarily empty
                            await asyncio.sleep(0.01) 
                except asyncio.CancelledError:
                    logger.info("Event generator cancelled.")
                except Exception as e:
                    logger.error(f"Error in event generator: {e}", exc_info=True)
                finally:
                    logger.info("Closing event_queue for this request.")
                    event_queue.close()


            return EventSourceResponse(event_generator())

        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/a2a/agent-card", response_model=AgentCard)
    async def get_agent_card():
        if not agent_card_global:
            raise HTTPException(status_code=500, detail="Agent card not initialized")
        return agent_card_global
    
    @app.get("/health")
    async def health_check():
        return JSONResponse({"status": "ok"})

    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
