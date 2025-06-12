# Scheduler Agent Executor (skeleton)

import asyncio
import logging
from typing import Any, AsyncIterator, Dict
import json

# Imports from the A2A framework - ensure these are available in your environment
from a2a.server.agent_execution import AgentExecutor as A2AExecutorBase # Renamed to avoid clash
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState, Part, TextPart

# Import the custom Langchain agent
from .adk_agent import SchedulerLangchainAgent #, create_agent # create_agent might not be used directly by executor

logger = logging.getLogger(__name__)

class SchedulerAgentExecutor(A2AExecutorBase):
    """Executes scheduling tasks using a Langchain agent and reports status via A2A protocol."""

    def __init__(self, agent: SchedulerLangchainAgent, card: Any): 
        self.agent = agent
        self._card = card 

    async def _process_request(
        self, input_text: str, session_id: str, task_updater: TaskUpdater
    ):
        """Processes the request by streaming from the Langchain agent and updating the task."""
        final_response_artifact = None
        agent_produced_output = False

        try:
            async for log_entry in self.agent.stream(input_text):
                agent_produced_output = True 

                if log_entry["type"] == "agent" and log_entry["name"] == "AgentAction":
                    action_data = log_entry["data"]
                    tool_name = action_data.get('tool')
                    tool_input = action_data.get('tool_input')
                    logger.info(f"Agent action: tool={tool_name}, input={tool_input}")
                    task_updater.update_status(TaskState.working, message=f"Agent is using tool: {tool_name}")

                elif log_entry["type"] == "tool" and log_entry["name"] == "ToolExecution":
                    tool_data = log_entry["data"]
                    tool_name = tool_data.get('tool_name')
                    logger.info(f"Tool execution: tool={tool_name} completed.")
                    task_updater.update_status(TaskState.working, message=f"Tool {tool_name} executed.")

                elif log_entry["type"] == "agent" and log_entry["name"] == "AgentFinish":
                    finish_data = log_entry["data"]
                    final_output_str = finish_data.get("output") 
                    logger.info(f"Agent finished. Raw output from agent: {final_output_str}")

                    if final_output_str:
                        final_response_artifact = TextPart(text=final_output_str)
                        
                        try:
                            tool_response_json = json.loads(final_output_str)
                            confirmation_msg = str(tool_response_json.get("confirmation_message", "")).lower()
                            details_msg = str(tool_response_json.get("details", "")).lower()

                            if ("please provide the email" in confirmation_msg or 
                                "please provide the email" in details_msg):
                                final_response_artifact = TextPart(
                                    text=json.dumps({
                                        "status": "user_prompt",
                                        "details": "Please provide the email address for scheduling confirmation."
                                    })
                                )
                        except json.JSONDecodeError:
                            logger.warning(f"Final output from agent was not the expected JSON from a tool: {final_output_str}")
                    else:
                        logger.warning("AgentFinish event had no 'output' data.")
                        final_response_artifact = TextPart(text="Agent finished without providing output.")
                    break 

                elif log_entry["type"] == "llm":
                    task_updater.update_status(TaskState.working, message="Agent is processing...")

            if not agent_produced_output:
                logger.warning("Agent stream completed without yielding any log entries.")
                task_updater.fail(message="Agent did not produce any output.")
                return

            if final_response_artifact:
                task_updater.add_artifact([final_response_artifact])
                task_updater.complete()
            else:
                logger.warning("Agent stream finished, but no final response artifact was generated.")
                task_updater.fail(message="Agent did not produce a conclusive final response.")

        except Exception as e:
            logger.exception(f"Error during scheduler agent processing: {e}") # Use logger.exception for stack trace
            task_updater.fail(message=f"An error occurred in scheduler agent: {str(e)}")

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task:
            updater.submit()
        updater.start_work()

        raw_input_parts = []
        for part_wrapper in context.message.parts:
            part = part_wrapper.root
            if isinstance(part, TextPart):
                raw_input_parts.append(part.text)
        
        input_text = " ".join(raw_input_parts).strip()

        if not input_text:
            # The scheduler agent's prompt asks it to clarify if info is missing.
            # Sending a generic prompt might be okay, or a specific "what do you want to schedule?"
            input_text = "What would you like to do with the scheduler? (e.g., schedule an event, cancel an event, list events)"
            # Alternatively, fail if input is truly empty and critical.
            # logger.warning(f"Task {context.task_id}: No input text for scheduler agent.")
            # updater.add_artifact([TextPart(text="Error: No input provided for the scheduling task.")])
            # updater.fail(message="Missing input for scheduler agent.")
            # return

        logger.info(f"Executing scheduler agent for task {context.task_id} with input: '{input_text[:100]}...'")
        
        await self._process_request(
            input_text=input_text,
            session_id=context.context_id, 
            task_updater=updater,
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        logger.info(f"Cancellation requested for task {context.task_id} of scheduler agent.")
        updater.cancel()
