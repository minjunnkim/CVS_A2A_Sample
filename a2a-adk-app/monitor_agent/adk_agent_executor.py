# Monitor Agent Executor (skeleton)

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

# Imports from Langchain
from langchain_core.messages import AIMessage # For type checking if needed

# Import the custom Langchain agent
from .adk_agent import MonitorLangchainAgent, create_agent

logger = logging.getLogger(__name__)

class MonitorAgentExecutor(A2AExecutorBase):
    """Executes monitoring tasks using a Langchain agent and reports status via A2A protocol."""

    def __init__(self, agent: MonitorLangchainAgent, card: Any): # card might be for UI
        self.agent = agent
        self._card = card # Retain if used by the A2A framework or UI

    async def _process_request(
        self, input_text: str, session_id: str, task_updater: TaskUpdater
    ):
        """Processes the request by streaming from the Langchain agent and updating the task."""
        final_response_artifact = None
        agent_produced_output = False

        try:
            async for log_entry in self.agent.stream(input_text):
                agent_produced_output = True # Mark that the agent is processing

                if log_entry["type"] == "agent" and log_entry["name"] == "AgentAction":
                    action_data = log_entry["data"]
                    tool_name = action_data.get('tool')
                    tool_input = action_data.get('tool_input')
                    logger.info(f"Agent action: tool={tool_name}, input={tool_input}")
                    task_updater.update_status(TaskState.working, message=f"Agent is using tool: {tool_name}")

                elif log_entry["type"] == "tool" and log_entry["name"] == "ToolExecution":
                    tool_data = log_entry["data"]
                    tool_name = tool_data.get('tool_name')
                    tool_output = tool_data.get('output') # Output from the tool
                    logger.info(f"Tool execution: tool={tool_name}, output_length={len(tool_output) if tool_output else 0}")
                    task_updater.update_status(TaskState.working, message=f"Tool {tool_name} executed.")
                    # The agent will observe this tool output and continue.

                elif log_entry["type"] == "agent" and log_entry["name"] == "AgentFinish":
                    finish_data = log_entry["data"]
                    final_output_str = finish_data.get("output")
                    logger.info(f"Agent finished. Raw output: {final_output_str}")

                    if final_output_str:
                        # The agent is prompted to return a JSON string that includes an 'output' field
                        # which itself should be a JSON string matching RESPONSE_FORMAT_INSTRUCTION_STR.
                        try:
                            # First, parse the outer JSON from the agent (e.g., {\"thought\": \"...\", \"output\": \"{\\\"status\\\":...}\"})
                            agent_response_json = json.loads(final_output_str)
                            actual_output_payload_str = agent_response_json.get("output")

                            if actual_output_payload_str:
                                # Now, this actual_output_payload_str is what we expect to be the final JSON
                                # matching RESPONSE_FORMAT_INSTRUCTION_STR.
                                # We can validate it or use it directly.
                                # For the artifact, we'll send this payload string.
                                final_response_artifact = TextPart(text=actual_output_payload_str)
                                
                                # Optional: Parse the inner JSON for specific checks like "missing target"
                                inner_payload_json = json.loads(actual_output_payload_str)
                                details_text = str(inner_payload_json.get("details", "")).lower()
                                if ("target" in details_text and "missing" in details_text) or \
                                   ("please specify" in details_text and "target" in details_text):
                                    final_response_artifact = TextPart(
                                        text=json.dumps({
                                            "status": "user_prompt",
                                            "details": "Please specify the system or component you want to monitor or report an issue for."
                                        })
                                    )
                            else:
                                # Agent's final JSON didn't have an "output" field as expected
                                logger.warning("AgentFinish output missing 'output' field in its JSON.")
                                final_response_artifact = TextPart(text=f"Agent finished but response format was unexpected: {final_output_str}")
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse JSON from agent's final output: {e}. Output: {final_output_str}")
                            final_response_artifact = TextPart(text=f"Agent finished, but output was not valid JSON: {final_output_str}")
                    else:
                        logger.warning("AgentFinish event had no output data.")
                        final_response_artifact = TextPart(text="Agent finished without providing output.")
                    break # Agent has finished, exit stream processing

                elif log_entry["type"] == "llm":
                    # LLM is thinking or generating a response/action
                    task_updater.update_status(TaskState.working, message="Agent is processing...")

            if not agent_produced_output:
                logger.warning("Agent stream completed without yielding any log entries.")
                task_updater.fail(message="Agent did not produce any output.")
                return

            if final_response_artifact:
                task_updater.add_artifact([final_response_artifact])
                task_updater.complete()
            else:
                # This case implies the stream ended, but AgentFinish wasn't properly processed or didn't occur.
                logger.warning("Agent stream finished, but no final response artifact was generated.")
                task_updater.fail(message="Agent did not produce a conclusive final response.")

        except Exception as e:
            logger.exception(f"Error during agent processing: {e}")
            task_updater.fail(message=f"An error occurred: {str(e)}")

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task: # Should always be present if execute is called
            updater.submit() # Submit the task if not already
        updater.start_work()

        # Construct input_text for the Langchain agent from A2A message parts
        # This needs to be robust based on how users will send requests.
        # For now, concatenate text parts.
        # More sophisticated parsing might be needed if specific fields (target, issue) are expected.
        
        raw_input_parts = []
        for part_wrapper in context.message.parts:
            part = part_wrapper.root # Access the actual Part model
            if isinstance(part, TextPart):
                raw_input_parts.append(part.text)
        
        input_text = " ".join(raw_input_parts).strip()

        if not input_text:
            # Fallback or attempt to interpret structured input if provided differently
            # This is a placeholder for more complex input handling logic.
            # For example, if the user sends a JSON in a TextPart, you might parse it here.
            # For now, if no text, send a generic prompt.
            # The agent itself is prompted to ask for clarification if needed.
            input_text = "Check system status or report an issue if details are provided in context."
            # A better default might be to fail if input is truly empty and required.
            # However, the monitor agent might have a "general status" capability.

        if not input_text: # Final check if input is still empty
            logger.warning(f"Task {context.task_id}: No input text could be derived for the monitor agent.")
            updater.add_artifact([TextPart(text="Error: No input provided for the monitoring task.")])
            updater.fail(message="Missing input for monitoring agent.")
            return

        logger.info(f"Executing monitor agent for task {context.task_id} with input: '{input_text[:100]}...'")
        
        await self._process_request(
            input_text=input_text,
            session_id=context.context_id, # context_id can serve as a session identifier if needed
            task_updater=updater,
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        logger.info(f"Cancellation requested for task {context.task_id} of monitor agent.")
        # Langchain agent cancellation is not straightforward for an ongoing stream.
        # The primary effect here is to update the task status.
        # If the agent execution runs in a separate asyncio.Task, it could be cancelled.
        updater.cancel()

# Factory function to create the executor, if needed by the main application
# def create_monitor_executor(card: Any) -> MonitorAgentExecutor:
#     langchain_agent = create_agent() # from .adk_agent.py
#     return MonitorAgentExecutor(agent=langchain_agent, card=card)
