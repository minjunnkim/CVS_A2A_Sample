\
import uuid
import time
import json
from typing import Type, Dict, Any
from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool

from a2a.types import SendMessageRequest, MessageSendParams, SendMessageSuccessResponse, Task, Part
# Ensure RemoteAgentConnections is accessible, adjust import path as necessary
# from ..adk_agent.remote_agent_connection import RemoteAgentConnections
# For now, assuming it will be in the same directory or Python path handles it.
# We will pass RemoteAgentConnections instances directly to the tool.

class SendMessageToRemoteAgentToolInput(BaseModel):
    agent_name: str = Field(description="The name of the remote agent to send the task to.")
    task_description: str = Field(description="The detailed description of the task or query for the remote agent.")
    task_id: str = Field(description="The unique ID for the current task. This should be provided by the calling agent based on the current session's task.")
    context_id: str = Field(description="The unique ID for the context of this interaction. This should be provided by the calling agent based on the current session's context.")
    session_id: str = Field(description="The unique ID for the current session. This helps in tracking and managing state across multiple interactions.")

class SendMessageToRemoteAgentTool(BaseTool):
    name: str = "send_message_to_remote_agent"
    description: str = (
        "Sends a message (task or query) to a specified remote specialist agent (e.g., Drift, Monitor, Scheduler). "
        "Use this tool to delegate tasks to the appropriate agent based on its capabilities."
    )
    args_schema: Type[BaseModel] = SendMessageToRemoteAgentToolInput
    remote_agent_connections: Dict[str, Any] # Actually Dict[str, RemoteAgentConnections]
    # session_state: Dict[str, Any] # The agent will manage this and pass relevant parts

    # def _run(self, agent_name: str, task_description: str, task_id: str, context_id: str) -> str:
    #     # Langchain tools are encouraged to be async first if underlying operations are async
    #     raise NotImplementedError("Use arun for asynchronous execution")

    async def _arun(self, agent_name: str, task_description: str, task_id: str, context_id: str, session_id: str) -> str: # Removed call_history
        if agent_name not in self.remote_agent_connections:
            return f"Error: Agent '{agent_name}' not found or connection not established."

        client_connection = self.remote_agent_connections[agent_name]

        # Simplified anti-repeat logic: The HostLangchainAgent's prompt and max_iterations
        # provide primary loop control. A very simple debounce could be added here if needed,
        # perhaps using a small internal cache in this tool instance, keyed by session_id + agent_name + task_description hash,
        # but for now, we rely on higher-level controls.
        # The original complex call_history logic is removed from this tool's direct responsibility.
        # call_signature = f"{session_id}|{agent_name}|{task_description}" 
        # now = time.time()
        # min_repeat_interval = 10  # seconds
        # if hasattr(self, "_internal_call_cache") and call_signature in self._internal_call_cache:
        #    if now - self._internal_call_cache[call_signature] < min_repeat_interval:
        #        return "__REPEATED_TOOL_CALL_DEBOUNCED_BY_TOOL__"
        # if not hasattr(self, "_internal_call_cache"):
        #    self._internal_call_cache = {}
        # self._internal_call_cache[call_signature] = now

        message_id = uuid.uuid4().hex
        payload_dict = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": task_description}],
                "messageId": message_id,
                "taskId": task_id,
                "contextId": context_id,
            }
        }
        
        message_params = MessageSendParams.model_validate(payload_dict)
        message_request = SendMessageRequest(id=message_id, params=message_params)

        try:
            send_response = await client_connection.send_message(message_request=message_request)
            
            # Update call history (agent should do this with the result)
            # session_call_history[call_signature] = {"time": now, "response_summary": "success"} # placeholder
            # call_history[session_id] = session_call_history

            if isinstance(send_response.root, SendMessageSuccessResponse) and isinstance(send_response.root.result, Task):
                task_result: Task = send_response.root.result
                response_parts = []
                if task_result.artifacts:
                    for artifact in task_result.artifacts:
                        if artifact.parts:
                            for part_data in artifact.parts:
                                # Assuming part_data is already a Part instance or dict convertible to Part
                                if isinstance(part_data, dict):
                                    part = Part(**part_data)
                                else:
                                    part = part_data # Assuming it's already a Part object

                                if part.type == "text" and part.text:
                                    response_parts.append(part.text)
                                elif part.type == "tool_code" and part.tool_code: # Changed from function_call to tool_code
                                    response_parts.append(f"[Tool Code: {part.tool_code.name}({json.dumps(part.tool_code.args)})]")
                                elif part.type == "tool_code_result" and part.tool_code_result: # Changed from function_response to tool_code_result
                                    response_parts.append(f"[Tool Result for {part.tool_code_result.name}: {json.dumps(part.tool_code_result.result)}]")
                                # Add more part types if necessary (e.g., function_call for Langchain)

                if not response_parts and task_result.status:
                     response_parts.append(f"Task status: {task_result.status.state}")


                final_response = "\\n".join(response_parts) if response_parts else "Task sent, but no immediate textual response or artifacts."
                # Storing the raw response might be too verbose for history, agent decides
                return final_response
            else:
                # Log the actual response for debugging
                # print(f"Non-success or non-task response: {send_response.model_dump_json(exclude_none=True)}")
                return f"Error: Received non-success or non-task response from {agent_name}. Response: {send_response.model_dump_json(exclude_none=True, indent=2)}"

        except Exception as e:
            # import traceback
            # traceback.print_exc()
            return f"Error sending message to {agent_name}: {str(e)}"

    # Override if a custom input model is needed for the tool method invocation by Langchain
    def _parse_input(self, tool_input: Any) -> Any:
        if isinstance(tool_input, str):
            # This case should ideally not happen if the agent prepares a dict
            # Attempt to parse if it's a JSON string, otherwise, it's an error
            try:
                tool_input = json.loads(tool_input)
            except json.JSONDecodeError:
                raise ValueError(
                    f"Invalid input: Expected a JSON string or a dictionary, got string: '{tool_input}'"
                )
        if not isinstance(tool_input, dict):
            raise ValueError(
                f"Invalid input: Expected a dictionary, got {type(tool_input)}"
            )
        return tool_input
