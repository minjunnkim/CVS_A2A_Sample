import json
import uuid
from typing import List
import httpx
from typing import Any
import asyncio
import os
import time

from google.adk import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext
from .remote_agent_connection import RemoteAgentConnections, TaskUpdateCallback
from a2a.client import A2ACardResolver

from a2a.types import (
    SendMessageResponse,
    SendMessageRequest,
    MessageSendParams,
    SendMessageSuccessResponse,
    Task,
    Part,
    AgentCard,
)

from dotenv import load_dotenv

load_dotenv()


def convert_part(part: Part, tool_context: ToolContext):
    # Currently only support text parts
    if part.type == "text":
        return part.text

    return f"Unknown type: {part.type}"


def convert_parts(parts: list[Part], tool_context: ToolContext):
    rval = []
    for p in parts:
        rval.append(convert_part(p, tool_context))
    return rval


def create_send_message_payload(
    text: str, task_id: str | None = None, context_id: str | None = None
) -> dict[str, Any]:
    """Helper function to create the payload for sending a task."""
    payload: dict[str, Any] = {
        "message": {
            "role": "user",
            "parts": [{"type": "text", "text": text}],
            "messageId": uuid.uuid4().hex,
        },
    }

    if task_id:
        payload["message"]["taskId"] = task_id

    if context_id:
        payload["message"]["contextId"] = context_id
    return payload


class RoutingAgent:
    """The Routing agent.

    This is the agent responsible for choosing which remote seller agents to send
    tasks to and coordinate their work.
    """

    # __init__ becomes synchronous and simple
    def __init__(
        self,
        task_callback: TaskUpdateCallback | None = None,
    ):
        self.task_callback = task_callback
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        self.agents: str = ""

    # Asynchronous part of initialization
    async def _async_init_components(self, remote_agent_addresses: List[str]):
        # Use a single httpx.AsyncClient for all card resolutions for efficiency
        async with httpx.AsyncClient(timeout=30) as client:
            for address in remote_agent_addresses:
                card_resolver = A2ACardResolver(client, address) # Constructor is sync
                try:
                    card = await card_resolver.get_agent_card() # get_agent_card is async
                    
                    remote_connection = RemoteAgentConnections(
                        agent_card=card, agent_url=address
                    )
                    self.remote_agent_connections[card.name] = remote_connection
                    self.cards[card.name] = card
                except httpx.ConnectError as e:
                    print(f"ERROR: Failed to get agent card from {address}: {e}")
                except Exception as e: # Catch other potential errors
                    print(f"ERROR: Failed to initialize connection for {address}: {e}")
        
        # Populate self.agents using the logic from original __init__ (via list_remote_agents)
        agent_info = []
        for agent_detail_dict in self.list_remote_agents(): 
            agent_info.append(json.dumps(agent_detail_dict))
        self.agents = "\n".join(agent_info)

    # Class method to create and asynchronously initialize an instance
    @classmethod
    async def create(
        cls,
        remote_agent_addresses: List[str],
        task_callback: TaskUpdateCallback | None = None,
    ):
        instance = cls(task_callback)
        await instance._async_init_components(remote_agent_addresses)
        return instance

    def create_agent(self) -> Agent:
        return Agent(
            model="gemini-2.5-flash-preview-04-17",
            name="Routing_agent",
            instruction=self.root_instruction,
            before_model_callback=self.before_model_callback,
            description=(
                "This Routing agent orchestrates the decomposition of the user asking for scheduling, monitoring, or drift analysis tasks."
            ),
            tools=[
                self.send_message,
            ],
        )

    def root_instruction(self, context: ReadonlyContext) -> str:
        current_agent = self.check_active_agent(context)
        return f"""
        **Role:** You are an expert Routing Delegator. Your primary function is to accurately delegate user inquiries regarding scheduling, monitoring, or drift analysis to the appropriate specialized remote agents.

        **Core Directives:**
        * **Task Delegation:** Utilize the `send_message` function to assign actionable tasks to remote agents.
        * **Session Context Awareness:** If there is an `active_agent` in the session state, always route follow-up or clarification messages to that agent unless the user clearly changes the topic or requests a different agent.
        * **No Context Mixing:** Do not interpret follow-up clarifications as new tasks for other agents. Only switch agents if the user explicitly changes the topic.
        * **Strict Follow-up Routing:** Under no circumstances should a follow-up or clarification response be routed to a different agent than the current `active_agent`, unless the user clearly initiates a new, unrelated request.
        * **Loop Prevention:** Never repeatedly ask the same question or re-invoke the same tool/function in a loop. If a tool/function call has already been made in this context, do not call it again. If you detect repeated or circular tool calls, stop and inform the user.
        * **Contextual Awareness for Remote Agents:** If a remote agent repeatedly requests user confirmation, assume it lacks access to the full conversation history. In such cases, enrich the task description with all necessary contextual information relevant to that specific agent.
        * **Autonomous Agent Engagement:** Never seek user permission before engaging with remote agents. If multiple agents are required to fulfill a request, connect with them directly without requesting user preference or confirmation.
        * **Transparent Communication:** Always present the complete and detailed response from the remote agent to the user.
        * **User Confirmation Relay:** If a remote agent asks for confirmation, and the user has not already provided it, relay this confirmation request to the user.
        * **Focused Information Sharing:** Provide remote agents with only relevant contextual information. Avoid extraneous details.
        * **No Redundant Confirmations:** Do not ask remote agents for confirmation of information or actions.
        * **Tool Reliance:** Strictly rely on available tools to address user requests. Do not generate responses based on assumptions. If information is insufficient, request clarification from the user.
        * **Prioritize Recent Interaction:** Focus primarily on the most recent parts of the conversation when processing requests.
        * **Active Agent Prioritization:** If an active agent is already engaged, route subsequent related requests to that agent using the appropriate task update tool.
        * **New Task Context Reset:** If the user initiates a new, unrelated request (e.g., changes topic or agent), reset the session state (`active_agent`, `_call_history`, `task_id`, `context_id`) before routing the message. This ensures no context-mixing or leakage between unrelated tasks.

        **Agent Roster:**
        * Available Agents: `{self.agents}`
        * Currently Active Agent: `{current_agent["active_agent"]}`
        """

    def check_active_agent(self, context: ReadonlyContext):
        state = context.state
        if (
            "session_id" in state
            and "session_active" in state
            and state["session_active"]
            and "active_agent" in state
        ):
            return {"active_agent": f"{state['active_agent']}"}
        return {"active_agent": "None"}

    def before_model_callback(self, callback_context: CallbackContext, llm_request):
        state = callback_context.state
        if "session_active" not in state or not state["session_active"]:
            if "session_id" not in state:
                state["session_id"] = str(uuid.uuid4())
            state["session_active"] = True

    def list_remote_agents(self):
        """List the available remote agents you can use to delegate the task."""
        if not self.cards: 
            return []

        remote_agent_info = []
        for card in self.cards.values():
            print(f"Found agent card: {card.model_dump(exclude_none=True)}")
            print("=" * 100)
            remote_agent_info.append(
                {"name": card.name, "description": card.description}
            )
        return remote_agent_info

    
    def _is_new_task(self, agent_name: str, tool_context: ToolContext) -> bool:
        """Heuristic to detect if the user is starting a new, unrelated task."""
        state = tool_context.state
        # If there is no active agent, it's a new task
        if "active_agent" not in state or not state["active_agent"]:
            return True
        # If the agent being called is different from the current active agent, treat as new task
        if state["active_agent"] != agent_name:
            return True
        # Optionally, could add more heuristics here (e.g., user input analysis)
        return False

    async def send_message(
        self, agent_name: str, task: str, tool_context: ToolContext
    ):
        """Sends a task to remote seller agent

        This will send a message to the remote agent named agent_name.

        Args:
            agent_name: The name of the agent to send the task to.
            task: The comprehensive conversation context summary
                and goal to be achieved regarding user inquiry and purchase request.
            tool_context: The tool context this method runs in.

        Yields:
            A dictionary of JSON data.
        """
        # Prevent infinite recursion: if this context has already performed a tool/function call, abort
        if getattr(tool_context, "_chained_call", False):
            return "[SIMULATION] Tool/function call chaining limit reached. No further calls performed."
        # Prevent repeated tool calls for the same (agent_name, task) in the same session
        state = tool_context.state

        # --- Context-mixing prevention: reset session state for new, unrelated tasks ---
        if self._is_new_task(agent_name, tool_context):
            # Reset session state for new task
            state["active_agent"] = agent_name
            state["_call_history"] = {}
            state["task_id"] = str(uuid.uuid4())
            state["context_id"] = str(uuid.uuid4())
        # Otherwise, preserve session state for follow-ups

        call_signature = f"{agent_name}|{task}"
        now = time.time()
        # Use a dict to store timestamps and responses for each call_signature
        if "_call_history" not in state or not isinstance(state["_call_history"], dict):
            state["_call_history"] = {}
        min_repeat_interval = 10  # seconds
        call_entry = state["_call_history"].get(call_signature)
        if call_entry is not None:
            last_call_time = call_entry.get("time")
            if last_call_time is not None and now - last_call_time < min_repeat_interval:
                # Return a special marker so the host agent can move on
                return "__REPEATED_TOOL_CALL__"

        state["active_agent"] = agent_name
        client = self.remote_agent_connections[agent_name]

        if not client:
            raise ValueError(f"Client not available for {agent_name}")
        if "task_id" in state:
            taskId = state["task_id"]

        else:
            taskId = str(uuid.uuid4())
        task_id = taskId
        sessionId = state["session_id"]
        if "context_id" in state:
            context_id = state["context_id"]
        else:
            context_id = str(uuid.uuid4())

        messageId = ""
        metadata = {}
        if "input_message_metadata" in state:
            metadata.update(**state["input_message_metadata"])
            if "message_id" in state["input_message_metadata"]:
                messageId = state["input_message_metadata"]["message_id"]
        if not messageId:
            messageId = str(uuid.uuid4())

        payload = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": task}], # Use the 'task' argument here
                "messageId": messageId,
            },
        }

        if task_id:
            payload["message"]["taskId"] = task_id

        if context_id:
            payload["message"]["contextId"] = context_id
        
        message_request = SendMessageRequest(
            id=messageId, params=MessageSendParams.model_validate(payload)
        )
        send_response: SendMessageResponse = await client.send_message( message_request= message_request)
        print("send_response", send_response)

        if not isinstance(send_response.root, SendMessageSuccessResponse):
            print("received non-success response. Aborting get task ")
            return

        if not isinstance(send_response.root.result, Task):
            print("received non-task response. Aborting get task ")
            return

        response = send_response
        if hasattr(response, "root"):
            content = response.root.model_dump_json(exclude_none=True)
        else:
            content = response.model_dump(mode="json", exclude_none=True)

        resp = []
        function_call_part = None
        json_content = json.loads(content)
        print(json_content)
        if json_content.get("result") and json_content["result"].get("artifacts"):
            for artifact in json_content["result"]["artifacts"]:
                if artifact.get("parts"):
                    for part in artifact["parts"]:
                        if part.get("type") in ("function_call", "tool_call"):
                            function_call_part = part
                        resp.append(part)
        # --- If a function/tool call is present, attempt to perform it (one level only) ---
        if function_call_part:
            # Extract function/tool name and arguments
            func_name = function_call_part.get("function_name") or function_call_part.get("name")
            func_args = function_call_part.get("parameters") or function_call_part.get("args") or {}
            # Mark context to prevent further chaining
            setattr(tool_context, "_chained_call", True)
            # Attempt to call the function/tool if it exists on this agent
            func = getattr(self, func_name, None)
            if callable(func):
                try:
                    # If arguments are a dict, unpack; else, pass as is
                    if isinstance(func_args, dict):
                        result = await func(**func_args, tool_context=tool_context)
                    else:
                        result = await func(func_args, tool_context=tool_context)
                    return f"[SIMULATION] Tool/function '{func_name}' was called. Result: {result}"
                except Exception as e:
                    return f"[SIMULATION] Error calling tool/function '{func_name}': {e}"
            else:
                return f"[SIMULATION] Tool/function '{func_name}' not found on RoutingAgent."
        # Store the response for repeated calls
        state["_call_history"][call_signature] = {"time": now, "response": resp}
        return resp


def _get_initialized_routing_agent_sync():
    """Synchronously creates and initializes the RoutingAgent."""
    async def _async_main():
        routing_agent_instance = await RoutingAgent.create(
            remote_agent_addresses=[
                os.getenv("SCHEDULER_AGENT_URL", "http://localhost:11001"),
                os.getenv("MONITOR_AGENT_URL", "http://localhost:11002"),
                os.getenv("DRIFT_AGENT_URL", "http://localhost:11003"),
            ]
        )
        return routing_agent_instance.create_agent()

    try:
        return asyncio.run(_async_main())
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            print(f"Warning: Could not initialize RoutingAgent with asyncio.run(): {e}. "
                  "This can happen if an event loop is already running (e.g., in Jupyter). "
                  "Consider initializing RoutingAgent within an async function in your application.")
        raise


root_agent = _get_initialized_routing_agent_sync()
