import asyncio
import json
import os
import uuid
import time
from typing import List, Dict, Any, Optional, Tuple

import httpx
from langchain_community.llms import HuggingFaceHub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from pydantic import BaseModel

from a2a.client import A2ACardResolver
from a2a.types import AgentCard
from host_agent.host_tools import SendMessageToRemoteAgentTool
from host_agent.remote_agent_connection import RemoteAgentConnections  # Adjusted import path

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
DEFAULT_MODEL_REPO_ID = "HuggingFaceH4/zephyr-7b-alpha"  # Changed to Zephyr


class HostLangchainAgent:
    """
    Langchain-based Host Agent responsible for routing tasks to specialist agents
    (Drift, Monitor, Scheduler) using the A2A protocol.
    """

    def __init__(self, remote_agent_addresses: List[str], llm_repo_id: str = DEFAULT_MODEL_REPO_ID):
        self.remote_agent_addresses = remote_agent_addresses
        self.remote_agent_connections: Dict[str, RemoteAgentConnections] = {}
        self.agent_cards: Dict[str, AgentCard] = {}
        self.agent_descriptions_for_prompt: str = "No agents available."
        self.llm = HuggingFaceHub(
            repo_id=llm_repo_id,
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
            model_kwargs={"temperature": 0.7, "max_new_tokens": 1024},
        )
        self.tools: List[Tool] = []
        self.agent_executor: Optional[AgentExecutor] = None
        # Per-session state, keyed by session_id
        self.session_states: Dict[str, Dict[str, Any]] = {}

    async def _initialize_remote_connections(self):
        """Asynchronously initializes connections to remote agents and fetches their cards."""
        async with httpx.AsyncClient(timeout=30) as client:
            for address in self.remote_agent_addresses:
                try:
                    card_resolver = A2ACardResolver(client, address)
                    card = await card_resolver.get_agent_card()
                    connection = RemoteAgentConnections(agent_card=card, agent_url=address)
                    self.remote_agent_connections[card.name] = connection
                    self.agent_cards[card.name] = card
                    print(f"Successfully connected to and got card for: {card.name} at {address}")
                except httpx.ConnectError as e:
                    print(f"ERROR: Failed to connect to {address}: {e}")
                except Exception as e:
                    print(f"ERROR: Failed to initialize connection for {address}: {e}")
        self._update_agent_descriptions_for_prompt()

    def _update_agent_descriptions_for_prompt(self):
        """Updates the agent descriptions string used in the prompt."""
        if not self.agent_cards:
            self.agent_descriptions_for_prompt = "No agents available. Cannot delegate tasks."
            return

        agent_info_list = []
        for name, card in self.agent_cards.items():
            agent_info_list.append(f"- Agent Name: '{name}', Description: '{card.description}'")
        self.agent_descriptions_for_prompt = "\n".join(agent_info_list)
        print(f"Updated agent descriptions for prompt:\n{self.agent_descriptions_for_prompt}")

    def _initialize_tools(self):
        """Initializes the tools for the agent."""
        if not self.remote_agent_connections:
            print("WARNING: Remote agent connections not established. SendMessageTool will not be effective.")

        # The tool needs access to the connections, but not direct access to session_states
        # Session-specific data like task_id, context_id will be passed during _arun
        self.tools = [
            SendMessageToRemoteAgentTool(remote_agent_connections=self.remote_agent_connections)
        ]

    def _initialize_agent_executor(self):
        """Initializes the Langchain agent executor."""
        if not self.tools:
            print("WARNING: No tools initialized for the agent executor.")

        # Note: The prompt will get `session_active_agent` and `session_task_history` from input variables
        # These will be populated by the `process_message` method for the current session.
        prompt_template = PromptTemplate.from_template(
            f"""
            **Role:** You are an expert Routing Delegator for a multi-agent system. 
            Your primary function is to accurately delegate user inquiries to the appropriate specialized remote agents.
            You have access to the following tools: {{tools}}
            Use the following format for your responses:

            Question: the input question you must answer
            Thought: you should always think about what to do. 
            Action: the action to take, should be one of [{{tool_names}}]
            Action Input: a JSON blob containing the necessary arguments for the chosen action. For `send_message_to_remote_agent`, this MUST include `agent_name`, `task_description`, `task_id`, `context_id`, and `session_id`. Use the provided `current_task_id`, `current_context_id`, and `current_session_id` values for these fields.
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer based on the user's request and the specialist agent's response.
            Final Answer: the final answer to the original input question, incorporating information from the specialist agent.

            **Core Directives:**
            1.  **Task Delegation:** Analyze the user's query. Based on the query and the descriptions of available agents, decide which agent is best suited. Then, use the `send_message_to_remote_agent` tool to delegate the task.
            2.  **Agent Selection:** Choose an agent from the roster below. Provide its exact `agent_name` to the tool.
            3.  **Task Description:** Formulate a clear and comprehensive `task_description` for the chosen agent. This description should include all relevant information from the user's query and any necessary prior context from the current conversation for *this specific task*.
            4.  **ID Usage:** When using `send_message_to_remote_agent`, you MUST populate the `task_id`, `context_id`, and `session_id` fields in the Action Input. Use the values provided to you as `current_task_id`, `current_context_id`, and `current_session_id` for this purpose.
            5.  **Session Context Awareness (Active Agent):**
                - If an `session_active_agent` is specified, it means a task is ongoing with that agent. 
                - If the user's new message is a follow-up or clarification for the `session_active_agent`'s current task, route the message to that *same* agent using the *current* `task_id` and `context_id`.
                - If the user's message indicates a *new, unrelated task* or explicitly asks for a different agent, then you can choose a new agent. The system will provide new `task_id` and `context_id` for such new tasks; ensure you use these new IDs.
            6.  **Conversation History:** The `session_task_history` provides a summary of previous interactions in this session. Use it to understand context, but primarily focus on the current query for delegation.
            7.  **Clarity and Conciseness:** Be clear in your `task_description`. Avoid ambiguity.
            8.  **Tool Usage:** Strictly use the `send_message_to_remote_agent` tool for delegation. Do not attempt to answer questions that require a specialist agent yourself.
            9.  **Error Handling:** If a tool call fails or an agent is unavailable, inform the user clearly.

            **Available Specialist Agents:**
            {{agent_descriptions_for_prompt}}

            **Current Conversation Context:**
            - Currently Active Agent for this task: `{{session_active_agent}}`
            - Summary of previous interactions in this task: `{{session_task_history}}`
            - Use this Task ID for the current operation: `{{current_task_id}}`
            - Use this Context ID for the current operation: `{{current_context_id}}`
            - Use this Session ID for the current operation: `{{current_session_id}}`

            Begin!

            Question: {{input}}
            Thought: {{agent_scratchpad}}
            """
        )

        self.agent_executor = AgentExecutor(
            agent=create_react_agent(llm=self.llm, tools=self.tools, prompt=prompt_template),
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,  # Handle potential output parsing errors
            max_iterations=5,  # Prevent overly long chains
        )
        print("HostLangchainAgent executor initialized.")

    @classmethod
    async def create(cls, remote_agent_addresses: List[str], llm_repo_id: str = DEFAULT_MODEL_REPO_ID) -> "HostLangchainAgent":
        """Factory method to create and asynchronously initialize an instance."""
        instance = cls(remote_agent_addresses, llm_repo_id)
        await instance._initialize_remote_connections()
        instance._initialize_tools()  # Tools depend on connections
        instance._initialize_agent_executor()  # Agent executor depends on tools and prompt (which needs descriptions)
        return instance

    def _get_or_create_session_state(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.session_states:
            self.session_states[session_id] = {
                "task_id": uuid.uuid4().hex,
                "context_id": uuid.uuid4().hex,
                "active_agent": "None",  # Name of the agent currently handling a task for this session
                "call_history": {},  # History of calls to remote agents for this session {signature: {time: ts, response: resp}}
                "task_interaction_history": [],  # Simplified history of (user_query, agent_response) for current task
            }
        return self.session_states[session_id]

    def _is_new_task_heuristic(self, current_message: str, session_state: Dict[str, Any], new_agent_selected_by_llm: Optional[str]) -> bool:
        """Determines if the current message likely starts a new task."""
        if session_state["active_agent"] == "None":
            return True  # No active agent, so it's a new task

        if new_agent_selected_by_llm and new_agent_selected_by_llm != session_state["active_agent"]:
            # LLM wants to switch agent, could be a new task or a complex multi-agent task.
            # For simplicity now, we treat agent switch as a new task context.
            # More sophisticated logic could analyze if it's a sub-task for the same overall goal.
            print(f"Heuristic: New agent '{new_agent_selected_by_llm}' selected, different from active '{session_state['active_agent']}'. Treating as new task.")
            return True

        # Add more heuristics if needed, e.g., analyzing message content for topic shifts.
        # For now, if the agent is the same, assume it's a follow-up unless LLM explicitly changes it.
        return False

    async def process_message(self, session_id: str, user_message: str) -> str:
        """Processes a user message for a given session, routes to an agent, and returns the response."""
        if not self.agent_executor:
            return "Error: Host agent executor is not initialized."

        session_state = self._get_or_create_session_state(session_id)

        # The agent's ReAct prompt needs to know about the current active agent and history for *this session*.
        # The SendMessageToRemoteAgentTool also needs task_id, context_id, and call_history for *this session*.

        # Heuristic to determine if this is a new task or a follow-up
        # This is tricky. The LLM itself will decide which agent to call. 
        # We need to guide it if it should reset context (task_id, context_id).
        # For now, the prompt tells the LLM about active_agent. If it picks a *different* agent,
        # the tool call itself won't reset task_id/context_id. We need to manage that here.
        
        # Let the LLM first decide the target agent based on the prompt that includes current active_agent.
        # The actual `SendMessageToRemoteAgentTool._arun` will receive the task_id and context_id.
        # We need to decide *before* calling `agent_executor.ainvoke` if we should use a *new* task_id/context_id.

        # This is a simplified approach: if the LLM implies a new agent or the user message seems like a new topic,
        # we can reset task_id/context_id. The LLM's choice of agent is the primary driver.
        # The prompt already guides the LLM on how to handle follow-ups vs new tasks.

        # The `send_message_to_remote_agent` tool will be invoked by the AgentExecutor.
        # We need to ensure that tool has access to the *correct* task_id, context_id, and call_history for the session.
        # This is tricky because tools are instantiated once. 
        # Solution: The tool's `_arun` method will now explicitly accept `task_id`, `context_id`, `session_id`, `call_history`.
        # The AgentExecutor needs to be able to pass these. This requires customizing the agent invocation or tool definition.
        # Langchain's default ReAct agent passes only the direct `action_input` string/dict to the tool.
        # We will pass session_id and the agent will retrieve other state.
        # The tool's `_arun` signature has been updated to accept these.
        # The `agent_executor.invoke` will pass these through `tool_input` by structuring it.

        # The prompt variables `session_active_agent` and `session_task_history` are for the LLM's reasoning.
        current_task_history_summary = "\n".join([f"User: {uh}\nAgent: {ah}" for uh, ah in session_state["task_interaction_history"][-3:]]) # Last 3 exchanges

        # Determine if this is a new task and update session state IDs if necessary.
        # The LLM will be prompted with the *current* IDs. If it's a new task, these IDs will be fresh.
        is_new_task = self._is_new_task_heuristic(user_message, session_state, None) # Pass None for new_agent_selected_by_llm initially
        if is_new_task:
            print(f"Session {session_id}: New task detected by heuristic. Resetting task_id, context_id, active_agent, task_interaction_history.")
            session_state["task_id"] = uuid.uuid4().hex
            session_state["context_id"] = uuid.uuid4().hex
            session_state["active_agent"] = "None" # LLM will determine this based on the new task
            session_state["task_interaction_history"] = []
        
        # The agent_executor will use the latest state through the prompt variables.
        input_to_executor = {
            "input": user_message,
            "agent_descriptions_for_prompt": self.agent_descriptions_for_prompt,
            "session_active_agent": session_state["active_agent"],
            "session_task_history": current_task_history_summary,
            "current_task_id": session_state["task_id"],
            "current_context_id": session_state["context_id"],
            "current_session_id": session_id
        }

        try:
            response = await self.agent_executor.ainvoke(input_to_executor)
            output = response.get("output", "No output from agent.")

            # Update task_interaction_history
            session_state["task_interaction_history"].append((user_message, output))

            # Attempt to update active_agent based on LLM's action
            # This requires parsing the intermediate steps, which can be complex.
            # A simpler way is if the LLM explicitly states its choice or if the tool returns it.
            # For now, the prompt guides the LLM. If it calls a tool with agent_name X,
            # and it was a new task, X becomes the active_agent.
            # If it was a follow-up, active_agent should remain the same.
            # The _is_new_task_heuristic and subsequent ID reset handles the "new task" case.
            # If it was a new task, session_state["active_agent"] was set to "None".
            # We need to capture the agent_name chosen by the LLM during the tool call.
            # This is not straightforward from just the final response.
            # One way is to have a callback that inspects AgentAction.
            # For now, we rely on the LLM to correctly use the session_active_agent for follow-ups.
            # If a tool was called, we can try to infer the agent_name if the tool returns it or from logs.
            # This part remains a slight challenge for robust state tracking without deeper inspection of agent trajectory.

            # A temporary placeholder for active_agent update - this needs a more robust solution
            # Potentially, the tool could return the agent_name it contacted, or a callback could capture it.
            # If the LLM successfully used the tool, we can assume the agent mentioned in the tool call is now active *for that task*.
            # This is implicitly handled if the LLM follows instructions to use the `session_active_agent` for follow-ups.
            # If `is_new_task` was true, `active_agent` is 'None'. The LLM's first tool call would define the new active agent.
            # We need to extract this from the `response` if possible (e.g. from intermediate steps if verbose mode is on and parsed).
            # For now, this state update is reliant on the LLM correctly interpreting the prompt.

            return output

        except Exception as e:
            import traceback

            print(f"Error processing message in HostLangchainAgent: {e}")
            traceback.print_exc()
            return f"Error: Could not process your request due to an internal error: {e}"


async def main():
    # Example usage:
    remote_addresses = [
        os.getenv("SCHEDULER_AGENT_URL", "http://localhost:11001"),
        os.getenv("MONITOR_AGENT_URL", "http://localhost:11002"),
        os.getenv("DRIFT_AGENT_URL", "http://localhost:11003"),
    ]
    host_agent = await HostLangchainAgent.create(remote_agent_addresses=remote_addresses)

    session_1 = "session_abc_123"

    print("\n--- Test Case 1: Schedule a new event ---")
    response1 = await host_agent.process_message(session_1, "Can you schedule a meeting for tomorrow at 2 PM about project Alpha?")
    print(f"Session {session_1} Response 1: {response1}")
    # Expected: LLM identifies Scheduler, calls send_message_to_remote_agent with new task_id, context_id, session_id.
    # session_state[session_1]["active_agent"] should become "SchedulerAgent" (or similar, based on card name).

    print("\n--- Test Case 2: Follow-up on the scheduled event ---")
    response2 = await host_agent.process_message(session_1, "Actually, can you make that 3 PM instead?")
    print(f"Session {session_1} Response 2: {response2}")
    # Expected: LLM identifies this is a follow-up for SchedulerAgent, uses existing task_id, context_id.

    print("\n--- Test Case 3: New task for a different agent (Monitor) ---")
    response3 = await host_agent.process_message(session_1, "What is the current system status?")
    print(f"Session {session_1} Response 3: {response3}")
    # Expected: LLM identifies MonitorAgent, heuristic detects new task, new task_id, context_id generated.
    # session_state[session_1]["active_agent"] should become "MonitorAgent".

    print("\n--- Test Case 4: Another session, new user ---")
    session_2 = "session_xyz_789"
    response4 = await host_agent.process_message(session_2, "Is there any model drift detected recently?")
    print(f"Session {session_2} Response 4: {response4}")
    # Expected: New session state for session_2. LLM identifies DriftAgent. New task_id, context_id.

if __name__ == "__main__":
    # Ensure HUGGINGFACEHUB_API_TOKEN is set in .env or environment
    if not HUGGINGFACEHUB_API_TOKEN:
        print("Error: HUGGINGFACEHUB_API_TOKEN not found. Please set it in your .env file or environment variables.")
    else:
        asyncio.run(main())

