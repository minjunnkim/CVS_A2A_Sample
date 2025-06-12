# Drift Agent refactored for Langchain with Hugging Face model

import logging
import asyncio
from collections.abc import AsyncIterable
from typing import Any, Literal, List
import httpx # Keep for now, might be used by tools
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.runnables.config import RunnableConfig
from langchain_community.llms import HuggingFaceHub # Example, choose appropriate
from langchain.agents import AgentExecutor, create_react_agent # or other agent types
from langchain_core.prompts import PromptTemplate # For creating prompts
# from langgraph.checkpoint.memory import MemorySaver # If using langgraph
from pydantic import BaseModel

# Import the new Langchain tools
from .drift_tool import DriftAnalysisTool, DetectDriftSpecificTool, ResetDriftSpecificTool, AnalyzeRunDataSpecificTool

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

# memory = MemorySaver() # If using langgraph

class ResponseFormat(BaseModel):
    """Respond to the user in this format."""
    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str

class DriftLangchainAgent:
    """Drift Analysis Agent using Langchain."""

    SYSTEM_INSTRUCTION = """
        **Role:** You are a dedicated Drift Analysis Agent. Your sole responsibility is to analyze data or model drift and report findings.

        **Behavioral Directives:**
        * Only respond to requests related to drift analysis, drift detection, or drift reporting.
        * Never answer questions or perform actions outside of drift analysis and reporting.
        * Never provide generic, pending, or placeholder responses. Only emit a final, user-facing result when drift analysis is complete or a finding is available.
        * If required information is missing, prompt the user for it and wait for their response before proceeding.
        * Do not attempt to answer questions about scheduling, monitoring, or any other domain.
        * Do not ask the user for permission to perform your core drift analysis duties—just proceed as required by the task.
        * Utilize the provided tools for drift analysis.
    """

    RESPONSE_FORMAT_INSTRUCTION: str = (
        'Select status as "completed" if the request is fully addressed and no further input is needed. '
        'Select status as "input_required" if you need more information from the user or are asking a clarifying question. '
        'Select status as "error" if an error occurred or the request cannot be fulfilled.'
    )

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"] # A2A specific, might not be directly used by Langchain agent

    def __init__(self, mcp_tools: List[Any] = None): # mcp_tools are now Langchain tools
        logger.info("Initializing DriftLangchainAgent...")
        try:
            # Replace with your chosen Hugging Face model
            # This might require HUGGINGFACEHUB_API_TOKEN if using a model from the Hub that's not free
            # For truly API-key-free, you'd use a locally downloaded model with HuggingFacePipeline or similar
            self.model = HuggingFaceHub(
                repo_id="mistralai/Mistral-7B-Instruct-v0.2", # Example
                # Ensure HUGGINGFACEHUB_API_TOKEN is set in environment if needed by model/Hub
                model_kwargs={"temperature": 0.7, "max_length": 1024} # Increased max_length
            )
            logger.info("HuggingFaceHub model initialized successfully.")
        except Exception as e:
            logger.error(
                f"Failed to initialize HuggingFaceHub model: {e}", exc_info=True
            )
            raise

        # Initialize with specific Langchain tools derived from drift_server.py
        # If mcp_tools are passed, they might be other generic tools.
        # For this agent, we primarily care about its specialized drift tools.
        self.specialized_tools = [
            DetectDriftSpecificTool(), 
            ResetDriftSpecificTool(), 
            AnalyzeRunDataSpecificTool()
            # Or, if using the single dispatching tool: DriftAnalysisTool()
        ]
        self.tools = self.specialized_tools + (mcp_tools or [])
        
        if not self.tools:
            logger.warning("DriftLangchainAgent initialized with no tools.")
        else:
            logger.info(f"DriftLangchainAgent initialized with {len(self.tools)} tools: {[tool.name for tool in self.tools]}")

        # Create the Langchain agent (e.g., ReAct agent)
        prompt_template_str = self.SYSTEM_INSTRUCTION + """

TOOLS:
------
You have access to the following tools:
{tools}

To use a tool, please use the following format:
```
Thought: Do I need to use a tool? Yes
Action: The action to take. Must be one of [{tool_names}]
Action Input: The input to the action, as a JSON string that matches the tool's args_schema.
Observation: The result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
```
Thought: Do I need to use a tool? No
Final Answer: [your response here, ensure it directly addresses the user's query and summarizes findings if applicable]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}"""
        prompt = PromptTemplate.from_template(prompt_template_str)

        # Create the agent using AgentExecutor for more control if needed, or create_react_agent
        # self.agent_runnable = create_react_agent( # This creates a runnable, not an executor directly
        #     llm=self.model,
        #     tools=self.tools,
        #     prompt=prompt
        # )
        # To use it as an executor:
        self.agent_executor = AgentExecutor(
            agent=create_react_agent(llm=self.model, tools=self.tools, prompt=prompt),
            tools=self.tools,
            verbose=True, # For debugging
            handle_parsing_errors=True # Important for ReAct agents
        )


    async def ainvoke(self, query: str, sessionId: str) -> dict[str, Any]:
        logger.info(
            f"DriftLangchainAgent.ainvoke called with query: '{query}', sessionId: '{sessionId}'"
        )
        # config = RunnableConfig(configurable={"session_id": sessionId}) # If using memory with session_id

        try:
            langchain_input = {"input": query, "chat_history": []} # Add chat_history if needed
            
            # Invoke the AgentExecutor
            agent_response = await self.agent_executor.ainvoke(langchain_input)
            
            final_message = agent_response.get("output", "Could not process the request.")
            
            return {
                "is_task_complete": True,
                "require_user_input": False, 
                "content": final_message,
            }

        except Exception as e:
            logger.error(
                f"Unhandled exception in DriftLangchainAgent.ainvoke: {type(e).__name__} - {e}",
                exc_info=True,
            )
            return {
                "is_task_complete": True,
                "require_user_input": False,
                "content": f"An unexpected error occurred: {type(e).__name__}.",
            }

    def _get_agent_response_from_state(
        self, config: RunnableConfig, agent_runnable_or_executor # This would be your langgraph app if using langgraph
    ) -> dict[str, Any]:
        # This method is more relevant if using Langgraph and its state management.
        # For a simple agent, the response is usually directly from ainvoke.
        # Adapting from AirbnbAgent:
        logger.debug(
            f"Entering _get_agent_response_from_state for config: {config}"
        )
        # If using Langgraph:
        # current_state_snapshot = agent_runnable_or_executor.get_state(config)
        # state_values = getattr(current_state_snapshot, "values", {})
        # structured_response = state_values.get("structured_response") # If LLM generates this
        # final_messages = state_values.get("messages", [])
        # ... (rest of the logic from AirbnbAgent._get_agent_response_from_state) ...

        # Simplified for non-Langgraph direct agent output:
        # This function might not be directly called if ainvoke already formats the response.
        # The logic here would be to take the raw LLM/agent output and fit it into ResponseFormat.
        # For now, assume ainvoke handles it.
        return {
            "is_task_complete": True,
            "require_user_input": False,
            "content": "Error: State retrieval logic not fully implemented for this agent type.",
        }


    async def stream(self, query: str, sessionId: str) -> AsyncIterable[Any]:
        logger.info(
            f"DriftLangchainAgent.stream called with query: '{query}', sessionId: '{sessionId}'"
        )
        # config = RunnableConfig(configurable={"session_id": sessionId})
        langchain_input = {"input": query, "chat_history": []}

        try:
            # Use astream_log for AgentExecutor to get detailed intermediate steps
            # Note: astream_events is for runnables created by create_react_agent, not AgentExecutor directly.
            # AgentExecutor's astream gives final response chunks. astream_log gives more detail.
            async for chunk in self.agent_executor.astream_log(langchain_input, include_types=["llm", "tool", "agent"]):
                # logger.debug(f"Stream chunk for {sessionId}: {chunk}")
                content_to_yield = None
                is_final_for_turn = False

                # Example of processing different parts of the log stream
                # The structure of `chunk` from astream_log is more complex than astream_events
                # It's a list of operations. We are interested in specific ops.
                for op in chunk.ops:
                    path = op.get("path")
                    value = op.get("value")

                    if path and isinstance(value, dict):
                        if path.startswith("/logs/ChatHuggingFace") and "streamed_output_str" in value:
                            # LLM token stream
                            content_to_yield = value["streamed_output_str"]
                            break
                        elif path.startswith("/logs/ToolUsage") and "tool_input" in value:
                            tool_name = value.get("tool_name", "a tool")
                            # tool_input_str = json.dumps(value["tool_input"])
                            content_to_yield = f"Thinking about using tool: {tool_name}..."
                            break
                        elif path.startswith("/logs/ToolUsage/final_output") or path.startswith("/logs/PydanticToolsAgent/final_output") : # for older langchain versions
                            # Tool result
                            # tool_name might be part of a higher level log entry
                            observation = value.get("output") if isinstance(value,dict) else str(value)
                            if observation and isinstance(observation, str) and len(observation) > 200:
                                observation = observation[:200] + "..."
                            content_to_yield = f"Tool finished. Observation: {observation}"
                            break
                        elif path == "/final_output" and value.get("output"):
                            # Final agent response
                            content_to_yield = value["output"]
                            is_final_for_turn = True
                            break
                
                if content_to_yield:
                    yield {
                        "is_task_complete": is_final_for_turn,
                        "require_user_input": False, # Determine this based on agent's final output if needed
                        "content": content_to_yield,
                    }
                if is_final_for_turn:
                    return
            
            # Fallback if stream ends without a clear final output (should be handled by agent_executor)
            logger.warning(f"Stream for {sessionId} ended without explicit final output signal.")
            # final_response_after_stream = await self.ainvoke(query, sessionId)
            # yield final_response_after_stream

        except Exception as e:
            logger.error(
                f"Error during DriftLangchainAgent.stream for session {sessionId}: {e}",
                exc_info=True,
            )
            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "content": f"An error occurred during streaming: {getattr(e, 'message', str(e))}",
            }

# Removed create_agent() and drift_root_instruction() as their logic is integrated into DriftLangchainAgent
# The instantiation of the agent will be handled by the executor or main application logic.
