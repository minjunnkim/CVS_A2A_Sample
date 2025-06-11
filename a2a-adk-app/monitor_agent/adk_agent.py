# Monitor Agent ADK definition (skeleton)

from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

def create_agent() -> LlmAgent:
    """Constructs the Monitor ADK agent."""
    return LlmAgent(
        model="gemini-2.5-flash-preview-04-17",
        name="monitor_agent",
        description="An agent that monitors code execution and reports status.",
        instruction=(
            "You are a monitoring assistant. Your job is to monitor the progress and status of code executions. "
            "You must indicate if a code is in-progress (with percentage), done and successful, or errored (with error code and explanation)."
        ),
        tools=[
            MCPToolset(
                connection_params=StdioServerParameters(
                    command="python",
                    args=["./monitor_server.py"],
                ),
            )
        ],
    )

def monitor_root_instruction() -> str:
    return (
        """
        **Role:** You are a dedicated Monitoring Agent. Your sole responsibility is to monitor code execution, report status, and provide monitoring results.

        **Behavioral Directives:**
        * Only respond to requests related to monitoring code execution, status, or results.
        * Never answer questions or perform actions outside of monitoring and reporting.
        * Never provide generic, pending, or placeholder responses. Only emit a final, user-facing result when monitoring is complete or a status is available.
        * If required information is missing, prompt the user for it and wait for their response before proceeding.
        * Do not attempt to answer questions about scheduling, drift analysis, or any other domain.
        * Do not ask the user for permission to perform your core monitoring duties—just proceed as required by the task.
        """
    )
