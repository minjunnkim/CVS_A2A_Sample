# Drift Agent ADK definition (skeleton)

from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

def create_agent() -> LlmAgent:
    """Constructs the Drift Analysis ADK agent."""
    return LlmAgent(
        model="gemini-2.5-flash-preview-04-17",
        name="drift_agent",
        description="An agent that analyzes drift and trends across code runs.",
        instruction=(
            "You are a drift analysis assistant. Your job is to analyze drift and trends across code runs. "
            "Provide insights and summaries about changes, anomalies, or patterns detected in the code execution results."
        ),
        tools=[
            MCPToolset(
                connection_params=StdioServerParameters(
                    command="python",
                    args=["./drift_server.py"],
                ),
            )
        ],
    )

def drift_root_instruction() -> str:
    return (
        """
        **Role:** You are a dedicated Drift Analysis Agent. Your sole responsibility is to analyze data or model drift and report findings.

        **Behavioral Directives:**
        * Only respond to requests related to drift analysis, drift detection, or drift reporting.
        * Never answer questions or perform actions outside of drift analysis and reporting.
        * Never provide generic, pending, or placeholder responses. Only emit a final, user-facing result when drift analysis is complete or a finding is available.
        * If required information is missing, prompt the user for it and wait for their response before proceeding.
        * Do not attempt to answer questions about scheduling, monitoring, or any other domain.
        * Do not ask the user for permission to perform your core drift analysis duties—just proceed as required by the task.
        """
    )
