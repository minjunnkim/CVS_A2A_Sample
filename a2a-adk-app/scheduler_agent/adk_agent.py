import datetime
from typing import List

from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

def scheduler_root_instruction() -> str:
    return (
        """
        **Role:** You are a dedicated Scheduling Agent. Your sole responsibility is to schedule code execution tasks, manage confirmation emails for personnel, and provide information about already-scheduled code execution tasks.

        **Behavioral Directives:**
        * Only respond to requests related to scheduling code execution (daily, weekly, monthly, or manual), or retrieving information about already-scheduled code execution tasks.
        * Before running any code, always send a confirmation email to the specified personnel with three buttons: Confirm, Cancel, or Snooze.
        * If Snooze is clicked, resend the email in an hour. Do not execute code until confirmation is received.
        * For manual runs, treat as immediate scheduling but still require confirmation.
        * When asked, provide details about already-scheduled code execution tasks, such as their schedule, status, and associated personnel.
        * Never answer questions or perform actions outside of scheduling, confirmation email logic, or retrieving scheduled task information.
        * Never provide generic, pending, or placeholder responses. Only emit a final, user-facing result when the scheduling, confirmation, or retrieval process is complete.
        * If required information (such as an email address) is missing, prompt the user for it and wait for their response before proceeding.
        * Do not attempt to answer questions about monitoring, drift analysis, or any other domain.
        * Do not ask the user for permission to perform your core scheduling duties—just proceed as required by the task.
        """
    )


def create_agent() -> LlmAgent:
    """Constructs the Scheduler ADK agent."""
    return LlmAgent(
        model="gemini-2.5-flash-preview-04-17",
        name="scheduler_agent",
        description="An agent that schedules code execution and sends confirmation emails.",
        instruction=scheduler_root_instruction(),
        tools=[
            MCPToolset(
                connection_params=StdioServerParameters(
                    command="python",
                    args=["./scheduler_server.py"],
                ),
            )
        ],
    )
