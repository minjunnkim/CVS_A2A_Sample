from langchain_community.llms import HuggingFaceHub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from typing import AsyncIterator, Dict, Any

from .scheduler_tool import ScheduleEventTool, CancelEventTool, ListScheduledEventsTool

SYSTEM_INSTRUCTION = '''
**Role:** You are a dedicated Scheduling Agent. Your sole responsibility is to schedule code execution tasks, manage confirmation emails for personnel, and provide information about already-scheduled code execution tasks.

**Behavioral Directives:**
*   Only respond to requests related to scheduling code execution (daily, weekly, monthly, or manual), or retrieving information about already-scheduled code execution tasks.
*   When scheduling, ensure you have the event name, time, and optionally frequency and personnel email.
*   When cancelling, ensure you have the event name and optionally personnel email.
*   When listing events, no specific input is usually needed unless filtering is requested (which is not supported by the current dummy tool).
*   Simulate sending confirmation emails as part of the tool's response; do not try to implement actual email sending or button interactions like Confirm/Cancel/Snooze, as these are beyond your current tool capabilities. The tools will return a string indicating this simulation.
*   Never answer questions or perform actions outside of scheduling or retrieving scheduled task information.
*   Never provide generic, pending, or placeholder responses. Only emit a final, user-facing result when the scheduling, cancellation, or retrieval process is complete via a tool.
*   If required information (like event name for scheduling/cancelling, or time for scheduling) is missing, you should indicate what is missing. The ReAct agent should ideally re-prompt or indicate failure if critical info is missing and tools can't proceed.
*   Do not attempt to answer questions about monitoring, drift analysis, or any other domain.
*   Do not ask the user for permission to perform your core scheduling duties—just proceed by calling the appropriate tool if you have enough information.
'''

RESPONSE_FORMAT_INSTRUCTION_STR = '''{
  "status": "scheduled" | "cancelled" | "success" | "error",
  "event_name": "string (if applicable)",
  "time": "string (if applicable)",
  "frequency_description": "string (if applicable)",
  "personnel_email": "string (if applicable, even if None)",
  "confirmation_message": "string (detailing simulated action)",
  "scheduled_events": "array (if listing events)",
  "details": "string (additional details or error message)"
}'''

class SchedulerLangchainAgent:
    def __init__(self, llm_model_name: str = "HuggingFaceH4/zephyr-7b-alpha"):
        self.llm = HuggingFaceHub(
            repo_id=llm_model_name,
            model_kwargs={"temperature": 0.7, "max_length": 1024},
        )
        self.tools = [ScheduleEventTool(), CancelEventTool(), ListScheduledEventsTool()]

        prompt_str = f"""\
{SYSTEM_INSTRUCTION}

TOOLS:
------
You have access to the following tools. You can use these tools by responding with a JSON blob that includes an "action" key (the name of the tool to use) and an "action_input" key (the input to the tool).
Valid "action" values are: [{{tool_names}}]

Tool Descriptions:
{{tools}}

RESPONSE FORMAT (when you have a final answer):
----------------
When you have a response to convey to the Human, or if you decide that no tool is necessary, you MUST use the following JSON format. Your response should be enclosed in a single JSON blob.

```json
{{
  "thought": "A brief explanation of your decision.",
  "output": "Your final answer here. This answer MUST be the JSON string provided by the tool if a tool was used. The expected JSON structure from tools is: {RESPONSE_FORMAT_INSTRUCTION_STR}"
}}
```

If you determine that a tool is necessary, use the following JSON format for your action:
```json
{{
  "thought": "A brief explanation of why you are using this specific tool and what parameters you are passing to it based on the user's input.",
  "action": "The name of the action to take, which must be one of the available tools: [{{tool_names}}]",
  "action_input": {{ "parameter_name": "value" }}  // This must be a valid JSON object for the tool's input arguments
}}
```
After the tool action is executed, you will receive an Observation. Based on this observation, you should think again and decide if another tool use is needed or if you can now provide the final answer (which will be the tool's output) in the specified format.

Begin!

New input: {{input}}
{{agent_scratchpad}}
"""
        self.prompt = PromptTemplate.from_template(prompt_str)

        self.agent = create_react_agent(self.llm, self.tools, self.prompt)
        
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True, 
            max_iterations=5
        )

    async def ainvoke(self, input_text: str) -> Dict[str, Any]:
        return await self.agent_executor.ainvoke({"input": input_text})

    async def stream(self, input_text: str) -> AsyncIterator[Dict[str, Any]]:
        async for log_entry in self.agent_executor.astream_log(
            {"input": input_text}, include_types=["llm", "tool", "agent"]
        ):
            yield log_entry

def create_agent() -> SchedulerLangchainAgent:
    """Constructs the Scheduler Langchain agent."""
    return SchedulerLangchainAgent()
