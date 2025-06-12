# Monitor Agent ADK definition (skeleton)

from langchain_community.llms import HuggingFaceHub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from typing import List, AsyncIterator, Dict, Any

from .monitor_tool import MonitorStatusTool, ReportIssueTool

# Define the system instruction for the Langchain agent
SYSTEM_INSTRUCTION = """
**Role:** You are a dedicated Monitoring Agent. Your sole responsibility is to monitor code execution, report status, and provide monitoring results.

**Behavioral Directives:**
* Only respond to requests related to monitoring code execution, status, or results.
* Never answer questions or perform actions outside of monitoring and reporting.
* Never provide generic, pending, or placeholder responses. Only emit a final, user-facing result when monitoring is complete or a status is available.
* If required information is missing, prompt the user for it and wait for their response before proceeding.
* Do not attempt to answer questions about scheduling, drift analysis, or any other domain.
* Do not ask the user for permission to perform your core monitoring duties—just proceed as required by the task.
"""

# This is the string representation of the JSON format for the agent's output.
# It's used within the main prompt.
RESPONSE_FORMAT_INSTRUCTION_STR = '''{
  "status": "in-progress" | "done" | "error",
  "code_path": "string (optional, if applicable)",
  "progress": "string (e.g., ''''60%'''' optional, if in-progress)",
  "details": "string (detailed message or error explanation)"
}'''

class MonitorLangchainAgent:
    def __init__(self, llm_model_name: str = "HuggingFaceH4/zephyr-7b-alpha"): # Changed to Zephyr
        self.llm = HuggingFaceHub(
            repo_id=llm_model_name,
            model_kwargs={"temperature": 0.7, "max_length": 1024}, # Increased max_length
            # HUGGINGFACEHUB_API_TOKEN should be set in the environment
        )
        self.tools = [MonitorStatusTool(), ReportIssueTool()]

        # Construct the prompt for the ReAct agent
        # This structure is based on successful ReAct agent prompts.
        prompt_str = f"""
{SYSTEM_INSTRUCTION}

TOOLS:
------
You have access to the following tools. You can use these tools by responding with a JSON blob that includes an "action" key (the name of the tool to use) and an "action_input" key (the input to the tool).
Valid "action" values are: [{{tool_names}}]

Tool Descriptions:
{{tools}}

RESPONSE FORMAT:
----------------
When you have a response to convey to the Human, or if you decide that no tool is necessary, you MUST use the following JSON format. Your response should be enclosed in a single JSON blob.

```json
{{
  "thought": "A brief explanation of your decision (e.g., 'Do I need to use a tool? No, I have the final answer.' or 'Do I need to use a tool? No, the query is a direct question not requiring a tool.').",
  "output": "Your final answer here. This answer MUST strictly adhere to the JSON structure: {RESPONSE_FORMAT_INSTRUCTION_STR} OR be a descriptive text if the query is not a direct command that produces this JSON."
}}
```

If you determine that a tool is necessary, use the following JSON format for your action:
```json
{{
  "thought": "A brief explanation of why you are using this specific tool and what you expect to achieve.",
  "action": "The name of the action to take, which must be one of the available tools: [{{tool_names}}]",
  "action_input": "The input required by the selected tool."
}}
```
After the tool action is executed, you will receive an Observation. Based on this observation, you should think again and decide if another tool use is needed or if you can now provide the final answer in the specified format.

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
            handle_parsing_errors=True, # Important for ReAct agents
            max_iterations=5 # Prevent overly long loops
        )

    async def ainvoke(self, input_text: str) -> Dict[str, Any]:
        """Invokes the agent with the given input text."""
        return await self.agent_executor.ainvoke({"input": input_text})

    async def stream(self, input_text: str) -> AsyncIterator[Dict[str, Any]]:
        """Streams the agent's thoughts and actions."""
        async for log_entry in self.agent_executor.astream_log(
            {"input": input_text}, include_types=["llm", "tool", "agent"]
        ):
            yield log_entry

def create_agent() -> MonitorLangchainAgent:
    """Constructs the Monitor Langchain agent."""
    return MonitorLangchainAgent()

# The monitor_root_instruction function is now incorporated into SYSTEM_INSTRUCTION
# and can be removed if not used elsewhere.
