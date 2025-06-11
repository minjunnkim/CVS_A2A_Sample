# Drift Agent Executor (skeleton)

import asyncio
import logging
from typing import Any

from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState, Part, TextPart, FilePart, FileWithUri, FileWithBytes
from google.adk import Runner
from google.genai import types

logger = logging.getLogger(__name__)

def convert_a2a_parts_to_genai(parts: list[Part]) -> list[types.Part]:
    return [convert_a2a_part_to_genai(part) for part in parts]

def convert_a2a_part_to_genai(part: Part) -> types.Part:
    part = part.root
    if isinstance(part, TextPart):
        return types.Part(text=part.text)
    if isinstance(part, FilePart):
        if isinstance(part.file, FileWithUri):
            return types.Part(
                file_data=types.FileData(
                    file_uri=part.file.uri, mime_type=part.file.mime_type
                )
            )
        if isinstance(part.file, FileWithBytes):
            return types.Part(
                inline_data=types.Blob(
                    data=part.file.bytes, mime_type=part.file.mime_type
                )
            )
        raise ValueError(f"Unsupported file type: {type(part.file)}")
    raise ValueError(f"Unsupported part type: {type(part)}")

def convert_genai_parts_to_a2a(parts: list[types.Part]) -> list[Part]:
    return [
        convert_genai_part_to_a2a(part)
        for part in parts
        if (part.text or part.file_data or part.inline_data)
    ]

def convert_genai_part_to_a2a(part: types.Part) -> Part:
    if part.text:
        return TextPart(text=part.text)
    if part.file_data:
        return FilePart(
            file=FileWithUri(
                uri=part.file_data.file_uri,
                mime_type=part.file_data.mime_type,
            )
        )
    if part.inline_data:
        return Part(
            root=FilePart(
                file=FileWithBytes(
                    bytes=part.inline_data.data,
                    mime_type=part.inline_data.mime_type,
                )
            )
        )
    raise ValueError(f"Unsupported part type: {part}")

class DriftAgentExecutor(AgentExecutor):
    """Executes drift analysis and trend detection tasks."""
    def __init__(self, runner: Runner, card):
        self.runner = runner
        self._card = card
        self._running_sessions = {}

    def _run_agent(self, session_id, new_message: types.Content):
        return self.runner.run_async(session_id=session_id, user_id="self", new_message=new_message)

    async def _process_request(self, new_message: types.Content, session_id: str, task_updater: TaskUpdater):
        session_obj = await self._upsert_session(session_id)
        session_id = session_obj.id
        missing_model_prompted = False
        async for event in self._run_agent(session_id, new_message):
            if event.is_final_response():
                parts = convert_genai_parts_to_a2a(event.content.parts)
                # Check if drift analysis failed due to missing model and prompt user
                if any("model" in (getattr(p, 'text', '') or '').lower() and "missing" in (getattr(p, 'text', '') or '').lower() for p in parts):
                    if not missing_model_prompted:
                        task_updater.add_artifact([
                            TextPart(text="Please provide the model name for drift analysis.")
                        ])
                        missing_model_prompted = True
                        task_updater.complete()
                        break
                elif parts:
                    # Only emit the final tool result as user-facing output
                    task_updater.add_artifact(parts)
                    task_updater.complete()
                    break
                else:
                    # If no parts, do not emit any message
                    task_updater.complete()
                    break
            # Do not emit any user-facing artifact for non-final responses
            # Only update status internally
            if not event.get_function_calls():
                task_updater.update_status(TaskState.working)

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task:
            updater.submit()
        updater.start_work()
        await self._process_request(
            types.UserContent(parts=convert_a2a_parts_to_genai(context.message.parts)),
            context.context_id,
            updater,
        )

    async def analyze_drift(self, run_data: list[dict[str, Any]]):
        # Simulate drift analysis logic
        print("Analyzing drift and trends across code runs...")
        # Example: find changes in metrics, anomalies, or trends
        if not run_data:
            print("No run data provided.")
            return {"status": "no_data"}
        # Simulate a trend summary
        print(f"Analyzed {len(run_data)} runs. Example trend: metric X is increasing.")
        return {"status": "analyzed", "trend": "metric X is increasing"}

    async def handle_request(self, request: dict[str, Any]):
        run_data = request.get("run_data", [])
        result = await self.analyze_drift(run_data)
        return result

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        updater.cancel()

    async def _upsert_session(self, session_id: str):
        session = await self.runner.session_service.get_session(
            app_name=self.runner.app_name, user_id="self", session_id=session_id
        )
        if session is None:
            session = await self.runner.session_service.create_session(
                app_name=self.runner.app_name, user_id="self", session_id=session_id
            )
        if session is None:
            logger.error(
                f"Critical error: Session is None even after create_session for session_id: {session_id}"
            )
            raise RuntimeError(f"Failed to get or create session: {session_id}")
        return session
