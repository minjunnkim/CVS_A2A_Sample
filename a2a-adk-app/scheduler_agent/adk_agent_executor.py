# Scheduler Agent Executor (skeleton)

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

class SchedulerAgentExecutor(AgentExecutor):
    """Executes scheduling tasks and handles email confirmation logic."""
    def __init__(self, runner: Runner, card):
        self.runner = runner
        self._card = card
        self._running_sessions = {}

    def _run_agent(self, session_id, new_message: types.Content):
        return self.runner.run_async(session_id=session_id, user_id="self", new_message=new_message)

    async def _process_request(self, new_message: types.Content, session_id: str, task_updater: TaskUpdater):
        session_obj = await self._upsert_session(session_id)
        session_id = session_obj.id
        missing_email_prompted = False
        async for event in self._run_agent(session_id, new_message):
            if event.is_final_response():
                parts = convert_genai_parts_to_a2a(event.content.parts)
                # Check if scheduling failed due to missing email and prompt user
                if any("email" in (getattr(p, 'text', '') or '').lower() and "missing" in (getattr(p, 'text', '') or '').lower() for p in parts):
                    if not missing_email_prompted:
                        task_updater.add_artifact([
                            TextPart(text="Please provide the email address for scheduling confirmation.")
                        ])
                        missing_email_prompted = True
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

    async def schedule_code(self, code_path: str, schedule_type: str, personnel_email: str):
        # Simulate scheduling logic
        # schedule_type: 'daily', 'weekly', 'monthly', 'manual'
        # Send confirmation email (simulated)
        print(f"Scheduling {code_path} as {schedule_type} for {personnel_email}")
        await self.send_confirmation_email(personnel_email, code_path, schedule_type)

    async def send_confirmation_email(self, email: str, code_path: str, schedule_type: str):
        # Simulate sending an email with Confirm, Cancel, Snooze buttons
        print(f"Sending confirmation email to {email} for {code_path} [{schedule_type}]")
        print("[Confirm] [Cancel] [Snooze]")
        # In a real implementation, integrate with an email API and handle button callbacks

    async def run_code(self, code_path: str):
        # Simulate running the code file
        print(f"Running code: {code_path}")
        # In a real implementation, use subprocess or similar to execute the code
        await asyncio.sleep(1)
        print(f"Execution complete: {code_path}")

    async def handle_request(self, request: dict[str, Any]):
        # Entry point for handling scheduling requests
        code_path = request.get("code_path")
        schedule_type = request.get("schedule_type", "manual")
        personnel_email = request.get("personnel_email")
        if not code_path or not personnel_email:
            return {"status": "error", "message": "Missing code_path or personnel_email"}
        await self.schedule_code(code_path, schedule_type, personnel_email)
        return {"status": "scheduled", "code_path": code_path, "schedule_type": schedule_type}

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
