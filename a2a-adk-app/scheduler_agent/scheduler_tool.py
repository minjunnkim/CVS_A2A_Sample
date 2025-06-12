
from langchain_core.tools import BaseTool
from typing import Type, Optional
from pydantic import BaseModel, Field
import json

class ScheduleEventInput(BaseModel):
    event_name: str = Field(description="The name or identifier of the event/code file to schedule.")
    time: str = Field(description="The time at which the event should be scheduled (e.g., '09:00 AM').")
    frequency: str = Field(default="once", description="The frequency of the event (e.g., 'daily', 'weekly', 'monthly', 'once', 'manual').")
    personnel_email: Optional[str] = Field(default=None, description="Email address of the personnel to notify.")

class ScheduleEventTool(BaseTool):
    name = "schedule_event"
    description = "Schedules a code file/event to run at a specified time and frequency. Simulates the scheduling and notification."
    args_schema: Type[BaseModel] = ScheduleEventInput

    def _run(self, event_name: str, time: str, frequency: str = "once", personnel_email: Optional[str] = None) -> str:
        freq_text_map = {
            "daily": "every day",
            "weekly": "every week",
            "monthly": "every month",
            "once": "once",
            "manual": "manually triggered"
        }
        freq_text = freq_text_map.get(frequency.lower(), frequency)
        email_text = f" Confirmation email sent to {personnel_email}." if personnel_email else ""
        # Simulate the response based on the original scheduler_server.py logic
        response = {
            "status": "scheduled",
            "event_name": event_name,
            "time": time,
            "frequency_description": freq_text,
            "personnel_email": personnel_email,
            "confirmation_message": f"[SIMULATION] ✅ Code file '{event_name}' would be scheduled to run {freq_text} at {time}.{email_text} (No real scheduling performed.)"
        }
        return json.dumps(response)

    async def _arun(self, event_name: str, time: str, frequency: str = "once", personnel_email: Optional[str] = None) -> str:
        return self._run(event_name, time, frequency, personnel_email)

class CancelEventInput(BaseModel):
    event_name: str = Field(description="The name or identifier of the event/code file to cancel.")
    personnel_email: Optional[str] = Field(default=None, description="Email address of the personnel to notify of cancellation.")

class CancelEventTool(BaseTool):
    name = "cancel_event"
    description = "Cancels a previously scheduled code file/event. Simulates the cancellation and notification."
    args_schema: Type[BaseModel] = CancelEventInput

    def _run(self, event_name: str, personnel_email: Optional[str] = None) -> str:
        email_text = f" Cancellation email sent to {personnel_email}." if personnel_email else ""
        response = {
            "status": "cancelled",
            "event_name": event_name,
            "personnel_email": personnel_email,
            "confirmation_message": f"[SIMULATION] ✅ Scheduled run for code file '{event_name}' would be cancelled.{email_text} (No real cancellation performed.)"
        }
        return json.dumps(response)

    async def _arun(self, event_name: str, personnel_email: Optional[str] = None) -> str:
        return self._run(event_name, personnel_email)

class ListScheduledEventsTool(BaseTool):
    name = "list_scheduled_events"
    description = "Retrieves a list of all currently scheduled code execution tasks. Returns a simulated list."
    # No arguments needed for this tool

    def _run(self) -> str:
        # Static dummy response from original scheduler_server.py
        simulated_list = [
            {"event_name": "code_file_a.py", "schedule_info": "Daily at 09:00 AM", "personnel": "alice@example.com"},
            {"event_name": "code_file_b.py", "schedule_info": "Weekly on Monday at 02:00 PM", "personnel": "bob@example.com"},
            {"event_name": "code_file_c.py", "schedule_info": "Monthly on the 1st at 11:00 AM", "personnel": "carol@example.com"}
        ]
        response = {
            "status": "success",
            "scheduled_events": simulated_list,
            "details": "[SIMULATION] 📋 Scheduled code executions listed above. (No real schedule data. This is a dummy response.)"
        }
        return json.dumps(response)

    async def _arun(self) -> str:
        return self._run()
