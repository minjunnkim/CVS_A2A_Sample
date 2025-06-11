import sys
import json
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("scheduler")

@mcp.tool()
async def schedule_event(event_name: str, time: str, frequency: str = "once", personnel_email: str = None) -> str:
    """Dummy tool to simulate scheduling a code file. Always returns a simulated response."""
    freq_text = {
        "daily": "every day",
        "weekly": "every week",
        "monthly": "every month",
        "once": "once",
        "manual": "manually triggered"
    }.get(frequency.lower(), frequency)
    email_text = f" Confirmation email sent to {personnel_email}." if personnel_email else ""
    return f"[SIMULATION] ✅ Code file '{event_name}' would be scheduled to run {freq_text} at {time}.{email_text} (No real scheduling performed.)"

@mcp.tool()
async def cancel_event(event_name: str, personnel_email: str = None) -> str:
    """Dummy tool to simulate cancelling a scheduled code file. Always returns a simulated response."""
    email_text = f" Cancellation email sent to {personnel_email}." if personnel_email else ""
    return f"[SIMULATION] ✅ Scheduled run for code file '{event_name}' would be cancelled.{email_text} (No real cancellation performed.)"

def main():
    # Read input from stdin (simulate MCPToolset stdio)
    input_data = sys.stdin.read()
    try:
        request = json.loads(input_data)
        action = request.get("action", "schedule")
        code_path = request.get("code_path", "unknown.py")
        schedule_type = request.get("schedule_type", "manual")
        personnel_email = request.get("personnel_email", "user@example.com")
        # Simulate scheduling logic
        if action == "schedule":
            response = {
                "status": "scheduled",
                "code_path": code_path,
                "schedule_type": schedule_type,
                "personnel_email": personnel_email,
                "confirmation": "Email sent for confirmation."
            }
        else:
            response = {"status": "error", "message": "Unknown action"}
    except Exception as e:
        response = {"status": "error", "message": str(e)}
    print(json.dumps(response))
    sys.stdout.flush()

if __name__ == "__main__":
    mcp.run(transport="stdio")
