import sys
import json
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("monitor")

@mcp.tool()
async def monitor_status(target: str) -> str:
    """Dummy tool to monitor the status of a target. Always returns a simulated response."""
    return f"[SIMULATION] Status of '{target}': OK. (No real monitoring performed.)"

@mcp.tool()
async def report_issue(target: str, issue: str) -> str:
    """Dummy tool to report an issue for a target. Always returns a simulated response."""
    return f"[SIMULATION] ✅ Issue has been reported for '{target}': {issue} (No real issue reporting performed.)"

def main():
    input_data = sys.stdin.read()
    try:
        request = json.loads(input_data)
        action = request.get("action", "status")
        code_path = request.get("code_path")
        # Simulate monitoring logic
        if action == "status":
            if code_path:
                response = {
                    "status": "in-progress",
                    "code_path": code_path,
                    "progress": "60%",
                    "details": "Execution is 60% complete."
                }
            else:
                response = {
                    "status": "system-ok",
                    "details": "All scheduled codes are running as expected."
                }
        else:
            response = {"status": "error", "message": "Unknown action"}
    except Exception as e:
        response = {"status": "error", "message": str(e)}
    print(json.dumps(response))
    sys.stdout.flush()

if __name__ == "__main__":
    mcp.run(transport="stdio")
