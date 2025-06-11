import sys
import json
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("drift")

@mcp.tool()
async def detect_drift(model_name: str) -> str:
    """Dummy tool to detect drift for a model. Always returns a simulated response."""
    return f"[SIMULATION] ✅ No drift detected for model '{model_name}'. (No real drift detection performed.)"

@mcp.tool()
async def reset_drift(model_name: str) -> str:
    """Dummy tool to reset drift status for a model. Always returns a simulated response."""
    return f"[SIMULATION] ✅ Drift status for model '{model_name}' has been reset. (No real drift reset performed.)"

def main():
    input_data = sys.stdin.read()
    try:
        request = json.loads(input_data)
        action = request.get("action", "analyze")
        run_data = request.get("run_data", [])
        # Simulate drift analysis logic
        if action == "analyze":
            if run_data:
                response = {
                    "status": "analyzed",
                    "trend": "metric X is increasing",
                    "details": f"Analyzed {len(run_data)} runs."
                }
            else:
                response = {"status": "no_data", "message": "No run data provided."}
        else:
            response = {"status": "error", "message": "Unknown action"}
    except Exception as e:
        response = {"status": "error", "message": str(e)}
    print(json.dumps(response))
    sys.stdout.flush()

if __name__ == "__main__":
    mcp.run(transport="stdio")
