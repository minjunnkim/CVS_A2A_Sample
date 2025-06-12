# filepath: /Users/minjunkim/Desktop/CVS Health/CVS_A2A_Sample/a2a-adk-app/main.py
import os
import sys
import subprocess
import asyncio

# Add project root to Python path to allow for absolute imports
# This assumes main.py is in the root of the a2a-adk-app project
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def print_instructions():
    """Prints instructions on how to run the agent system."""
    print("""
    A2A Multi-Agent System (Langchain + Hugging Face)
    -------------------------------------------------

    This system consists of multiple agents:
    1. Host Agent: User-facing interface (Gradio app) that delegates tasks.
    2. Scheduler Agent: Manages scheduling tasks.
    3. Monitor Agent: Monitors system status.
    4. Drift Agent: Detects model or data drift.

    To run the system:

    1. Ensure all dependencies are installed (from pyproject.toml):
       - cd /path/to/a2a-adk-app
       - (Recommended) Create and activate a virtual environment:
         - python -m venv .venv
         - source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
       - pip install -e .  # Or use your preferred package manager with pyproject.toml

    2. Set required environment variables:
       - HUGGINGFACEHUB_API_TOKEN: Your Hugging Face API token.
       - (Optional) SCHEDULER_AGENT_URL, MONITOR_AGENT_URL, DRIFT_AGENT_URL if they
         differ from the defaults (http://localhost:11001, ...:11002, ...:11003).

    3. Start each specialist agent server in separate terminals:
       - Terminal 1 (Scheduler Agent):
         python -m scheduler_agent  # (This will run scheduler_agent/__main__.py)
       - Terminal 2 (Monitor Agent):
         python -m monitor_agent    # (This will run monitor_agent/__main__.py)
       - Terminal 3 (Drift Agent):
         python -m drift_agent      # (This will run drift_agent/__main__.py)

       *Note: The __main__.py files for these agents need to be updated to run
        their Langchain executors with an A2A-compliant HTTP server (e.g., FastAPI/Uvicorn).*

    4. Start the Host Agent Gradio application:
       - Terminal 4 (Host Agent):
         python host_agent/app.py

    5. Access the Host Agent UI in your browser (usually http://localhost:8083).

    """)

if __name__ == "__main__":
    print_instructions()
    # Placeholder for future: Option to launch all agents for local dev.
    # For now, manual startup of each agent is required as per instructions.
