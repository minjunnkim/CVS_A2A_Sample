"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import gradio as gr
from typing import List, AsyncIterator, Tuple, Dict, Any
import asyncio
import traceback
import os
import uuid

# Assuming host_langchain_agent.py is in the same directory or accessible in PYTHONPATH
from host_langchain_agent import HostLangchainAgent 

# Environment variables for agent URLs (can be overridden by .env)
DEFAULT_SCHEDULER_AGENT_URL = "http://localhost:11001"
DEFAULT_MONITOR_AGENT_URL = "http://localhost:11002"
DEFAULT_DRIFT_AGENT_URL = "http://localhost:11003"

# Global instance of the host agent (initialized asynchronously later)
HOST_AGENT_INSTANCE: HostLangchainAgent = None

async def initialize_host_agent():
    """Initializes the global HOST_AGENT_INSTANCE."""
    global HOST_AGENT_INSTANCE
    if HOST_AGENT_INSTANCE is None:
        print("Initializing HostLangchainAgent...")
        remote_addresses = [
            os.getenv("SCHEDULER_AGENT_URL", DEFAULT_SCHEDULER_AGENT_URL),
            os.getenv("MONITOR_AGENT_URL", DEFAULT_MONITOR_AGENT_URL),
            os.getenv("DRIFT_AGENT_URL", DEFAULT_DRIFT_AGENT_URL),
        ]
        try:
            HOST_AGENT_INSTANCE = await HostLangchainAgent.create(remote_agent_addresses=remote_addresses)
            print("HostLangchainAgent initialized successfully.")
        except Exception as e:
            print(f"FATAL: Failed to initialize HostLangchainAgent: {e}")
            traceback.print_exc()
            # In a real app, you might want to prevent Gradio from starting or show an error state.
    return HOST_AGENT_INSTANCE

async def chat_interface_fn(
    message: str, 
    history: List[Tuple[str, str]], 
    session_state: Dict[str, Any]
) -> AsyncIterator[Tuple[List[Tuple[str, str]], Dict[str, Any]]]:
    """
    Handles a new message from the user in the Gradio chat interface.
    Manages session state for the HostLangchainAgent.
    """
    if HOST_AGENT_INSTANCE is None:
        yield history + [(message, "Error: Host Agent not initialized. Please try again later.")], session_state
        return

    # Ensure session_id exists in session_state
    if "session_id" not in session_state or not session_state["session_id"]:
        session_state["session_id"] = str(uuid.uuid4())
        print(f"New Gradio session started: {session_state['session_id']}")

    current_session_id = session_state["session_id"]
    print(f"Processing message for session {current_session_id}: '{message}'")

    # Append user message to history for display
    history.append((message, None)) # Placeholder for assistant response
    yield history, session_state

    try:
        # Get response from the host agent
        # The process_message method is async
        assistant_response = await HOST_AGENT_INSTANCE.process_message(current_session_id, message)
        
        # Update history with assistant's response
        history[-1] = (message, assistant_response) # Update the placeholder
        yield history, session_state

    except Exception as e:
        print(f"Error in chat_interface_fn (Session: {current_session_id}, Type: {type(e)}): {e}")
        traceback.print_exc()
        error_message = "An error occurred while processing your request. Please check server logs."
        history[-1] = (message, error_message) # Update placeholder with error
        yield history, session_state

async def launch_gradio_app():
    """Initializes agent and launches the Gradio app."""
    agent = await initialize_host_agent()
    if not agent:
        print("Gradio app will not launch as agent initialization failed.")
        return

    with gr.Blocks(theme=gr.themes.Ocean(), title="A2A Langchain Host Agent") as demo:
        gr.Image(
            "static/a2a.png",  # Assuming this path is correct relative to where app.py is run
            width=100,
            height=100,
            scale=0,
            show_label=False,
            show_download_button=False,
            container=False,
            show_fullscreen_button=False,
        )
        gr.Markdown("<h1>A2A Host Agent (Langchain + Hugging Face)</h1>")
        gr.Markdown(
            "Interact with the A2A multi-agent system. Ask about scheduling, system monitoring, or model drift."
        )

        # Session state for Gradio. Each user session gets its own dictionary.
        # We store our session_id within this dictionary.
        session_state_gr = gr.State({})

        chatbot = gr.Chatbot(label="Conversation", height=600)
        msg_textbox = gr.Textbox(
            placeholder="Type your message here... e.g., 'Schedule a new meeting' or 'What is the system status?'",
            label="Your Message",
            lines=2
        )
        clear_button = gr.Button("Clear Conversation")

        # Wire up the chat interface logic
        msg_textbox.submit(
            chat_interface_fn, 
            [msg_textbox, chatbot, session_state_gr], 
            [chatbot, session_state_gr]
        )
        clear_button.click(lambda: ([], {}), None, [chatbot, session_state_gr], queue=False)

    print("Launching Gradio interface...")
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=8083, # Keep the same port as before or choose a new one
        # share=True # Uncomment for public link if needed, requires login if on Hugging Face Spaces
    )
    print("Gradio application has been shut down.")

if __name__ == "__main__":
    # Check for Hugging Face API token (HostLangchainAgent will also check, but good for early feedback)
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        print("WARNING: HUGGINGFACEHUB_API_TOKEN is not set. The agent may not function correctly.")
    
    asyncio.run(launch_gradio_app())
