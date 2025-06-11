# Host Agent built by ADK

This example shows how to create an A2A Server that uses an ADK-based Agent.


## Setting up the agent

1. Create the .env file with your API Key
   ```bash
   GOOGLE_GENAI_USE_VERTEXAI=TRUE
   GOOGLE_CLOUD_PROJECT="your project id"
   GOOGLE_CLOUD_LOCATION=us-central1
   AIR_AGENT_URL=http://localhost:10002
   WEA_AGENT_URL=http://localhost:10001
   SCHEDULER_AGENT_URL=http://localhost:11001
   MONITOR_AGENT_URL=http://localhost:11002
   DRIFT_AGENT_URL=http://localhost:11003
   ```

