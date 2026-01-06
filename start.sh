#!/bin/bash

# Start the FastAPI backend in the background
uvicorn predict:app --host 0.0.0.0 --port 9696 &

# Start the Streamlit frontend
# Note: Ensure the UI is pointing to localhost:9696
streamlit run ui.py --server.port 8501 --server.address 0.0.0.0