"""
Deep Health AI - REST API Server Entry Point
--------------------------------------------
Runs the FastAPI microservice on host 0.0.0.0 and port 8000.

Usage:
    python api_server.py [--port 8000] [--reload]
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Deep Health AI Clinical REST API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code change")
    args = parser.parse_args()

    print(f"🩺 Starting Deep Health AI REST API on http://{args.host}:{args.port}")
    print(f"📖 OpenAPI Swagger UI available at: http://{args.host}:{args.port}/docs")

    uvicorn.run("src.api.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
