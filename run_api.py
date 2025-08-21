#!/usr/bin/env python3
"""
D3-DNA API Server Launcher

Quick launcher script for the D3-DNA sequence generation API.
Run this script on your GPU cluster to start the server on port 2025.

Usage:
    python run_api.py                    # Start on localhost:2025
    python run_api.py --host 0.0.0.0     # Start on all interfaces
    python run_api.py --port 3000        # Start on custom port
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from api.main import main
from api.config import config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="D3-DNA Sequence Generation API")
    parser.add_argument("--host", default=config.host, 
                       help=f"Host to bind to (default: {config.host})")
    parser.add_argument("--port", type=int, default=config.port,
                       help=f"Port to bind to (default: {config.port})")
    parser.add_argument("--reload", action="store_true",
                       help="Enable auto-reload for development")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Update config with command line arguments
    config.host = args.host
    config.port = args.port
    
    print("🧬 D3-DNA Sequence Generator API")
    print("=" * 50)
    print(f"Host: {config.host}")
    print(f"Port: {config.port}")
    print(f"Device: {config.device}")
    print("=" * 50)
    print()
    print("🌐 Access the web interface at:")
    print(f"   http://{config.host}:{config.port}")
    print()
    print("📡 API endpoints:")
    print(f"   POST http://{config.host}:{config.port}/generate")
    print(f"   GET  http://{config.host}:{config.port}/health")
    print(f"   GET  http://{config.host}:{config.port}/datasets/info")
    print(f"   GET  http://{config.host}:{config.port}/models/status")
    print()
    
    if config.host == "localhost":
        print("💡 For SSH tunneling from local machine:")
        print(f"   ssh -L {config.port}:localhost:{config.port} user@your-gpu-server")
        print()
    
    # Start the server
    main()