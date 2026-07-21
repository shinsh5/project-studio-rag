"""
PROJECT Studio Web GUI 실행 스크립트 (run_gui.py)
이 스크립트를 실행하면 PROJECT Studio 웹 GUI 서버가 시작되고 브라우저가 자동으로 열립니다.
"""
import sys
import os
import argparse
from run_inference import run_gui_mode

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start PROJECT Studio Web GUI server and open browser.")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Port for GUI web server (default: 8000).")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address for GUI web server (default: 127.0.0.1).")
    args = parser.parse_args()

    run_gui_mode(host=args.host, port=args.port)
