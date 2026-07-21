"""
ROI-RAG CLI 추론 도구.
사용법:
  # 단발성 질의 실행
  python run_inference.py --query "엔비디아의 주요 매출 동력은 무엇인가요?"
  
  # 대화형 CLI 모드 실행
  python run_inference.py --interactive
"""
import sys
import argparse
from roi_rag import get_roi_rag_pipeline

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

def print_response(query: str, res: dict):
    print("\n" + "="*60)
    print(f"[Query] {query}")
    print("="*60)
    print(f"[ROI-RAG Answer]\n{res['answer']}")
    print("-" * 60)
    print("[Retrieved Evidence Units (Top-K)]")
    for i, eu_text in enumerate(res.get("retrieved_contexts", []), 1):
        print(f"\n[{i}] {eu_text}")
    print("-" * 60)
    print(f"[Metrics] Latency: {res['latency_ms']} ms | API Calls: {res['api_calls']} | Est. Tokens Used: {res['tokens_used']}")
    print("="*60 + "\n")

import threading
import time
import webbrowser

import socket

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def run_gui_mode(host: str = "127.0.0.1", port: int = 8000):
    try:
        import uvicorn
    except ImportError:
        print("[Error] 'uvicorn' is required for GUI mode. Please install it via `pip install uvicorn`.")
        return

    local_url = f"http://127.0.0.1:{port}" if host == "0.0.0.0" else f"http://{host}:{port}"
    print("\n" + "="*60)
    print("[GUI Mode] Starting PROJECT Studio Web GUI...")
    print(f"[Local Access]   {local_url}")
    if host == "0.0.0.0":
        lan_ip = get_local_ip()
        print(f"[Network Access] http://{lan_ip}:{port} (다른 PC/스마트폰 접속용 URL)")
    print("[GUI Mode] Opening your default web browser...")
    print("Press Ctrl+C in terminal to stop the GUI server.")
    print("="*60 + "\n")

    def open_browser():
        time.sleep(1.2)
        try:
            webbrowser.open(local_url)
        except Exception as e:
            print(f"[Warning] Could not automatically open browser: {e}")

    threading.Thread(target=open_browser, daemon=True).start()

    import os
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        uvicorn.run("app.main:app", host=host, port=port, reload=False)
    except KeyboardInterrupt:
        print("\n[GUI Mode] Server stopped by user.")
    except Exception as e:
        print(f"\n[Error] Failed to start GUI server: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run online inference using built ROI-RAG index or start GUI.")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--query", "-q", type=str, help="Single query string to answer.")
    group.add_argument("--interactive", "-i", action="store_true", help="Start interactive CLI loop mode.")
    group.add_argument("--gui", "-g", action="store_true", help="Start PROJECT Studio Web GUI and open in browser.")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Port for GUI web server (default: 8000).")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address for GUI web server (default: 127.0.0.1).")
    
    args = parser.parse_args()
    
    # 만약 --query나 --interactive 없이 인자가 없거나 --gui가 선택된 경우 GUI 모드 실행
    if args.gui or (not args.query and not args.interactive):
        if not args.gui:
            print("[Notice] No execution mode specified. Defaulting to GUI mode (--gui).")
            print("         (Use --help to check CLI --query or --interactive options)")
        run_gui_mode(host=args.host, port=args.port)
        return

    print("[Inference] Initializing ROI-RAG Pipeline...")
    pipeline = get_roi_rag_pipeline()
    
    if args.query:
        res = pipeline(args.query)
        print_response(args.query, res)
    elif args.interactive:
        print("\n" + "="*60)
        print("[Interactive] ROI-RAG Interactive CLI Mode Started.")
        print("Type your question below and press Enter. Type 'exit' or 'quit' to stop.")
        print("="*60)
        
        while True:
            try:
                query = input("\n[You] > ").strip()
                if not query:
                    continue
                if query.lower() in ["exit", "quit", "q"]:
                    print("Exiting ROI-RAG Interactive Mode. Goodbye!")
                    break
                    
                res = pipeline(query)
                print_response(query, res)
            except KeyboardInterrupt:
                print("\nExiting CLI.")
                break
            except Exception as e:
                print(f"[Error] {e}")

if __name__ == "__main__":
    main()
