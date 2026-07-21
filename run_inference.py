"""
ROI-RAG CLI 추론 도구.
사용법:
  # 단발성 질의 실행
  python run_inference.py --query "엔비디아의 주요 매출 동력은 무엇인가요?"
  
  # 대화형 CLI 모드 실행
  python run_inference.py --interactive
"""
import argparse
from roi_rag import get_roi_rag_pipeline

def print_response(query: str, res: dict):
    print("\n" + "="*60)
    print(f"❓ Query: {query}")
    print("="*60)
    print(f"💡 ROI-RAG Answer:\n{res['answer']}")
    print("-" * 60)
    print("📚 Retrieved Evidence Units (Top-K):")
    for i, eu_text in enumerate(res.get("retrieved_contexts", []), 1):
        print(f"\n[{i}] {eu_text}")
    print("-" * 60)
    print(f"⚡ Latency: {res['latency_ms']} ms | API Calls: {res['api_calls']} | Est. Tokens Used: {res['tokens_used']}")
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Run online inference using built ROI-RAG index.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", "-q", type=str, help="Single query string to answer.")
    group.add_argument("--interactive", "-i", action="store_true", help="Start interactive CLI loop mode.")
    
    args = parser.parse_args()
    
    print("[Inference] Initializing ROI-RAG Pipeline...")
    pipeline = get_roi_rag_pipeline()
    
    if args.query:
        res = pipeline(args.query)
        print_response(args.query, res)
    elif args.interactive:
        print("\n" + "="*60)
        print("🤖 ROI-RAG Interactive CLI Mode Started.")
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
