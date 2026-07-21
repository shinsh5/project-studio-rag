"""
ROI-RAG CLI 인덱스 생성 도구.
사용법:
  python build_index.py --file path/to/document.txt
  python build_index.py --text "직접 입력할 텍스트 내용"
"""
import os
import argparse
from indexer import build_roi_rag_index

def main():
    parser = argparse.ArgumentParser(description="Build ROI-RAG Offline Index from text or document file.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", "-f", type=str, help="Path to text document file (.txt) to index.")
    group.add_argument("--text", "-t", type=str, help="Direct text input string to index.")
    
    args = parser.parse_args()
    
    if args.file:
        if not os.path.exists(args.file):
            print(f"[Error] File not found: {args.file}")
            return
        print(f"[Build Index] Reading text from file: {args.file}")
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = args.text
        
    if not text.strip():
        print("[Error] Input text is empty.")
        return
        
    print("[Build Index] Executing ROI-RAG Indexer...")
    result = build_roi_rag_index(text)
    
    segments_cnt = len(result.get("segments", []))
    eus_cnt = len(result.get("evidence_units", []))
    
    print("\n" + "="*50)
    print("        ROI-RAG Index Building Summary")
    print("="*50)
    print(f"Total Text Segments Chunked : {segments_cnt}")
    print(f"Total Evidence Units Formed : {eus_cnt}")
    
    regime_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    for eu in result.get("evidence_units", []):
        regime = eu.get("regime", "LOW")
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
    print(f"Evidence Unit Redundancy Regimes:")
    print(f"  - LOW (Bypass LLM)          : {regime_counts['LOW']}")
    print(f"  - MEDIUM (Synthesize Facts) : {regime_counts['MEDIUM']}")
    print(f"  - HIGH (Dense Summary)      : {regime_counts['HIGH']}")
    print("="*50)
    print("[Success] ROI-RAG index ready for inference.")

if __name__ == "__main__":
    main()
