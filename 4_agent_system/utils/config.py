"""
설정 파일 - 경로 및 하이퍼파라미터 중앙 관리
"""
import os
from pathlib import Path

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 모델 경로
LORA_MODEL_PATH = PROJECT_ROOT / "2_model_training" / "manufacturing_lora_output"
ANOMALY_MODEL_PATH = PROJECT_ROOT / "2_model_training" / "anomaly_model"
KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "3_knowledge_base" / "knowledge_base"
VECTOR_DB_PATH = PROJECT_ROOT / "3_knowledge_base" / "vector_db"

# 출력 경로
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = OUTPUT_DIR / "reports"
LOGS_DIR = OUTPUT_DIR / "logs"
RESULTS_DIR = OUTPUT_DIR / "results"

# Base 모델 설정
BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# LoRA 설정
LORA_CONFIG = {
    "base_model_path": BASE_MODEL_NAME,
    "max_new_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1
}

# RAG 설정
RAG_CONFIG = {
    "embedding_model": "BAAI/bge-m3",
    "fallback_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "search_k": 3
}

# 이상 탐지 설정
DETECTION_CONFIG = {
    "threshold": 0.7,
    "seq_len": 50,
    "model_types": ["TimesNet", "AnomalyTransformer", "TranAD", "rule_based"]
}

# 디렉토리 자동 생성
for path in [OUTPUT_DIR, REPORTS_DIR, LOGS_DIR, RESULTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

