"""
RAG용 지식 베이스 구축 스크립트
- knowledge_base/ 폴더의 문서들을 읽어서
- 문단 단위로 청크를 자르고
- 임베딩을 생성해
- vector_db/ 폴더에 ChromaDB로 저장

사용 예시:
    # 최초 구축
    python setup_rag.py

    # 기존 벡터DB 삭제 후 재구축
    python setup_rag.py --rebuild

    # 로그 더 자세히
    python setup_rag.py --verbose
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from pypdf import PdfReader
from docx import Document as DocxDocument

# -----------------------
# 기본 설정
# -----------------------

BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "knowledge_base"      # 문서 폴더
VECTOR_DB_DIR = BASE_DIR / "vector_db"    # 벡터 DB 폴더
COLLECTION_NAME = "manufacturing_kb"

# 사용할 임베딩 모델 (멀티링구얼)
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"


# -----------------------
# 유틸 함수들
# -----------------------

def log(msg: str, verbose: bool = True):
    if verbose:
        print(msg)


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(texts)


def read_docx_file(path: Path) -> str:
    doc = DocxDocument(str(path))
    return "\n".join([p.text for p in doc.paragraphs])


def read_json_file(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    # 문서 구조가 다양할 수 있으니, 일단 전체를 텍스트로 풀어서 저장
    return json.dumps(data, ensure_ascii=False, indent=2)


def read_jsonl_file(path: Path) -> List[Dict]:
    """jsonl은 한 줄 = 한 케이스로 보고, 각 라인을 별도 문서로 취급"""
    docs = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                docs.append(obj)
            except Exception:
                continue
    return docs


def load_text_from_path(path: Path, verbose: bool = False):
    """
    파일 확장자에 따라 텍스트를 읽어오는 함수
    - .jsonl 은 여러 문서
    - 나머지는 한 파일 = 한 문서
    """
    suffix = path.suffix.lower()

    if suffix in [".md", ".txt"]:
        return [read_text_file(path)]
    if suffix == ".pdf":
        return [read_pdf_file(path)]
    if suffix in [".docx", ".doc"]:
        return [read_docx_file(path)]
    if suffix == ".json":
        return [read_json_file(path)]
    if suffix == ".jsonl":
        jsonl_docs = read_jsonl_file(path)
        docs = []
        for obj in jsonl_docs:
            # object에 text 필드가 있으면 우선 사용
            if isinstance(obj, dict):
                if "text" in obj:
                    docs.append(str(obj["text"]))
                else:
                    docs.append(json.dumps(obj, ensure_ascii=False))
            else:
                docs.append(str(obj))
        return docs

    log(f"[WARN] 지원하지 않는 확장자: {path.name} (무시)", verbose)
    return []


def split_into_chunks(text: str,
                      max_chars: int = 800,
                      min_chars: int = 200) -> List[str]:
    """
    단순 문자 길이 기준 청크 분할
    - 문단 단위(빈 줄)를 기준으로 자르고
    - 너무 긴 문단은 다시 잘라서 max_chars 근처로 분할
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []

    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
        else:
            # 너무 긴 문단은 max_chars 기준으로 슬라이스
            start = 0
            while start < len(para):
                end = start + max_chars
                chunk = para[start:end]
                chunks.append(chunk)
                start = end

    # 너무 짧은 청크는 앞/뒤와 합쳐서 조금 더 길게 만드는 것도 가능하지만
    # 일단은 그대로 사용
    filtered = [c for c in chunks if len(c) >= min_chars]
    # 너무 짧은 게 많으면 최소 길이 조건을 조금 완화할 수 있음
    if not filtered and chunks:
        filtered = chunks
    return filtered


# -----------------------
# 메인 로직
# -----------------------

def build_knowledge_base(rebuild: bool = False, verbose: bool = True):
    # 1. 벡터 DB 초기화
    if rebuild and VECTOR_DB_DIR.exists():
        log(f"[INFO] 기존 vector_db 삭제: {VECTOR_DB_DIR}", verbose)
        shutil.rmtree(VECTOR_DB_DIR)

    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    log(f"[INFO] ChromaDB 초기화: {VECTOR_DB_DIR}", verbose)
    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_DIR),
        settings=Settings(allow_reset=True)
    )

    # 기존 콜렉션 삭제(옵션)
    try:
        client.delete_collection(COLLECTION_NAME)
        log(f"[INFO] 기존 컬렉션 삭제: {COLLECTION_NAME}", verbose)
    except Exception:
        pass

    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # 2. 임베딩 모델 로드
    log(f"[INFO] 임베딩 모델 로드 중: {EMBEDDING_MODEL_NAME}", verbose)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    log("[INFO] 임베딩 모델 로드 완료", verbose)

    # 3. knowledge_base 폴더 순회
    if not KB_DIR.exists():
        raise FileNotFoundError(f"knowledge_base 폴더가 없습니다: {KB_DIR}")

    log(f"[INFO] 지식 베이스 스캔 시작: {KB_DIR}", verbose)
    all_files: List[Path] = []
    for ext in ["*.md", "*.txt", "*.pdf", "*.docx", "*.doc", "*.json", "*.jsonl"]:
        all_files.extend(KB_DIR.rglob(ext))

    if not all_files:
        log("[WARN] 처리할 문서를 찾지 못했습니다.", verbose)
        return

    log(f"[INFO] 발견된 문서 수: {len(all_files)}", verbose)

    doc_count = 0
    chunk_count = 0
    batch_docs = []
    batch_ids = []
    batch_metadatas = []

    BATCH_SIZE = 32
    global_id = 0

    for file_path in all_files:
        rel_path = file_path.relative_to(KB_DIR)
        category = rel_path.parts[0] if len(rel_path.parts) > 1 else "root"

        log(f"\n[FILE] {rel_path}", verbose)

        texts = load_text_from_path(file_path, verbose=verbose)
        if not texts:
            continue

        for idx, raw_text in enumerate(texts):
            doc_count += 1
            chunks = split_into_chunks(raw_text)
            log(f"  - 문서 {idx+1} → 청크 {len(chunks)}개", verbose)
            if not chunks:
                continue

            for chunk in chunks:
                chunk_id = f"kb_{global_id}"
                metadata = {
                    "source_file": str(rel_path),
                    "category": category,
                    "doc_index": idx,
                    "chunk_index": len(batch_docs),  # 배치 내부 번호
                }

                batch_docs.append(chunk)
                batch_ids.append(chunk_id)
                batch_metadatas.append(metadata)

                global_id += 1
                chunk_count += 1

                # 배치 단위로 insert
                if len(batch_docs) >= BATCH_SIZE:
                    embeddings = model.encode(batch_docs, show_progress_bar=False).tolist()
                    collection.add(
                        ids=batch_ids,
                        documents=batch_docs,
                        metadatas=batch_metadatas,
                        embeddings=embeddings,
                    )
                    batch_docs, batch_ids, batch_metadatas = [], [], []

    # 마지막 남은 배치 처리
    if batch_docs:
        embeddings = model.encode(batch_docs, show_progress_bar=False).tolist()
        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metadatas,
            embeddings=embeddings,
        )

    log("\n====================================", verbose)
    log("✅ RAG 지식 베이스 구축 완료!", verbose)
    log(f"   - 처리 문서 수: {doc_count}", verbose)
    log(f"   - 전체 청크 수: {chunk_count}", verbose)
    log(f"   - 벡터DB 경로: {VECTOR_DB_DIR}", verbose)
    log("====================================", verbose)


# -----------------------
# CLI entry
# -----------------------

def parse_args():
    parser = argparse.ArgumentParser(description="RAG용 지식 베이스 구축 스크립트")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="기존 vector_db 삭제 후 재구축"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="로그 자세히 출력"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_knowledge_base(rebuild=args.rebuild, verbose=True if args.verbose else True)
