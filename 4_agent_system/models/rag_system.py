"""
4_agent_system/models/rag_system.py
RAG 시스템 - 과거 이력, 매뉴얼 검색
"""

import os
from typing import List, Dict, Optional
import torch

try:
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.docstore.document import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️  LangChain 미설치: pip install langchain faiss-cpu sentence-transformers")

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️  ChromaDB 미설치: pip install chromadb")


class RAGSystem:
    """RAG 시스템 - 과거 이력, 매뉴얼 검색"""
    
    def __init__(self, knowledge_base_path: str = None, use_chromadb: bool = False):
        """
        Args:
            knowledge_base_path: 지식 베이스 경로 (None이면 자동 탐지)
            use_chromadb: ChromaDB 사용 여부 (True면 ChromaDB, False면 FAISS)
        """
        # 경로 자동 설정
        if knowledge_base_path is None:
            from ..utils.config import KNOWLEDGE_BASE_PATH, VECTOR_DB_PATH
            knowledge_base_path = str(KNOWLEDGE_BASE_PATH)
            self.vector_db_path = str(VECTOR_DB_PATH)
        else:
            self.vector_db_path = None
        
        self.knowledge_base_path = knowledge_base_path
        self.use_chromadb = use_chromadb and CHROMADB_AVAILABLE
        self.embeddings = None
        self.vectorstore = None
        self.chroma_collection = None
        self.documents = []
        self.is_loaded = False
        
        if not LANGCHAIN_AVAILABLE:
            print("⚠️  LangChain 없음. RAG 기능을 사용할 수 없습니다.")
            return
        
        print("📚 RAG 시스템 초기화 중...")
        self._initialize_embeddings()
        self._load_knowledge_base()
    
    def _initialize_embeddings(self):
        """임베딩 모델 로드"""
        if not LANGCHAIN_AVAILABLE:
            return
        
        try:
            from ..utils.config import RAG_CONFIG
            embedding_model = RAG_CONFIG.get("embedding_model", "BAAI/bge-m3")
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
            )
            print(f"✅ 임베딩 모델 로드 완료 ({embedding_model})")
        except Exception as e:
            print(f"⚠️  임베딩 모델 로드 실패: {e}")
            # Fallback
            try:
                from ..utils.config import RAG_CONFIG
                fallback_model = RAG_CONFIG.get("fallback_embedding_model", 
                                                "sentence-transformers/all-MiniLM-L6-v2")
                self.embeddings = HuggingFaceEmbeddings(model_name=fallback_model)
                print(f"✅ Fallback 임베딩 모델 로드: {fallback_model}")
            except Exception as e2:
                print(f"❌ Fallback 임베딩 모델도 로드 실패: {e2}")
                self.embeddings = None
    
    def _load_knowledge_base(self):
        """지식 베이스 로드 및 벡터화"""
        if not LANGCHAIN_AVAILABLE or self.embeddings is None:
            print("⚠️  LangChain 또는 임베딩 모델 없음. 샘플 문서만 사용합니다.")
            self._load_sample_documents()
            return
        
        os.makedirs(self.knowledge_base_path, exist_ok=True)
        
        # ChromaDB 사용 시도
        if self.use_chromadb and self.vector_db_path:
            try:
                self._load_from_chromadb()
                if self.is_loaded:
                    return
            except Exception as e:
                print(f"⚠️  ChromaDB 로드 실패: {e}")
                print("   FAISS로 전환합니다.")
        
        # 파일에서 로드 시도
        try:
            self._load_from_files()
            if self.is_loaded:
                return
        except Exception as e:
            print(f"⚠️  파일에서 로드 실패: {e}")
        
        # 샘플 문서로 폴백
        print("⚠️  지식 베이스 파일 없음. 샘플 문서를 사용합니다.")
        self._load_sample_documents()
    
    def _load_from_chromadb(self):
        """ChromaDB에서 로드"""
        if not CHROMADB_AVAILABLE or not self.vector_db_path:
            return
        
        try:
            client = chromadb.PersistentClient(
                path=str(self.vector_db_path),
                settings=Settings(allow_reset=False)
            )
            collection = client.get_collection("manufacturing_kb")
            self.chroma_collection = collection
            self.is_loaded = True
            print(f"✅ ChromaDB에서 지식 베이스 로드 완료")
        except Exception:
            pass
    
    def _load_from_files(self):
        """파일에서 지식 베이스 로드"""
        from pathlib import Path
        kb_path = Path(self.knowledge_base_path)
        
        if not kb_path.exists():
            return
        
        # 실제 파일 로드 로직은 setup_rag.py 참고
        # 여기서는 일단 샘플만 사용
        pass
    
    def _load_sample_documents(self):
        """샘플 문서 로드 (폴백)"""
        if not LANGCHAIN_AVAILABLE or self.embeddings is None:
            return
        
        sample_docs = self._create_sample_documents()
        
        # 문서 분할
        from ..utils.config import RAG_CONFIG
        chunk_size = RAG_CONFIG.get("chunk_size", 500)
        chunk_overlap = RAG_CONFIG.get("chunk_overlap", 50)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.documents = []
        for doc_dict in sample_docs:
            doc = Document(
                page_content=doc_dict['content'],
                metadata=doc_dict['metadata']
            )
            self.documents.append(doc)
        
        splits = text_splitter.split_documents(self.documents)
        
        # Vector DB 생성
        if len(splits) > 0:
            self.vectorstore = FAISS.from_documents(splits, self.embeddings)
            self.is_loaded = True
            print(f"✅ 샘플 지식 베이스 로드 완료 ({len(splits)}개 청크)")
        else:
            print("⚠️  지식 베이스 문서 없음")
    
    def _create_sample_documents(self) -> List[Dict]:
        """샘플 지식 베이스 문서 생성"""
        return [
            {
                'content': """
                [과거 이력 #2023-08-15]
                설비: 사출기-2호기
                증상: 실린더 온도 급상승 (235°C)
                원인: 히터 코일 단선
                조치: 히터 교체 후 정상화
                소요시간: 4시간
                """,
                'metadata': {
                    'type': '과거_이력',
                    'equipment': '사출기',
                    'anomaly': '온도_이상'
                }
            },
            {
                'content': """
                [설비 매뉴얼 3.2절 - 온도 관리]
                실린더 온도가 설정값 ±15°C를 벗어날 경우:
                1. 히터 저항값 측정 (정상: 30~35Ω)
                2. 열전대 센서 점검
                3. 온도 제어기 파라미터 확인
                긴급 조치: 즉시 설비 정지
                """,
                'metadata': {
                    'type': '매뉴얼',
                    'equipment': '사출기',
                    'section': '온도_관리'
                }
            },
            {
                'content': """
                [Trouble Shooting Guide]
                압력 이상 발생 시 점검 순서:
                1. 유압펌프 압력 게이지 확인
                2. 실린더 씰 누유 점검
                3. 배관 연결부 점검
                4. 압력 센서 교정
                주의: 압력이 정상 범위의 70% 이하면 즉시 정지
                """,
                'metadata': {
                    'type': 'Trouble_Shooting',
                    'equipment': '전체',
                    'anomaly': '압력_이상'
                }
            },
            {
                'content': """
                [정비 이력 DB]
                최근 6개월 진동 이상 케이스:
                - 베어링 마모: 5건 (평균 복구 시간 6시간)
                - 구동부 언밸런스: 3건 (평균 복구 시간 4시간)
                - 체결부 풀림: 2건 (평균 복구 시간 2시간)
                예방 조치: 월 1회 진동 측정 및 베어링 그리스 주입
                """,
                'metadata': {
                    'type': '정비_이력',
                    'anomaly': '진동_이상',
                    'period': '최근_6개월'
                }
            },
            {
                'content': """
                [8D Report 예시 #2024-03-20]
                D1: 팀 구성 - 생산기술팀, 품질팀
                D2: 문제 정의 - 사출기 3호기 온도 이상
                D3: 임시 조치 - 설비 정지, 생산품 격리
                D4: 근본 원인 - 냉각수 순환 펌프 고장
                D5: 영구 대책 - 펌프 교체, 예비품 확보
                D6: 실행 및 검증 - 48시간 모니터링 정상
                D7: 재발 방지 - PM 주기 조정, 센서 추가
                """,
                'metadata': {
                    'type': '8D_Report',
                    'equipment': '사출기',
                    'date': '2024-03-20'
                }
            }
        ]
    
    def search(self, query: str, k: int = None) -> List[Dict]:
        """
        유사 문서 검색
        
        Args:
            query: 검색 쿼리
            k: 검색할 문서 수 (None이면 설정값 사용)
        
        Returns:
            검색 결과 리스트
        """
        if k is None:
            from ..utils.config import RAG_CONFIG
            k = RAG_CONFIG.get("search_k", 3)
        
        # ChromaDB 사용
        if self.use_chromadb and self.chroma_collection is not None:
            try:
                results = self.chroma_collection.query(
                    query_texts=[query],
                    n_results=k
                )
                
                retrieved = []
                if results['documents'] and len(results['documents'][0]) > 0:
                    for i, (doc, metadata) in enumerate(zip(
                        results['documents'][0],
                        results['metadatas'][0] if results['metadatas'] else [{}] * len(results['documents'][0])
                    )):
                        retrieved.append({
                            'content': doc,
                            'metadata': metadata,
                            'similarity': 0.9 - (i * 0.1)  # 간단한 유사도 추정
                        })
                return retrieved
            except Exception as e:
                print(f"⚠️  ChromaDB 검색 실패: {e}")
        
        # FAISS 사용
        if self.vectorstore is not None:
            try:
                results = self.vectorstore.similarity_search_with_score(query, k=k)
                
                retrieved = []
                for doc, score in results:
                    retrieved.append({
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'similarity': float(1 - score) if score <= 1.0 else float(1 / (1 + score))  # 거리를 유사도로 변환
                    })
                
                return retrieved
            except Exception as e:
                print(f"⚠️  FAISS 검색 실패: {e}")
        
        # 검색 실패 시 빈 리스트 반환
        print("⚠️  벡터 스토어 없음. 검색 결과 없음")
        return []