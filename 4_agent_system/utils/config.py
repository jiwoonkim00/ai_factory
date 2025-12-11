"""
4_agent_system/models/rag_system.py
RAG 시스템 - 과거 이력, 매뉴얼 검색
"""


from typing import List, Dict, Optional
import torch

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    print(f"⚠️  LangChain 모듈 로드 실패: {e}")
    print("   필요 패키지: pip install langchain langchain-community langchain-text-splitters faiss-cpu sentence-transformers")

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
            try:
                from utils.config import KNOWLEDGE_BASE_PATH, VECTOR_DB_PATH
                knowledge_base_path = str(KNOWLEDGE_BASE_PATH)
                self.vector_db_path = str(VECTOR_DB_PATH)
            except ImportError:
                # config 로드 실패 시 기본 경로 사용
                
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                knowledge_base_path = os.path.join(project_root, "3_knowledge_base", "knowledge_base")
                self.vector_db_path = os.path.join(project_root, "3_knowledge_base", "vector_db")
        else:
            # knowledge_base_path가 전달된 경우에도 vector_db_path 설정
            print(f"🔍 knowledge_base_path가 전달됨: {knowledge_base_path}")
            try:
                from utils.config import VECTOR_DB_PATH
                print(f"🔍 config.VECTOR_DB_PATH = {VECTOR_DB_PATH}")
                self.vector_db_path = str(VECTOR_DB_PATH)
                print(f"🔍 설정된 vector_db_path (from config): {self.vector_db_path}")
            except ImportError as e:
                # config 없으면 knowledge_base_path 기준으로 자동 설정
                print(f"🔍 config import 실패: {e}, 자동 계산 사용")
                kb_parent = os.path.dirname(knowledge_base_path)
                self.vector_db_path = os.path.join(kb_parent, "vector_db")
                print(f"🔍 설정된 vector_db_path (자동): {self.vector_db_path}")
        
        self.knowledge_base_path = knowledge_base_path
        self.use_chromadb = use_chromadb and CHROMADB_AVAILABLE
        self.embeddings = None
        self.embedding_model = None  # sentence-transformers 모델 (ChromaDB용)
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
            # config 파일에서 설정 읽기
            embedding_model = "BAAI/bge-m3"
            try:
                from utils.config import RAG_CONFIG
                embedding_model = RAG_CONFIG.get("embedding_model", embedding_model)
            except:
                pass
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
            )
            print(f"✅ 임베딩 모델 로드 완료 ({embedding_model})")
        except Exception as e:
            print(f"⚠️  임베딩 모델 로드 실패: {e}")
            # Fallback
            try:
                fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
                try:
                    from utils.config import RAG_CONFIG
                    fallback_model = RAG_CONFIG.get("fallback_embedding_model", fallback_model)
                except:
                    pass
                
                self.embeddings = HuggingFaceEmbeddings(model_name=fallback_model)
                print(f"✅ Fallback 임베딩 모델 로드: {fallback_model}")
            except Exception as e2:
                print(f"❌ Fallback 임베딩 모델도 로드 실패: {e2}")
                self.embeddings = None
    
    def _load_knowledge_base(self):
        """지식 베이스 로드 및 벡터화"""
        os.makedirs(self.knowledge_base_path, exist_ok=True)
        
        # 디버깅 정보
        print(f"🔍 RAG 로딩 설정:")
        print(f"   - use_chromadb: {self.use_chromadb}")
        print(f"   - vector_db_path: {self.vector_db_path}")
        print(f"   - CHROMADB_AVAILABLE: {CHROMADB_AVAILABLE}")
        
        # ChromaDB 우선 시도 (LangChain 불필요)
        if self.use_chromadb and self.vector_db_path and CHROMADB_AVAILABLE:
            print(f"🔄 ChromaDB 로드 시도 중: {self.vector_db_path}")
            try:
                self._load_from_chromadb()
                if self.is_loaded:
                    print(f"✅ ChromaDB에서 성공적으로 로드됨")
                    return
                else:
                    print(f"⚠️  ChromaDB 로드했지만 is_loaded=False")
            except Exception as e:
                print(f"⚠️  ChromaDB 로드 실패: {e}")
                import traceback
                traceback.print_exc()
        elif self.use_chromadb:
            if not CHROMADB_AVAILABLE:
                print(f"⚠️  ChromaDB가 설치되지 않음 (pip install chromadb)")
            if not self.vector_db_path:
                print(f"⚠️  vector_db_path가 None임")
        
        # FAISS 사용 시에는 LangChain 필요
        if not LANGCHAIN_AVAILABLE or self.embeddings is None:
            print("⚠️  LangChain 없음. ChromaDB 또는 샘플 문서만 사용 가능합니다.")
            if not self.is_loaded:
                self._load_sample_documents_simple()
            return
        
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
            print(f"⚠️  ChromaDB 조건 미충족: AVAILABLE={CHROMADB_AVAILABLE}, path={self.vector_db_path}")
            return
        
        try:
            # ChromaDB 클라이언트 연결
            print(f"🔗 ChromaDB 연결 중: {self.vector_db_path}")
            client = chromadb.PersistentClient(
                path=str(self.vector_db_path),
                settings=Settings(allow_reset=False)
            )
            
            # 컬렉션 존재 확인
            collections = client.list_collections()
            collection_names = [c.name for c in collections]
            print(f"📋 사용 가능한 컬렉션: {collection_names}")
            
            if "manufacturing_kb" not in collection_names:
                print(f"⚠️  'manufacturing_kb' 컬렉션을 찾을 수 없습니다.")
                print(f"   ChromaDB를 구축하려면: cd 3_knowledge_base && python setup_rag.py")
                return
            
            # 컬렉션 로드
            collection = client.get_collection("manufacturing_kb")
            self.chroma_collection = collection
            
            # 컬렉션 정보 확인
            count = collection.count()
            print(f"📚 컬렉션 문서 수: {count}개")
            
            if count == 0:
                print(f"⚠️  컬렉션이 비어있습니다.")
                return
            
            # ChromaDB 검색을 위한 임베딩 모델 초기화 (sentence-transformers 직접 사용)
            try:
                from sentence_transformers import SentenceTransformer
                
                # 여러 모델 시도 (fallback)
                models_to_try = [
                    "sentence-transformers/all-MiniLM-L6-v2",  # 가볍고 안정적
                    "BAAI/bge-m3",  # 고성능
                ]
                
                for embedding_model in models_to_try:
                    try:
                        print(f"🔄 임베딩 모델 로드 시도: {embedding_model}")
                        self.embedding_model = SentenceTransformer(embedding_model)
                        print(f"✅ ChromaDB용 임베딩 모델 로드 완료: {embedding_model}")
                        break
                    except Exception as e:
                        print(f"⚠️  {embedding_model} 로드 실패: {e}")
                        continue
                
                if self.embedding_model is None:
                    print(f"⚠️  모든 임베딩 모델 로드 실패. ChromaDB 기본 임베딩 사용")
                    
            except Exception as e:
                print(f"⚠️  sentence-transformers 로드 실패: {e}")
                print(f"   pip install sentence-transformers 필요")
                self.embedding_model = None
            
            # 성공
            self.is_loaded = True
            print(f"✅ ChromaDB에서 지식 베이스 로드 완료 ({count}개 문서)")
            
        except Exception as e:
            print(f"❌ ChromaDB 연결 실패: {e}")
            import traceback
            traceback.print_exc()
            self.is_loaded = False
    
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
        chunk_size = 500
        chunk_overlap = 50
        try:
            from utils.config import RAG_CONFIG
            chunk_size = RAG_CONFIG.get("chunk_size", chunk_size)
            chunk_overlap = RAG_CONFIG.get("chunk_overlap", chunk_overlap)
        except:
            pass
        
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
    
    def _load_sample_documents_simple(self):
        """LangChain 없이 간단한 샘플 문서 로드"""
        self.documents = self._create_sample_documents()
        self.is_loaded = True
        print(f"✅ 샘플 지식 베이스 로드 완료 ({len(self.documents)}개 문서, LangChain 없음)")
    
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
            k = 3  # 기본값
            try:
                from utils.config import RAG_CONFIG
                k = RAG_CONFIG.get("search_k", k)
            except:
                pass
        
        # ChromaDB 사용
        if self.use_chromadb and self.chroma_collection is not None:
            try:
                # 임베딩 생성 후 검색
                if self.embedding_model is not None:
                    query_embedding = self.embedding_model.encode([query]).tolist()
                    results = self.chroma_collection.query(
                        query_embeddings=query_embedding,
                        n_results=k
                    )
                else:
                    # 임베딩 모델 없으면 텍스트 직접 사용 (ChromaDB 기본 임베딩)
                    results = self.chroma_collection.query(
                        query_texts=[query],
                        n_results=k
                    )
                
                retrieved = []
                if results['documents'] and len(results['documents'][0]) > 0:
                    for i, (doc, metadata, distance) in enumerate(zip(
                        results['documents'][0],
                        results['metadatas'][0] if results['metadatas'] else [{}] * len(results['documents'][0]),
                        results['distances'][0] if results.get('distances') else [0] * len(results['documents'][0])
                    )):
                        # cosine distance를 similarity로 변환 (0=동일, 2=완전반대)
                        similarity = 1 - (distance / 2) if distance else 0.9 - (i * 0.1)
                        retrieved.append({
                            'content': doc,
                            'metadata': metadata,
                            'similarity': max(0, min(1, similarity))  # 0~1 범위로 제한
                        })
                return retrieved
            except Exception as e:
                print(f"⚠️  ChromaDB 검색 실패: {e}")
                import traceback
                traceback.print_exc()
        
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
        
        # 벡터 스토어가 없으면 샘플 문서에서 키워드 검색 (폴백)
        if self.documents:
            print("⚠️  벡터 검색 불가. 키워드 기반 검색을 시도합니다.")
            retrieved = []
            query_lower = query.lower()
            for doc in self.documents[:k]:
                content = doc.get('content', '')
                if any(keyword in content.lower() for keyword in query_lower.split()):
                    retrieved.append({
                        'content': content,
                        'metadata': doc.get('metadata', {}),
                        'similarity': 0.5  # 키워드 매칭이므로 낮은 유사도
                    })
            return retrieved[:k]
        
        # 완전 실패
        print("⚠️  검색 시스템 없음. 검색 결과 없음")
        return []