"""
4_agent_system/agents/retrieval_agent.py
Retrieval Agent - RAG 기반 근거 검색
"""
import sys
import os

# 상위 디렉토리 import를 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rag_system import RAGSystem
from utils.state import AgentState

class RetrievalAgent:
    """Retrieval Agent - RAG 기반 근거 검색"""
    
    def __init__(self, knowledge_base_path: str = "./knowledge_base"):
        # 디버깅: 전달된 경로 확인
        print(f"🔍 RetrievalAgent 초기화:")
        print(f"   - 전달된 knowledge_base_path: {knowledge_base_path}")
        print(f"   - type: {type(knowledge_base_path)}")
        
        # ChromaDB 사용하도록 설정
        self.rag = RAGSystem(knowledge_base_path, use_chromadb=True)
        print("✅ Retrieval Agent 초기화 완료")
    
    def run(self, state: AgentState) -> AgentState:
        """RAG 검색 실행"""
        print(f"\n{'='*60}")
        print("📖 Retrieval Agent 실행 중...")
        print(f"{'='*60}")
        
        # 이상이 아니면 스킵
        if not state.get('is_anomaly', False):
            state['messages'].append("Retrieval: 이상 없음, 검색 스킵")
            state['similar_cases'] = []
            state['rag_context'] = ""
            return state
        
        try:
            # 검색 쿼리 구성
            query = f"""
            설비: {state.get('equipment_id', 'Unknown')}
            이상 유형: {state.get('anomaly_type', 'Unknown')}
            센서 데이터: {state.get('sensor_data', {})}
            """
            
            # RAG 검색
            if self.rag is None:
                raise RuntimeError("RAG 시스템이 초기화되지 않았습니다.")
            
            similar_cases = self.rag.search(query, k=3)
            
            # 컨텍스트 구성
            if similar_cases:
                rag_context = "\n\n".join([
                    f"[검색 결과 #{i+1}] (유사도: {case.get('similarity', 0):.1%})\n{case.get('content', '')}"
                    for i, case in enumerate(similar_cases)
                ])
            else:
                rag_context = "검색 결과 없음"
            
            # 상태 업데이트
            state['similar_cases'] = similar_cases
            state['rag_context'] = rag_context
            state['messages'].append(f"Retrieval: {len(similar_cases)}개 유사 사례 검색 완료")
            
            print(f"검색 완료: {len(similar_cases)}개 문서")
            
        except Exception as e:
            print(f"❌ Retrieval Agent 실행 실패: {e}")
            import traceback
            traceback.print_exc()
            
            # 기본값 설정
            state['similar_cases'] = []
            state['rag_context'] = f"[오류] 검색 실패: {str(e)}"
            state['messages'].append(f"Retrieval: 오류 발생 - {str(e)}")
        
        return state