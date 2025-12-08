"""
4_agent_system/main_system.py
AI 자율 운영 공정 시스템 - 메인 실행 파일

수정 사항: Import 경로 수정
"""

import os
import sys
from datetime import datetime
from typing import Dict
import json

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Agent imports
from agents.detection_agent import DetectionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.action_agent import ActionAgent
from agents.pm_agent import PMRecommendationAgent
from agents.report_agent import ReportAgent

# Model imports
from models.lora_inference import LoRAInferenceEngine

# LangGraph
try:
    from langgraph.graph import StateGraph, END
except ImportError:
    print("⚠️  LangGraph 미설치. 설치: pip install langgraph")
    StateGraph = None
    END = None

# Utils
from utils.state import AgentState


class ManufacturingAISystem:
    """전체 Multi-Agent 시스템 오케스트레이터"""
    
    def __init__(self,
                 detection_model_type: str = "TimesNet",  # "TimesNet", "rule_based"
                 lora_model_path: str = None,
                 knowledge_base_path: str = None):
        """
        Args:
            detection_model_type: 'TimesNet', 'AnomalyTransformer', 'TranAD', 'rule_based'
            lora_model_path: LoRA 모델 경로
            knowledge_base_path: RAG 지식 베이스 경로
        """
        
        print("=" * 80)
        print("🏭 AI 자율 운영 공정 시스템 초기화")
        print("=" * 80)
        
        # 기본 경로 설정 (config 사용)
        try:
            from utils.config import LORA_MODEL_PATH, KNOWLEDGE_BASE_PATH
            if lora_model_path is None:
                lora_model_path = str(LORA_MODEL_PATH)
            if knowledge_base_path is None:
                knowledge_base_path = str(KNOWLEDGE_BASE_PATH)
        except ImportError:
            # config 없을 때 폴백
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if lora_model_path is None:
                lora_model_path = os.path.join(project_root, "2_model_training", "manufacturing_lora_output")
            if knowledge_base_path is None:
                knowledge_base_path = os.path.join(project_root, "3_knowledge_base", "knowledge_base")
        
        # LoRA 엔진 초기화
        try:
            self.lora_engine = LoRAInferenceEngine(
                lora_adapter_path=lora_model_path
            )
            if not self.lora_engine.is_loaded:
                print("⚠️  LoRA 모델 로드 실패. Base 모델만 사용합니다.")
                self.lora_engine = None
        except Exception as e:
            print(f"⚠️  LoRA 모델 로드 실패: {e}")
            print("   Base 모델만 사용합니다.")
            import traceback
            traceback.print_exc()
            self.lora_engine = None
        
        # Agents 초기화
        try:
            self.detection_agent = DetectionAgent(
                model_type=detection_model_type,
                seq_len=50
            )
        except Exception as e:
            print(f"⚠️  Detection Agent 초기화 실패: {e}")
            raise
        
        try:
            self.retrieval_agent = RetrievalAgent(knowledge_base_path)
        except Exception as e:
            print(f"⚠️  Retrieval Agent 초기화 실패: {e}")
            raise
        
        try:
            self.action_agent = ActionAgent(self.lora_engine)
        except Exception as e:
            print(f"⚠️  Action Agent 초기화 실패: {e}")
            raise
        
        try:
            self.pm_agent = PMRecommendationAgent()
        except Exception as e:
            print(f"⚠️  PM Agent 초기화 실패: {e}")
            raise
        
        try:
            self.report_agent = ReportAgent(self.lora_engine)
        except Exception as e:
            print(f"⚠️  Report Agent 초기화 실패: {e}")
            raise
        
        # LangGraph 워크플로우 구성
        if StateGraph is not None:
            self.workflow = self._build_workflow()
        else:
            self.workflow = None
            print("⚠️  LangGraph 없음. 순차 실행 모드")
        
        print("\n✅ 시스템 초기화 완료!")
        print("=" * 80)
    
    def _build_workflow(self) -> StateGraph:
        """LangGraph 워크플로우 구성"""
        
        workflow = StateGraph(AgentState)
        
        # 노드 추가
        workflow.add_node("detect", self.detection_agent.run)
        workflow.add_node("retrieve", self.retrieval_agent.run)
        workflow.add_node("action", self.action_agent.run)
        workflow.add_node("pm", self.pm_agent.run)
        workflow.add_node("report", self.report_agent.run)
        
        # 엣지 정의 (순차 실행)
        workflow.set_entry_point("detect")
        workflow.add_edge("detect", "retrieve")
        workflow.add_edge("retrieve", "action")
        workflow.add_edge("action", "pm")
        workflow.add_edge("pm", "report")
        workflow.add_edge("report", END)
        
        return workflow.compile()
    
    def process_anomaly_event(self,
                              equipment_id: str,
                              sensor_data: Dict[str, float]) -> Dict:
        """
        이상 이벤트 처리 (전체 워크플로우 실행)
        
        Args:
            equipment_id: 설비 ID
            sensor_data: 센서 데이터 딕셔너리
        
        Returns:
            처리 결과 딕셔너리
        """
        
        print(f"\n{'='*80}")
        print(f"🚀 이상 이벤트 처리 시작")
        print(f"{'='*80}")
        print(f"설비: {equipment_id}")
        print(f"시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"센서: {sensor_data}")
        
        # 초기 상태 구성
        initial_state = {
            "equipment_id": equipment_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sensor_data": sensor_data,
            "messages": [],
            "workflow_start_time": datetime.now().isoformat(),
            "workflow_status": "running"
        }
        
        # 워크플로우 실행
        start_time = datetime.now()
        
        try:
            if self.workflow is not None:
                # LangGraph 사용
                result = self.workflow.invoke(initial_state)
            else:
                # 순차 실행
                result = self._sequential_execution(initial_state)
            
            # 실행 시간 계산
            elapsed = (datetime.now() - start_time).total_seconds()
            result['workflow_status'] = 'completed'
            result['elapsed_time'] = elapsed
            
            print(f"\n{'='*80}")
            print(f"✅ 워크플로우 완료 (소요 시간: {elapsed:.2f}초)")
            print(f"{'='*80}")
            
            # 결과 요약
            self._print_summary(result)
            
            return result
            
        except Exception as e:
            print(f"\n❌ 워크플로우 실행 실패: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _sequential_execution(self, state: Dict) -> Dict:
        """순차 실행 (LangGraph 없을 때)"""
        
        state = self.detection_agent.run(state)
        state = self.retrieval_agent.run(state)
        state = self.action_agent.run(state)
        state = self.pm_agent.run(state)
        state = self.report_agent.run(state)
        
        return state
    
    def _print_summary(self, result: Dict):
        """결과 요약 출력"""
        print("\n📊 실행 결과 요약:")
        print(f"   - 이상 여부: {'🚨 이상 감지' if result.get('is_anomaly') else '✅ 정상'}")
        
        if result.get('is_anomaly'):
            print(f"   - 이상 유형: {result.get('anomaly_type')}")
            print(f"   - 신뢰도: {result.get('anomaly_score', 0):.1%}")
            print(f"   - 검색된 사례: {len(result.get('similar_cases', []))}개")
            print(f"   - 원인 후보: {len(result.get('root_causes', []))}개")
            print(f"   - 체크리스트: {len(result.get('checklist', []))}개")
            print(f"   - Health Score: {result.get('health_score', 0):.1%}")
            print(f"   - 고장 위험도: {result.get('failure_risk', 0):.1%}")
            print(f"   - PM 추천: {len(result.get('pm_recommendations', []))}개")
        
        print(f"\n🔄 실행 로그:")
        for msg in result.get('messages', []):
            print(f"   {msg}")


# ============================================================================
# 메인 실행 예제
# ============================================================================

def main():
    """메인 실행 함수"""
    
    # 시스템 초기화 (규칙 기반으로 빠른 테스트)
    system = ManufacturingAISystem(
        detection_model_type="rule_based"  # 빠른 테스트용
    )
    
    # 테스트 시나리오 1: 온도 이상
    print("\n" + "=" * 80)
    print("테스트 시나리오 1: 온도 이상")
    print("=" * 80)
    
    result1 = system.process_anomaly_event(
        equipment_id="사출기-2호기",
        sensor_data={
            "temperature": 235.5,
            "pressure": 120.0,
            "vibration": 1.2,
            "cycle_time": 52
        }
    )
    
    # 결과 저장
    output_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "outputs", "results")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "result_temperature_anomaly.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result1, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n✅ 결과 저장: {output_file}")
    
    # 8D Report 출력
    print("\n" + "=" * 80)
    print("📄 생성된 8D Report")
    print("=" * 80)
    print(result1.get('report_8d', 'N/A')[:500] + "...")
    
    
    # 테스트 시나리오 2: 정상 운전
    print("\n\n" + "=" * 80)
    print("테스트 시나리오 2: 정상 운전")
    print("=" * 80)
    
    result2 = system.process_anomaly_event(
        equipment_id="CNC-1호기",
        sensor_data={
            "temperature": 200.0,
            "pressure": 120.0,
            "vibration": 1.0,
            "cycle_time": 50
        }
    )
    
    print("\n✅ 정상 운전 확인")


if __name__ == "__main__":
    # 필요한 디렉토리 생성
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    os.makedirs(os.path.join(project_root, "3_knowledge_base", "knowledge_base"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "2_model_training", "manufacturing_lora_output"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "outputs", "results"), exist_ok=True)
    
    print("""
    ⚠️  주의사항:
    1. Detection Agent: 규칙 기반 또는 DeepOD 학습 필요
       → cd 2_model_training && python test.py --model TimesNet
    
    2. LoRA 모델: train_lora.py 실행 필요
       → cd 2_model_training && python train_lora.py
    
    3. 필요 패키지:
       → pip install langchain langgraph faiss-cpu sentence-transformers deepod
    """)
    
    # 실행
    main()