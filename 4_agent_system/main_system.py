"""
4_agent_system/main_system.py
AI 자율 운영 공정 시스템 - 메인 실행 파일

수정 사항: Press 데이터 기반, 경로 자동 탐지
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
                 detection_model_type: str = "ensemble",  # "TimesNet", "AnomalyTransformer", "ensemble", "rule_based"
                 detection_model_path: str = None,  # 학습된 모델 경로 (None=자동 탐지)
                 lora_model_path: str = None,
                 knowledge_base_path: str = None):
        """
        Args:
            detection_model_type: 'TimesNet', 'AnomalyTransformer', 'ensemble', 'rule_based'
            detection_model_path: 학습된 이상 탐지 모델 경로 (None이면 자동 탐지)
            lora_model_path: LoRA 모델 경로 (None이면 자동 탐지)
            knowledge_base_path: RAG 지식 베이스 경로 (None이면 자동 탐지)
        """
        
        print("=" * 80)
        print("🏭 AI 자율 운영 공정 시스템 초기화")
        print("=" * 80)
        
        # 프로젝트 루트 자동 탐지
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 기본 경로 설정 (config 우선, 없으면 자동 탐지)
        try:
            from utils.config import LORA_MODEL_PATH, KNOWLEDGE_BASE_PATH
            if lora_model_path is None:
                lora_model_path = str(LORA_MODEL_PATH)
            if knowledge_base_path is None:
                knowledge_base_path = str(KNOWLEDGE_BASE_PATH)
        except ImportError:
            # config 없을 때 자동 탐지
            if lora_model_path is None:
                lora_model_path = os.path.join(project_root, "2_model_training", "manufacturing_lora_output")
            if knowledge_base_path is None:
                knowledge_base_path = os.path.join(project_root, "3_knowledge_base", "knowledge_base")
        
        # Detection 모델 경로 자동 탐지
        if detection_model_path is None:
            # 모델 타입에 따라 자동 선택
            if detection_model_type == "ensemble":
                detection_model_path = os.path.join(project_root, "2_model_training", "best_ensemble_2models.pkl")
            elif detection_model_type == "TimesNet":
                detection_model_path = os.path.join(project_root, "2_model_training", "best_timesnet.pkl")
            elif detection_model_type == "AnomalyTransformer":
                detection_model_path = os.path.join(project_root, "2_model_training", "best_anomalytransformer.pkl")
            else:
                # rule_based는 경로 불필요
                detection_model_path = None
        
        print(f"📁 경로 설정:")
        print(f"   - Detection 모델: {detection_model_path}")
        print(f"   - LoRA 모델: {lora_model_path}")
        print(f"   - 지식 베이스: {knowledge_base_path}")
        
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
            self.lora_engine = None
        
        # Agents 초기화
        try:
            self.detection_agent = DetectionAgent(
                model_type=detection_model_type,
                model_path=detection_model_path,
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
                - AI0_Vibration: 진동 센서 1 (g)
                - AI1_Vibration: 진동 센서 2 (g)
                - AI2_Current: 전류 센서 (A)
        
        Returns:
            처리 결과 딕셔너리
        """
        
        print(f"\n{'='*80}")
        print(f"🚀 이상 이벤트 처리 시작")
        print(f"{'='*80}")
        print(f"설비: {equipment_id}")
        print(f"시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"센서 데이터:")
        print(f"  - AI0_Vibration: {sensor_data.get('AI0_Vibration', 0):.4f} g")
        print(f"  - AI1_Vibration: {sensor_data.get('AI1_Vibration', 0):.4f} g")
        print(f"  - AI2_Current: {sensor_data.get('AI2_Current', 0):.2f} A")
        
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
    
    print("\n" + "=" * 80)
    print("🏭 AI 자율 운영 공정 시스템 - Press 이상 탐지 데모")
    print("=" * 80)
    
    # 시스템 초기화
    system = ManufacturingAISystem(
        detection_model_type="ensemble",  # TimesNet + AnomalyTransformer 앙상블
        detection_model_path=None  # 자동으로 best_ensemble_2models.pkl 탐지
    )
    
    # 테스트 시나리오 1: 프레스 고진동 + 과전류 이상
    print("\n" + "=" * 80)
    print("테스트 시나리오 1: 프레스 고진동 + 과전류 이상")
    print("=" * 80)
    print("📝 설명: 실제 이상 데이터 기반 (dataset_3/outlier_data.csv)")
    print("   - AI0_Vibration 1.07g: 정상 범위(±0.15g) 대비 7배 초과")
    print("   - AI1_Vibration -0.56g: 정상 범위 대비 3.7배 초과")
    print("   - AI2_Current 243A: 정상 범위(±230A) 약간 초과")

    result1 = system.process_anomaly_event(
        equipment_id="PRESS-01",
        sensor_data={
            "AI0_Vibration": 1.07,    # g (이상: 정상 ±0.15, 위험 ±0.30 이상)
            "AI1_Vibration": -0.56,   # g (이상)
            "AI2_Current": 243.30     # A (이상: 정상 ±230)
        }
    )
    
    # 결과 출력
    print("\n" + "=" * 80)
    print("📄 생성된 8D Report (미리보기)")
    print("=" * 80)
    if result1.get('report_8d'):
        print(result1.get('report_8d'))
        print("\n... (이하 생략) ...")
    else:
        print("⚠️  8D Report가 생성되지 않았습니다. (LoRA 모델 필요)")


    # 테스트 시나리오 2: 프레스 정상 운전
    print("\n\n" + "=" * 80)
    print("테스트 시나리오 2: 프레스 정상 운전")
    print("=" * 80)
    print("📝 설명: 정상 데이터 기반 (dataset_3/press_data_normal.csv)")
    print("   - 모든 센서 값이 정상 범위 내")

    result2 = system.process_anomaly_event(
        equipment_id="PRESS-02",
        sensor_data={
            "AI0_Vibration": 0.02,    # g (정상 범위)
            "AI1_Vibration": -0.01,   # g (정상 범위)
            "AI2_Current": 35.00      # A (정상 범위)
        }
    )

    print("\n✅ 정상 운전 확인")
    
    # 결과 저장
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # 이상 케이스 저장
    if result1.get('is_anomaly'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(output_dir, f"anomaly_result_{timestamp}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result1, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n💾 이상 결과 저장: {output_file}")
        
        # 8D 리포트 저장
        if result1.get('report_8d'):
            report_file = os.path.join(output_dir, f"8D_Report_{timestamp}.txt")
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(result1.get('report_8d'))
            print(f"💾 8D 리포트 저장: {report_file}")
    
    print("\n" + "=" * 80)
    print("✅ 데모 완료!")
    print("=" * 80)
    print("\n📌 다음 단계:")
    print("   1. Streamlit 대시보드 실행:")
    print("      cd 5_dashboard && streamlit run dashboard.py")
    print("\n   2. 결과 확인:")
    print(f"      ls {output_dir}")
    print("\n   3. 모델 재학습 (새 데이터 추가 시):")
    print("      cd 2_model_training && python train_best_2models.py")
    print("=" * 80)


if __name__ == "__main__":
    # 필요한 디렉토리 생성
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    os.makedirs(os.path.join(project_root, "3_knowledge_base", "knowledge_base"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "2_model_training", "manufacturing_lora_output"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "outputs", "results"), exist_ok=True)
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║     🏭 AI 자율 운영 공정 시스템 - Press 이상 탐지 v1.0       ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    📋 시스템 구성:
       - Detection: TimesNet + AnomalyTransformer 앙상블 (Recall 85%+)
       - Retrieval: RAG 기반 지식 베이스 검색
       - Action: LoRA LLM 기반 조치 방안 생성
       - PM: 예방 정비 추천
       - Report: 8D 리포트 자동 생성
    
    ⚠️  사전 요구사항:
       1. 이상 탐지 모델 학습 완료:
          → cd 2_model_training && python train_best_2models.py
       
       2. 지식 베이스 구축 (옵션):
          → cd 3_knowledge_base && python setup_rag.py --rebuild
       
       3. LoRA 모델 학습 (옵션):
          → cd 2_model_training && python train_lora.py
       
       4. 필요 패키지:
          → pip install deepod langchain langgraph faiss-cpu sentence-transformers
    
    🚀 시작합니다...
    """)
    
    # 실행
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
