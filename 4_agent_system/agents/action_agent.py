"""
4_agent_system/agents/action_agent.py
Action Agent - 원인 분석 및 조치 가이드 생성 (LoRA)
"""
import sys
import os

# 상위 디렉토리 import를 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lora_inference import LoRAInferenceEngine
from utils.state import AgentState
from typing import List, Dict

class ActionAgent:
    """Action Agent - 원인 분석 및 조치 가이드 생성 (LoRA)"""
    
    def __init__(self, lora_engine: LoRAInferenceEngine):
        self.lora_engine = lora_engine
        print("✅ Action Agent 초기화 완료")
    
    def run(self, state: AgentState) -> AgentState:
        """조치 가이드 생성"""
        print(f"\n{'='*60}")
        print("🔧 Action Agent 실행 중... (LoRA 모델)")
        print(f"{'='*60}")
        
        # 이상이 아니면 스킵
        if not state.get('is_anomaly', False):
            state['messages'].append("Action: 이상 없음, 조치 스킵")
            state['root_causes'] = []
            state['action_guide'] = "정상 운전 중입니다."
            state['checklist'] = []
            return state
        
        try:
            # 프롬프트 구성
            instruction = "당신은 제조 현장의 전문 설비 엔지니어입니다. 공정 이상 상황에 대해 원인을 분석하고, 구체적인 조치 가이드를 작성해주세요."
            
            input_text = f"""[공정 이상 이벤트]
설비: {state.get('equipment_id', 'Unknown')}
발생시각: {state.get('timestamp', 'Unknown')}
이상유형: {state.get('anomaly_type', 'Unknown')}

[센서 데이터]
{self._format_sensor_data(state.get('sensor_data', {}))}

[RAG 검색 결과]
{state.get('rag_context', '검색 결과 없음')}

위 정보를 바탕으로 원인 분석과 조치 가이드를 작성하세요."""
            
            # LoRA 모델로 생성
            if self.lora_engine is None:
                raise RuntimeError("LoRA 엔진이 초기화되지 않았습니다.")
            
            action_guide = self.lora_engine.generate(instruction, input_text)
            
            # 파싱
            root_causes = self._parse_root_causes(action_guide)
            checklist = self._parse_checklist(action_guide)
            
            # 상태 업데이트
            state['root_causes'] = root_causes
            state['action_guide'] = action_guide
            state['checklist'] = checklist
            state['messages'].append("Action: 조치 가이드 생성 완료")
            
            print(f"✅ 생성 완료: 원인 {len(root_causes)}개, 체크리스트 {len(checklist)}개")
            
        except Exception as e:
            print(f"❌ Action Agent 실행 실패: {e}")
            import traceback
            traceback.print_exc()
            
            # 기본값 설정
            state['root_causes'] = [{'rank': 1, 'cause': '분석 중', 'probability': '미정'}]
            state['action_guide'] = f"[오류] 조치 가이드 생성 실패: {str(e)}"
            state['checklist'] = ['체크리스트 생성 중']
            state['messages'].append(f"Action: 오류 발생 - {str(e)}")
        
        return state
    
    def _format_sensor_data(self, sensor_data: Dict) -> str:
        """센서 데이터 포맷팅"""
        return "\n".join([f"- {k}: {v}" for k, v in sensor_data.items()])
    
    def _parse_root_causes(self, text: str) -> List[Dict]:
        """원인 후보 파싱"""
        causes = []
        lines = text.split('\n')
        
        for line in lines:
            if '1순위:' in line or '**1순위:' in line:
                cause = line.split(':')[1].strip().replace('**', '')
                causes.append({'rank': 1, 'cause': cause, 'probability': '높음'})
        
        return causes if causes else [{'rank': 1, 'cause': '분석 중', 'probability': '미정'}]
    
    def _parse_checklist(self, text: str) -> List[str]:
        """체크리스트 파싱"""
        checklist = []
        lines = text.split('\n')
        
        for line in lines:
            if line.strip().startswith('□'):
                item = line.strip()[2:].strip()
                checklist.append(item)
        
        return checklist if checklist else ['체크리스트 생성 중']