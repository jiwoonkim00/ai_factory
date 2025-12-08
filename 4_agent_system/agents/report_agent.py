"""
4_agent_system/agents/report_agent.py
Report Agent - 8D 보고서 자동 생성 (LoRA)
"""
import sys
import os

# 상위 디렉토리 import를 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lora_inference import LoRAInferenceEngine
from utils.state import AgentState
from typing import Dict, List

class ReportAgent:
    """Report Agent - 8D 보고서 생성 (LoRA)"""
    
    def __init__(self, lora_engine: LoRAInferenceEngine):
        self.lora_engine = lora_engine
        print("✅ Report Agent 초기화 완료")
    
    def run(self, state: AgentState) -> AgentState:
        """8D 보고서 생성"""
        print(f"\n{'='*60}")
        print("📄 Report Agent 실행 중... (LoRA)")
        print(f"{'='*60}")
        
        if not state.get('is_anomaly', False):
            state['report_8d'] = f"""[정상 운전 보고서]

설비: {state.get('equipment_id', 'Unknown')}
점검 시각: {state.get('timestamp', 'Unknown')}
상태: 정상

모든 센서 값이 정상 범위 내에 있습니다.
"""
            state['messages'].append("Report: 정상 보고서")
            return state
        
        try:
            # 프롬프트 구성
            instruction = "당신은 제조 현장의 품질 관리 전문가입니다. 8D Report를 작성해주세요."
            
            input_text = f"""[이상 상황 요약]
설비: {state.get('equipment_id', 'Unknown')}
발생시각: {state.get('timestamp', 'Unknown')}
이상유형: {state.get('anomaly_type', 'Unknown')}

[원인 분석 결과]
{self._format_root_causes(state.get('root_causes', []))}

[조치 가이드]
{state.get('action_guide', '조치 가이드 없음')[:500]}...

[PM 추천사항]
- Health Score: {state.get('health_score', 0):.1%}
- 고장 위험도: {state.get('failure_risk', 0):.1%}
{self._format_pm_recommendations(state.get('pm_recommendations', []))}

위 정보를 바탕으로 8D Report (D1~D7)를 작성하세요."""
            
            # 생성
            if self.lora_engine is None:
                raise RuntimeError("LoRA 엔진이 초기화되지 않았습니다.")
            
            report_8d = self.lora_engine.generate(instruction, input_text, max_new_tokens=1024)
            
            state['report_8d'] = report_8d
            state['messages'].append("Report: 8D Report 완료")
            
            print("✅ 8D Report 생성 완료")
            
        except Exception as e:
            print(f"❌ Report Agent 실행 실패: {e}")
            import traceback
            traceback.print_exc()
            
            # 기본 보고서 생성
            state['report_8d'] = f"""[8D Report 초안]

설비: {state.get('equipment_id', 'Unknown')}
발생시각: {state.get('timestamp', 'Unknown')}
이상유형: {state.get('anomaly_type', 'Unknown')}

[오류] 8D Report 자동 생성 실패: {str(e)}
수동으로 작성이 필요합니다.
"""
            state['messages'].append(f"Report: 오류 발생 - {str(e)}")
        
        return state
    
    def _format_root_causes(self, root_causes: List[Dict]) -> str:
        """원인 분석 포맷팅"""
        if not root_causes:
            return "원인 분석 없음"
        
        lines = []
        for cause in root_causes:
            rank = cause.get('rank', 1)
            cause_text = cause.get('cause', 'Unknown')
            prob = cause.get('probability', '미정')
            lines.append(f"{rank}순위: {cause_text} (확률: {prob})")
        return "\n".join(lines)
    
    def _format_pm_recommendations(self, pm_recommendations: List[Dict]) -> str:
        """PM 추천사항 포맷팅"""
        if not pm_recommendations:
            return "추천사항 없음"
        
        lines = []
        for rec in pm_recommendations:
            priority = rec.get('priority', 'MEDIUM')
            action = rec.get('action', '')
            lines.append(f"- [{priority}] {action}")
        return "\n".join(lines)