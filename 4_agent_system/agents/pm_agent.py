"""
4_agent_system/agents/pm_agent.py
PM Recommendation Agent - 예방보전 추천
"""
import sys
import os

# 상위 디렉토리 import를 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.state import AgentState
from typing import Dict, List, Tuple

class PMRecommendationAgent:
    """PM Agent - 예방보전 추천"""
    
    def __init__(self):
        print("✅ PM Agent 초기화 완료")
    
    def run(self, state: AgentState) -> AgentState:
        """예방보전 추천"""
        print(f"\n{'='*60}")
        print("🔧 PM Agent 실행 중...")
        print(f"{'='*60}")
        
        try:
            # Health Score 계산
            health_score, failure_risk = self._calculate_health_score(
                state.get('sensor_data', {}),
                state.get('is_anomaly', False)
            )
            
            # PM 추천
            pm_recommendations = self._generate_pm_recommendations(
                state.get('equipment_id', 'Unknown'),
                health_score,
                failure_risk,
                state.get('anomaly_type', '정상')
            )
            
            # 상태 업데이트
            state['health_score'] = health_score
            state['failure_risk'] = failure_risk
            state['pm_recommendations'] = pm_recommendations
            state['messages'].append(
                f"PM: Health Score {health_score:.1%}, 고장 위험도 {failure_risk:.1%}"
            )
            
            print(f"✅ PM 분석 완료")
            print(f"   - Health Score: {health_score:.1%}")
            print(f"   - 고장 위험도: {failure_risk:.1%}")
            print(f"   - 추천 항목: {len(pm_recommendations)}개")
            
        except Exception as e:
            print(f"❌ PM Agent 실행 실패: {e}")
            import traceback
            traceback.print_exc()
            
            # 기본값 설정
            state['health_score'] = 0.85
            state['failure_risk'] = 0.15
            state['pm_recommendations'] = []
            state['messages'].append(f"PM: 오류 발생 - {str(e)}")
        
        return state
    
    def _calculate_health_score(self, sensor_data: Dict, is_anomaly: bool) -> Tuple[float, float]:
        """설비 건강도 계산"""
        health_score = 0.85
        failure_risk = 0.15
        
        if is_anomaly:
            health_score -= 0.30
            failure_risk += 0.40
        
        return max(0.0, min(1.0, health_score)), max(0.0, min(1.0, failure_risk))
    
    def _generate_pm_recommendations(self, equipment_id: str, health_score: float,
                                     failure_risk: float, anomaly_type: str) -> List[Dict]:
        """PM 추천 생성"""
        recommendations = []
        
        # 고위험
        if failure_risk > 0.5:
            recommendations.append({
                'priority': 'HIGH',
                'action': '48시간 내 긴급 점검 필요',
                'items': ['주요 부품 교체 검토', '전문가 진단 요청'],
                'estimated_time': '4~8시간'
            })
        
        # 중위험
        elif failure_risk > 0.3:
            recommendations.append({
                'priority': 'MEDIUM',
                'action': '1주일 내 정기 점검 권장',
                'items': ['센서 교정', '소모품 교체'],
                'estimated_time': '2~4시간'
            })
        
        # 저위험
        else:
            recommendations.append({
                'priority': 'LOW',
                'action': '정기 PM 스케줄 유지',
                'items': ['육안 점검', '청소 및 급유'],
                'estimated_time': '1~2시간'
            })
        
        # 이상 유형별 추가 권장사항
        if '온도' in anomaly_type:
            recommendations.append({
                'priority': 'HIGH',
                'action': '온도 관련 부품 집중 점검',
                'items': ['히터 저항값 측정', '냉각 시스템 점검'],
                'estimated_time': '3~5시간'
            })
        elif '압력' in anomaly_type:
            recommendations.append({
                'priority': 'HIGH',
                'action': '유압 시스템 점검',
                'items': ['펌프 성능 테스트', '씰 교체'],
                'estimated_time': '4~6시간'
            })
        elif '진동' in anomaly_type:
            recommendations.append({
                'priority': 'HIGH',
                'action': '구동부 정밀 점검',
                'items': ['베어링 교체', '정렬 조정'],
                'estimated_time': '5~8시간'
            })
        
        return recommendations