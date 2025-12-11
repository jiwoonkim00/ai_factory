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
            # 센서 데이터 포맷팅
            sensor_data = state.get('sensor_data', {})
            sensor_info = f"""- AI0_Vibration: {sensor_data.get('AI0_Vibration', 0):.4f} g (정상: ±0.15g, 위험: ±0.30g)
- AI1_Vibration: {sensor_data.get('AI1_Vibration', 0):.4f} g (정상: ±0.15g, 위험: ±0.30g)
- AI2_Current: {sensor_data.get('AI2_Current', 0):.2f} A (정상: ~35A, 위험: >230A)"""
            
            # RAG 검색 결과
            rag_context = state.get('rag_context', '유사 사례 없음')[:300]
            
            # 프롬프트 구성 (한국어 명시)
            instruction = """당신은 제조 현장의 품질 관리 전문가입니다. 
반드시 한국어로 8D Report를 작성해주세요.
프레스 설비의 진동 및 전류 이상에 대한 8D Report입니다."""
            
            input_text = f"""## 프레스 설비 이상 발생 상황

### 1. 기본 정보
- **설비명**: {state.get('equipment_id', 'Unknown')}
- **발생일시**: {state.get('timestamp', 'Unknown')}
- **이상 유형**: {state.get('anomaly_type', 'Unknown')}
- **신뢰도**: {state.get('anomaly_score', 0):.1%}

### 2. 센서 측정값
{sensor_info}

### 3. AI 분석 결과

#### 원인 분석
{self._format_root_causes(state.get('root_causes', []))}

#### 유사 사례 (RAG 검색)
{rag_context}

### 4. 설비 건강도 평가
- Health Score: {state.get('health_score', 0):.1%}
- 고장 위험도: {state.get('failure_risk', 0):.1%}

### 5. PM 추천사항
{self._format_pm_recommendations(state.get('pm_recommendations', []))}

---

**지시사항**: 위 정보를 바탕으로 **반드시 한국어로** 8D Report를 작성하세요.

다음 형식을 따라주세요:

**D1. 팀 구성**
- 담당 부서와 팀원 구성

**D2. 문제 정의**
- 현상, 영향 범위, 긴급도

**D3. 임시 조치 (ICA)**
- 즉시 실행한 안전 조치

**D4. 근본 원인 분석 (RCA)**
- AI 분석 결과 기반 원인 설명
- 센서 데이터를 근거로 활용

**D5. 영구 대책 (PCA)**
- 근본 원인 제거 방안
- 예방보전 계획

**D6. 대책 실행 및 검증**
- 실행 계획 및 검증 방법

**D7. 재발 방지**
- 표준화 및 교육 계획

**중요**: 프레스 설비의 진동(Vibration)과 전류(Current) 이상에 초점을 맞춰 작성하세요."""
            
            # 생성
            if self.lora_engine is None:
                raise RuntimeError("LoRA 엔진이 초기화되지 않았습니다.")
            
            # 한국어 출력을 위해 파라미터 조정
            report_8d = self.lora_engine.generate(
                instruction, 
                input_text, 
                max_new_tokens=2048,  # 8D Report는 긴 문서이므로 증가
                temperature=0.7
            )
            
            # 8D Report 포맷팅
            if report_8d:
                # 한국어 제목 추가
                formatted_report = f"""# 8D Report - 프레스 설비 이상 분석

**설비**: {state.get('equipment_id', 'Unknown')}  
**발생일시**: {state.get('timestamp', 'Unknown')}  
**이상 유형**: {state.get('anomaly_type', 'Unknown')}  
**작성일**: {state.get('timestamp', 'Unknown')}

---

{report_8d}

---

**첨부 정보**
- 센서 데이터: AI0_Vibration={sensor_data.get('AI0_Vibration', 0):.4f}g, AI1_Vibration={sensor_data.get('AI1_Vibration', 0):.4f}g, AI2_Current={sensor_data.get('AI2_Current', 0):.2f}A
- Health Score: {state.get('health_score', 0):.1%}
- 고장 위험도: {state.get('failure_risk', 0):.1%}
"""
                state['report_8d'] = formatted_report
            else:
                state['report_8d'] = self._create_fallback_report(state, sensor_data)
            
            state['messages'].append("Report: 8D Report 완료")
            
            print("✅ 8D Report 생성 완료")
            
        except Exception as e:
            print(f"❌ Report Agent 실행 실패 (LoRA): {e}")
            print("   → Fallback 템플릿 사용")
            import traceback
            traceback.print_exc()
            
            # Fallback 보고서 생성 (한국어 템플릿)
            sensor_data = state.get('sensor_data', {})
            state['report_8d'] = self._create_fallback_report(state, sensor_data)
            state['messages'].append(f"Report: Fallback 템플릿 사용 (LoRA 실패)")
        
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
    
    def _create_fallback_report(self, state: AgentState, sensor_data: Dict) -> str:
        """폴백 8D Report 생성 (LoRA 실패 시)"""
        equipment_id = state.get('equipment_id', 'Unknown')
        timestamp = state.get('timestamp', 'Unknown')
        anomaly_type = state.get('anomaly_type', 'Unknown')
        root_causes = state.get('root_causes', [])
        
        # 첫 번째 원인
        main_cause = root_causes[0] if root_causes else "진동 및 전류 이상"
        
        return f"""# 8D Report - 프레스 설비 이상 분석

**설비**: {equipment_id}  
**발생일시**: {timestamp}  
**이상 유형**: {anomaly_type}  

---

## **D1. 팀 구성**

**담당 팀**: 생산기술팀, 품질팀, 설비보전팀  
**팀장**: [담당자명]  
**발생 부서**: 프레스 공정  
**연락처**: [연락처]  

**팀 구성**:
- 팀장: 생산기술팀장
- 품질 담당: 품질관리팀
- 설비 담당: 설비보전팀
- 현장 담당: 프레스 작업자

---

## **D2. 문제 정의**

**문제 현상**: {anomaly_type}  

**센서 측정값**:
- AI0_Vibration: {sensor_data.get('AI0_Vibration', 0):.4f} g
- AI1_Vibration: {sensor_data.get('AI1_Vibration', 0):.4f} g  
- AI2_Current: {sensor_data.get('AI2_Current', 0):.2f} A

**영향 범위**:
- 생산 중단 위험
- 제품 품질 이슈 가능성
- 설비 손상 우려

**긴급도**: 높음 (즉시 조치 필요)

---

## **D3. 임시 조치 (Interim Containment Action)**

**즉시 실행 조치**:
1. 설비 즉시 정지 및 안전 확인
2. 생산 중인 제품 격리 및 검사 대기
3. 현장 안전 점검 실시
4. 비상 연락망 가동
5. 대체 설비 또는 라인 전환 검토

**조치 시각**: {timestamp}  
**조치자**: [현장 책임자]  
**효과**: 추가 피해 방지, 안전 확보

---

## **D4. 근본 원인 분석 (Root Cause Analysis)**

**AI 분석 결과**:
{self._format_root_causes(root_causes)}

**센서 데이터 분석**:
- 진동 수준이 정상 범위(±0.15g)를 {'초과' if abs(sensor_data.get('AI0_Vibration', 0)) > 0.15 else '유지'}
- 전류 값이 {'이상' if sensor_data.get('AI2_Current', 0) > 230 else '정상'} 범위

**추정 원인**:
- 주요 원인: {main_cause}
- 기여 요인: 베어링 마모, 언밸런스, 체결부 이완 가능성
- 물리적 영향: 진동 증가 → 전기적 부하 변동 → 전류 이상

**검증 방법**:
1. 진동 분석 장비를 이용한 정밀 측정
2. 육안 점검 (균열, 마모, 변형)
3. 베어링 및 구동부 점검
4. 전기 회로 저항값 측정

---

## **D5. 영구 대책 (Permanent Corrective Action)**

**근본 원인 제거 방안**:
1. 마모 부품 교체 (베어링, 부싱 등)
2. 체결부 재조립 및 토크 재설정
3. 밸런싱 작업 실시
4. 전기 회로 점검 및 정비

**예방보전 계획**:
- 정기 점검 주기: 주 1회 → 일 1회로 강화
- 진동 모니터링 시스템 설치
- 예방 정비(PM) 항목에 진동 측정 추가
- 예비 부품 확보 (베어링, 센서 등)

**완료 목표일**: {timestamp}로부터 48시간 이내

---

## **D6. 대책 실행 및 검증**

**실행 계획**:
1. 부품 조달: 24시간 이내
2. 정비 작업: 48시간 이내 완료
3. 시운전: 정비 완료 후 즉시
4. 성능 검증: 72시간 연속 모니터링

**검증 기준**:
- 진동 수준: ±0.15g 이내 유지
- 전류 값: 정상 범위(20~50A) 유지
- 제품 품질: 규격 내 합격
- 안정성: 72시간 무중단 운전

**책임자**: 설비보전팀장  
**모니터링**: 실시간 센서 데이터 확인

---

## **D7. 재발 방지**

**표준화**:
- 작업 표준서(SOP) 개정
- 진동 및 전류 허용 기준 명시
- 점검 체크리스트 업데이트

**교육 및 훈련**:
- 전 작업자 대상 교육 실시
- 이상 징후 조기 발견 방법 공유
- 비상 대응 절차 숙지

**시스템 개선**:
- AI 이상 탐지 시스템 활용 강화
- 실시간 모니터링 대시보드 운영
- 예측 정비(Predictive Maintenance) 도입

**수평 전개**:
- 동일 기종 설비 일제 점검
- 타 라인 예방 점검 실시
- 전사 설비 관리 기준 개선

---

**작성자**: AI 자율 운영 시스템  
**승인자**: [부서장]  
**작성일**: {timestamp}  
**Health Score**: {state.get('health_score', 0):.1%}  
**고장 위험도**: {state.get('failure_risk', 0):.1%}  
"""