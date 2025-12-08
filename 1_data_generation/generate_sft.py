"""
제조 공정 이상 대응 LoRA 파인튜닝 데이터셋 생성 스크립트 (고도화 버전)
AI 자율 운영 공정 시스템 - Action Agent용

주요 개선사항:
1. Chain of Thought (CoT) 추론 과정 포함
2. Instruction 다양화 (Overfitting 방지)
3. 입력 데이터 노이즈 추가 (현장감)
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict
import os

class ManufacturingSFTGenerator:
    def __init__(self):
        # 설비 유형
        self.equipment_types = [
            "사출기", "프레스", "CNC", "용접기", "조립라인", 
            "코팅장비", "건조로", "컨베이어", "로봇암", "검사장비"
        ]
        
        # 🆕 Instruction 다양화 (Overfitting 방지)
        self.instructions = [
            "당신은 제조 현장의 전문 설비 엔지니어입니다. 공정 이상 상황에 대해 원인을 분석하고, 구체적인 조치 가이드와 8D 보고서를 작성해주세요.",
            "주어진 센서 데이터와 RAG 검색 결과를 종합하여, 설비 이상의 근본 원인을 파악하고 대응 방안을 보고서 형태로 작성하시오.",
            "Smart Factory AI Agent로서, 현재 발생한 이상 징후를 분석하고 현장 작업자가 실행해야 할 체크리스트와 조치 사항을 제안하세요.",
            "다음 로그 데이터를 기반으로 설비 이상의 원인을 추론하고, 8D Report 초안을 작성하시오.",
            "제조 공정 전문가 관점에서 아래 이상 상황을 분석하고, 우선 점검 항목과 단계별 대응 방안을 제시해주세요.",
            "설비 관리 시스템에서 탐지된 이상 패턴을 해석하고, 근거 기반의 원인 분석 및 조치 계획을 수립하시오.",
            "현장 엔지니어를 위한 상세한 Trouble Shooting 가이드를 작성하세요. 센서 데이터와 과거 이력을 참고하여 체계적으로 접근하시오."
        ]
        
        # 이상 유형별 상세 정보
        self.anomaly_patterns = {
            "온도 이상": {
                "sensors": ["실린더 온도", "금형 온도", "냉각수 온도", "유압유 온도"],
                "threshold": "설정값 대비 ±15°C",
                "causes": [
                    "히터 고장 또는 성능 저하",
                    "온도 센서 오류 또는 배선 불량",
                    "냉각 시스템 막힘 또는 순환 불량",
                    "단열재 파손",
                    "제어 프로그램 파라미터 오류"
                ],
                "symptoms": ["제품 변형", "치수 불량", "표면 거칠기 증가"],
                "physics": ["열전달 효율 저하", "온도 분포 불균일", "열팽창 계수 변화"]
            },
            "압력 이상": {
                "sensors": ["유압 압력", "공압", "사출 압력", "보압"],
                "threshold": "정상 범위 ±20%",
                "causes": [
                    "유압펌프 성능 저하",
                    "실린더 씰 마모",
                    "압력 센서 드리프트",
                    "배관 누유 또는 막힘",
                    "밸브 고착 또는 오작동"
                ],
                "symptoms": ["사이클 타임 증가", "제품 중량 편차", "충진 불량"],
                "physics": ["유체 흐름 저항 증가", "압력 강하", "체적 효율 감소"]
            },
            "진동 이상": {
                "sensors": ["베어링 진동", "모터 진동", "구동축 진동"],
                "threshold": "RMS 값 2.5mm/s 초과",
                "causes": [
                    "베어링 마모 또는 손상",
                    "구동부 언밸런스",
                    "체결부 풀림",
                    "정렬 불량 (미스얼라인먼트)",
                    "이물질 혼입"
                ],
                "symptoms": ["소음 증가", "발열", "정밀도 저하"],
                "physics": ["공진 현상", "불균형 하중", "기계적 간극 증가"]
            },
            "사이클타임 지연": {
                "sensors": ["사이클 타임", "대기 시간", "동작 완료 시간"],
                "threshold": "표준 대비 +15% 초과",
                "causes": [
                    "동작 속도 파라미터 변경",
                    "구동부 마찰 증가",
                    "센서 응답 지연",
                    "제어 로직 오류",
                    "재료 공급 지연"
                ],
                "symptoms": ["생산량 감소", "납기 지연", "가동률 저하"],
                "physics": ["점성 마찰 증가", "응답 시간 증가", "구동 토크 부족"]
            },
            "불량률 급증": {
                "sensors": ["검사 결과", "NG 카운트", "품질 지표"],
                "threshold": "평균 불량률 2배 이상",
                "causes": [
                    "공정 조건 변화 (온도/압력/속도)",
                    "금형 또는 지그 마모",
                    "원자재 품질 변동",
                    "작업자 실수 또는 미숙",
                    "검사 기준 변경"
                ],
                "symptoms": ["폐기 비용 증가", "재작업 발생", "고객 클레임"],
                "physics": ["치수 공차 이탈", "재료 특성 변화", "성형 조건 불안정"]
            }
        }
        
        # 조치 단계 템플릿
        self.action_templates = {
            "1차_긴급조치": [
                "해당 설비 즉시 정지 및 안전 조치",
                "현재 생산 중인 제품 격리 및 전수 검사",
                "설비 상태 육안 점검 (누유, 이상음, 과열 확인)",
                "경보 이력 및 센서 로그 확인",
                "유사 설비로 생산 전환 검토",
                "생산 계획팀 및 품질팀에 즉시 통보",
                "현장 안전 점검 (화재, 누전, 가스 누출 등)"
            ],
            "2차_원인규명": [
                "센서 교정 및 정확도 검증",
                "관련 부품 분해 점검",
                "과거 동일 증상 이력 검색",
                "설비 매뉴얼 및 SOP 재확인",
                "필요시 외부 전문가 자문",
                "트렌드 데이터 분석 (24시간 이상)",
                "동일 라인 타 설비와 비교 분석",
                "부품 수명 및 교체 이력 확인"
            ],
            "3차_근본대책": [
                "마모 부품 교체 및 예비품 확보",
                "예방보전 주기 재설정",
                "작업 표준서 개정",
                "센서 모니터링 임계값 조정",
                "교육 실시 및 재발 방지 대책 수립",
                "유사 설비에 수평 전개",
                "IoT 센서 추가 설치 검토",
                "정비 이력 DB 업데이트 및 공유"
            ]
        }
        
        # 🆕 CoT에서 사용할 추론 패턴
        self.reasoning_patterns = [
            "데이터 트렌드를 분석한 결과",
            "RAG 검색 결과와 교차 검증했을 때",
            "과거 유사 사례와 비교하면",
            "물리적 인과관계를 고려할 때",
            "센서 상관관계를 분석한 결과",
            "설비 가동 이력을 종합하면",
            "통계적 이상 탐지 결과"
        ]
    
    def generate_timestamp(self):
        """무작위 타임스탬프 생성"""
        base = datetime(2024, 1, 1)
        random_days = random.randint(0, 365)
        random_hours = random.randint(0, 23)
        random_minutes = random.randint(0, 59)
        dt = base + timedelta(days=random_days, hours=random_hours, minutes=random_minutes)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    
    def generate_sensor_data(self, anomaly_type: str) -> Dict:
        """센서 데이터 생성"""
        pattern = self.anomaly_patterns[anomaly_type]
        sensor = random.choice(pattern["sensors"])
        
        if "온도" in anomaly_type:
            normal = random.randint(180, 220)
            abnormal = normal + random.randint(20, 40) * random.choice([1, -1])
            unit = "°C"
        elif "압력" in anomaly_type:
            normal = random.randint(100, 150)
            abnormal = int(normal * random.uniform(0.6, 1.4))
            unit = "bar"
        elif "진동" in anomaly_type:
            normal = round(random.uniform(0.5, 1.5), 2)
            abnormal = round(random.uniform(3.0, 5.0), 2)
            unit = "mm/s"
        elif "사이클타임" in anomaly_type:
            normal = random.randint(45, 65)
            abnormal = int(normal * random.uniform(1.2, 1.5))
            unit = "초"
        else:
            normal = round(random.uniform(1.0, 3.0), 1)
            abnormal = round(random.uniform(5.0, 10.0), 1)
            unit = "%"
        
        return {
            "sensor_name": sensor,
            "normal_value": normal,
            "abnormal_value": abnormal,
            "unit": unit,
            "threshold": pattern["threshold"]
        }
    
    def generate_rag_context(self, anomaly_type: str, cause: str) -> str:
        """RAG 검색 결과 시뮬레이션"""
        contexts = [
            f"[과거 이력 #2023-08-15] 동일 증상으로 {cause} 확인됨. 해당 부품 교체 후 정상화.",
            f"[설비 매뉴얼 3.2절] {anomaly_type} 발생 시 {cause} 가능성이 가장 높음. 즉시 점검 권장.",
            f"[Trouble Shooting Guide] {cause}는 {anomaly_type}의 주요 원인 중 하나. 정기 점검 필수.",
            f"[정비 이력 DB] 최근 6개월간 유사 케이스 3건 발생. 모두 {cause}로 판명.",
            f"[SOP 문서] {anomaly_type} 대응 절차: 1단계로 {cause} 점검 필수.",
            f"[전문가 의견] 과거 경험상 이 패턴은 90% 이상 {cause}와 연관됨."
        ]
        return "\n".join(random.sample(contexts, k=random.randint(2, 3)))
    
    # 🆕 노이즈 추가 함수
    def add_noise(self, text: str) -> str:
        """입력 데이터에 현장감 있는 노이즈 추가"""
        noise_options = []
        
        # 30% 확률로 시스템 헤더 추가
        if random.random() < 0.3:
            noise_options.append(f"[SYSTEM_LOG_DUMP_V{random.randint(1,9)}.{random.randint(0,9)}]\n")
        
        # 20% 확률로 타임스탬프 헤더 추가
        if random.random() < 0.2:
            noise_options.append(f">>> Log Extract Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} <<<\n")
        
        # 15% 확률로 설비 코드 추가
        if random.random() < 0.15:
            noise_options.append(f"Equipment_ID: EQ{random.randint(1000, 9999)} | Status: ALERT\n")
        
        # 노이즈 조합
        prefix = "".join(noise_options)
        return prefix + text if prefix else text
    
    # 🆕 CoT 추론 과정 생성
    def generate_cot_reasoning(self, sensor_data: Dict, anomaly_type: str, 
                               cause: str, pattern: Dict) -> str:
        """Chain of Thought 추론 과정 생성"""
        
        deviation = abs(sensor_data['abnormal_value'] - sensor_data['normal_value'])
        deviation_pct = round((deviation / sensor_data['normal_value']) * 100, 1)
        
        reasoning_step = random.choice(self.reasoning_patterns)
        physics_insight = random.choice(pattern['physics'])
        
        cot = f"""## 🧠 상황 분석 및 추론 과정

**1단계: 데이터 이상 징후 확인**
- {sensor_data['sensor_name']} 수치가 정상값 {sensor_data['normal_value']}{sensor_data['unit']}에서 {sensor_data['abnormal_value']}{sensor_data['unit']}로 변화
- 편차: {deviation}{sensor_data['unit']} ({deviation_pct}% 변동)
- 설정된 임계값({sensor_data['threshold']})을 명확히 초과
- 패턴 분류: 전형적인 **{anomaly_type}** 징후

**2단계: 근거 자료 교차 검증**
- {reasoning_step}, RAG 시스템에서 검색된 과거 사례와 **90% 이상 일치**
- 특히 '{cause}' 관련 이력에서 동일한 센서 패턴 확인
- 설비 매뉴얼 및 SOP 문서에서도 이 상황을 명시적으로 언급

**3단계: 물리적 인과관계 분석**
- 현 상황에서 예상되는 물리적 현상: {physics_insight}
- 이는 '{cause}'의 전형적인 증상과 일치
- 단순 센서 오류(교정 문제)보다는 **실제 설비 이상**으로 판단

**4단계: 최종 결론**
→ **근본 원인: {cause}**  
→ 확률: **높음 (85% 이상)**  
→ 근거: 센서 데이터, RAG 문맥, 물리적 분석 모두 일치"""

        return cot
    
    def generate_checklist(self, anomaly_type: str) -> List[str]:
        """점검 체크리스트 생성"""
        base_checklist = [
            "□ 경보 이력 및 트렌드 데이터 확인",
            "□ 육안 점검 (누유, 균열, 변색, 이물질)",
            "□ 센서 교정 상태 및 배선 점검"
        ]
        
        if "온도" in anomaly_type:
            base_checklist.extend([
                "□ 히터 및 열전대 저항값 측정",
                "□ 냉각 시스템 유량 및 온도 확인",
                "□ 단열재 상태 점검",
                "□ 온도 제어기 파라미터 확인"
            ])
        elif "압력" in anomaly_type:
            base_checklist.extend([
                "□ 유압펌프 압력 게이지 확인",
                "□ 실린더 씰 누유 여부 점검",
                "□ 배관 연결부 및 밸브 상태 확인",
                "□ 압력 센서 제로점 보정"
            ])
        elif "진동" in anomaly_type:
            base_checklist.extend([
                "□ 베어링 온도 및 소음 확인",
                "□ 체결 볼트 토크 점검",
                "□ 정렬 상태 (얼라인먼트) 측정",
                "□ 윤활유 상태 및 레벨 확인"
            ])
        elif "사이클타임" in anomaly_type:
            base_checklist.extend([
                "□ 동작 속도 설정값 확인",
                "□ 구동부 저항 측정",
                "□ 센서 응답 시간 테스트",
                "□ 제어 로직 및 인터록 점검"
            ])
        else:
            base_checklist.extend([
                "□ 공정 조건 (온도/압력/속도) 확인",
                "□ 금형/지그 마모 상태 점검",
                "□ 원자재 품질 검증",
                "□ 작업자 작업 방법 재확인"
            ])
        
        return base_checklist
    
    def generate_8d_section(self, equipment: str, anomaly_type: str, cause: str, timestamp: str) -> str:
        """8D 리포트 일부 생성"""
        return f"""## 📋 8D Report 초안

**D1. 팀 구성**
- 대상 설비: {equipment}
- 담당 부서: 생산기술팀, 품질팀, 설비보전팀
- 발생 일시: {timestamp}
- 보고자: AI Agent (검토 필요)

**D2. 문제 정의**
- 현상: {anomaly_type} 발생으로 정상 가동 불가
- 영향 범위: 생산 중단, 품질 이슈 발생 가능
- 긴급도: 높음 (즉시 조치 필요)

**D3. 임시 조치 (ICA)**
- 설비 즉시 정지 및 안전 조치 완료
- 생산 중 제품 격리 및 검사 대기
- 대체 설비로 생산 전환 (가능 시)

**D4. 근본 원인 분석 (RCA)**
- 추정 원인: **{cause}**
- 분석 근거: 센서 데이터 분석, RAG 과거 이력 검토, 물리적 인과관계 확인
- Why-Why 분석: (현장 검증 후 작성 필요)

**D5. 영구 대책 (PCA)**
- {cause} 해결을 위한 부품 교체 또는 조정
- 예방보전(PM) 주기 재설정
- 모니터링 시스템 강화

**D6. 대책 실행 및 검증**
- 조치 완료 후 48시간 연속 모니터링
- 동일 증상 재발 시 추가 분석 실시
- 성능 테스트 및 품질 검증

**D7. 재발 방지**
- 정기 점검 항목에 추가
- 작업 표준서(SOP) 개정
- 전 직원 교육 실시
- 유사 설비 수평 전개

**D8. 팀 노력 인정**
- (완료 후 작성)"""
    
    def generate_single_example(self) -> Dict:
        """단일 학습 예제 생성"""
        equipment = random.choice(self.equipment_types)
        equipment_id = f"{equipment}-{random.randint(1, 5)}호기"
        anomaly_type = random.choice(list(self.anomaly_patterns.keys()))
        pattern = self.anomaly_patterns[anomaly_type]
        cause = random.choice(pattern["causes"])
        timestamp = self.generate_timestamp()
        sensor_data = self.generate_sensor_data(anomaly_type)
        
        # Input 구성 (🆕 노이즈 추가)
        raw_input = f"""[공정 이상 이벤트]
설비: {equipment_id}
발생시각: {timestamp}
이상유형: {anomaly_type}

[센서 데이터]
- {sensor_data['sensor_name']}: {sensor_data['abnormal_value']}{sensor_data['unit']} (정상: {sensor_data['normal_value']}{sensor_data['unit']})
- 임계값: {sensor_data['threshold']}
- 증상: {', '.join(random.sample(pattern['symptoms'], k=2))}

[RAG 검색 결과]
{self.generate_rag_context(anomaly_type, cause)}

위 정보를 바탕으로 원인 분석, 조치 가이드, 8D 리포트를 작성하세요."""

        input_text = self.add_noise(raw_input)

        # 🆕 CoT 추론 과정 생성
        cot_reasoning = self.generate_cot_reasoning(sensor_data, anomaly_type, cause, pattern)
        
        # Output 구성 (CoT 포함)
        checklist = self.generate_checklist(anomaly_type)
        causes_ranked = random.sample(pattern["causes"], k=min(3, len(pattern["causes"])))
        
        output_text = f"""{cot_reasoning}

---

## ✅ 원인 분석 결과 (우선순위)

**1순위: {causes_ranked[0]}**
- 발생 확률: **높음 (80~90%)**
- 선정 근거: 상기 추론 과정에서 도출된 주 원인
- 즉시 점검 필요

**2순위: {causes_ranked[1] if len(causes_ranked) > 1 else '복합 요인'}**
- 발생 확률: 중간 (30~50%)
- 1순위 원인 배제 시 검토 필요

**3순위: {causes_ranked[2] if len(causes_ranked) > 2 else '기타 외부 요인'}**
- 발생 확률: 낮음 (10~20%)
- 종합 진단 필요

---

## 📝 우선 점검 체크리스트

{chr(10).join(checklist)}

---

## 🔧 단계별 조치 가이드

### [1차 긴급 조치] ⚠️
{chr(10).join(f"{i+1}. {action}" for i, action in enumerate(random.sample(self.action_templates['1차_긴급조치'], k=4)))}

### [2차 원인 규명] 🔍
{chr(10).join(f"{i+1}. {action}" for i, action in enumerate(random.sample(self.action_templates['2차_원인규명'], k=4)))}

### [3차 근본 대책] 🛠️
{chr(10).join(f"{i+1}. {action}" for i, action in enumerate(random.sample(self.action_templates['3차_근본대책'], k=4)))}

---

{self.generate_8d_section(equipment_id, anomaly_type, causes_ranked[0], timestamp)}

---

**📊 예상 복구 일정**
- 조치 예상 소요시간: {random.randint(2, 8)}시간
- 예상 복구시점: {timestamp} 이후 {random.randint(4, 12)}시간
- 후속 모니터링 기간: 48시간

**💡 추가 권장사항**
- 동일 라인 내 유사 설비 선제 점검 권장
- 예방보전(PM) 스케줄 재검토 필요
"""
        
        # 🆕 Instruction 랜덤 선택
        selected_instruction = random.choice(self.instructions)
        
        return {
            "instruction": selected_instruction,
            "input": input_text,
            "output": output_text,
            "metadata": {
                "equipment": equipment_id,
                "anomaly_type": anomaly_type,
                "timestamp": timestamp,
                "primary_cause": causes_ranked[0],
                "has_cot": True,
                "has_noise": "SYSTEM_LOG" in input_text or "Equipment_ID" in input_text
            }
        }
    
    def generate_dataset(self, num_examples: int = 100) -> List[Dict]:
        """전체 데이터셋 생성"""
        dataset = []
        for i in range(num_examples):
            example = self.generate_single_example()
            dataset.append(example)
            if (i + 1) % 10 == 0:
                print(f"생성 완료: {i + 1}/{num_examples}")
        return dataset
    
    def save_dataset(self, dataset: List[Dict], output_dir: str = "./sft_data"):
        """데이터셋 저장"""
        os.makedirs(output_dir, exist_ok=True)
        
        # JSON Lines 형식으로 저장 (Hugging Face 학습용)
        jsonl_path = os.path.join(output_dir, "manufacturing_sft_train.jsonl")
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for example in dataset:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        # JSON 형식으로도 저장 (가독성 확인용)
        json_path = os.path.join(output_dir, "manufacturing_sft_train.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 데이터셋 저장 완료:")
        print(f"   - JSONL: {jsonl_path} ({len(dataset)}개 샘플)")
        print(f"   - JSON: {json_path}")
        
        # 🆕 개선사항 통계
        cot_count = sum(1 for ex in dataset if ex['metadata'].get('has_cot', False))
        noise_count = sum(1 for ex in dataset if ex['metadata'].get('has_noise', False))
        unique_instructions = len(set(ex['instruction'] for ex in dataset))
        
        print(f"\n📊 데이터셋 품질 지표:")
        print(f"   - CoT 추론 포함: {cot_count}개 (100%)")
        print(f"   - 노이즈 추가: {noise_count}개 ({noise_count/len(dataset)*100:.1f}%)")
        print(f"   - Instruction 종류: {unique_instructions}개")
        
        # 이상 유형별 통계
        anomaly_counts = {}
        for example in dataset:
            anomaly = example['metadata']['anomaly_type']
            anomaly_counts[anomaly] = anomaly_counts.get(anomaly, 0) + 1
        
        print(f"\n📊 이상 유형별 분포:")
        for anomaly, count in sorted(anomaly_counts.items(), key=lambda x: -x[1]):
            print(f"\n📊 이상 유형별 분포:")
        for anomaly, count in sorted(anomaly_counts.items(), key=lambda x: -x[1]):
            print(f"   - {anomaly}: {count}개 ({count/len(dataset)*100:.1f}%)")


def main():
    print("=" * 60)
    print("제조 공정 AI Agent LoRA 파인튜닝 데이터셋 생성기 (고도화 버전)")
    print("=" * 60)

    generator = ManufacturingSFTGenerator()

    # 데이터셋 생성
    print("\n데이터셋 생성 중...")
    dataset = generator.generate_dataset(num_examples=100)

    # 저장
    generator.save_dataset(dataset, output_dir="./sft_data")

    # 샘플 출력
    print("\n" + "=" * 60)
    print("📄 생성된 샘플 예시 (첫 번째 항목)")
    print("=" * 60)
    sample = dataset[0]
    print(f"\n[Instruction]\n{sample['instruction']}")
    print(f"\n[Input]\n{sample['input'][:500]}...")
    print(f"\n[Output]\n{sample['output'][:800]}...")

    print("\n" + "=" * 60)
    print("✨ 생성 완료! ./sft_data 폴더에서 JSONL/JSON 파일을 확인하세요.")
    print("   → 이 파일을 LoRA/QLoRA 학습 스크립트에서 불러오면 됩니다.")
    print("=" * 60)


if __name__ == "__main__":
    main()