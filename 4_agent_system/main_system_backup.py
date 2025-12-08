"""
AI 자율 운영 공정 시스템 - 전체 Multi-Agent 구조
스마트 제조 AI Agent 해커톤 2025

전체 흐름:
Detection → Retrieval (RAG) → Action (LoRA) → PM → Report (LoRA) → Dashboard
"""

import os
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

# LangGraph
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
import operator
from typing import Annotated

# AI/ML
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# RAG
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


# ============================================================================
# 1. 상태 정의 (State Definition)
# ============================================================================

class AgentState(TypedDict):
    """Multi-Agent 시스템 전체 상태"""
    # 입력
    equipment_id: str
    timestamp: str
    sensor_data: Dict[str, Any]
    
    # Detection Agent 출력
    is_anomaly: bool
    anomaly_type: str
    anomaly_score: float
    
    # Retrieval Agent 출력
    rag_context: str
    similar_cases: List[Dict]
    
    # Action Agent 출력 (LoRA)
    root_causes: List[Dict]
    action_guide: str
    checklist: List[str]
    
    # PM Agent 출력
    health_score: float
    failure_risk: float
    pm_recommendations: List[Dict]
    
    # Report Agent 출력 (LoRA)
    report_8d: str
    
    # 메시지 로그
    messages: Annotated[List[str], operator.add]
    
    # 메타데이터
    workflow_start_time: str
    workflow_status: str


# ============================================================================
# 2. Detection Agent (이상 탐지)
# ============================================================================

class AnomalyDetectionModel:
    """시계열 이상 탐지 모델 (LSTM/AutoEncoder)"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.threshold = 0.7
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("⚠️  이상 탐지 모델 미설치. 규칙 기반 탐지 사용")
    
    def load_model(self, model_path: str):
        """학습된 이상 탐지 모델 로드"""
        # TODO: 실제 LSTM/AutoEncoder 모델 로드
        pass
    
    def detect_anomaly(self, sensor_data: Dict[str, float]) -> tuple[bool, float, str]:
        """
        이상 탐지 수행
        
        Returns:
            (is_anomaly, anomaly_score, anomaly_type)
        """
        # 간단한 규칙 기반 탐지 (MVP용)
        anomaly_type = "정상"
        anomaly_score = 0.0
        
        # 온도 체크
        if 'temperature' in sensor_data:
            temp = sensor_data['temperature']
            if temp > 230 or temp < 170:
                anomaly_score = max(anomaly_score, 0.9)
                anomaly_type = "온도 이상"
        
        # 압력 체크
        if 'pressure' in sensor_data:
            pressure = sensor_data['pressure']
            if pressure < 80 or pressure > 160:
                anomaly_score = max(anomaly_score, 0.85)
                anomaly_type = "압력 이상"
        
        # 진동 체크
        if 'vibration' in sensor_data:
            vibration = sensor_data['vibration']
            if vibration > 2.5:
                anomaly_score = max(anomaly_score, 0.8)
                anomaly_type = "진동 이상"
        
        # 사이클타임 체크
        if 'cycle_time' in sensor_data:
            cycle_time = sensor_data['cycle_time']
            if cycle_time > 75:
                anomaly_score = max(anomaly_score, 0.75)
                anomaly_type = "사이클타임 지연"
        
        is_anomaly = anomaly_score >= self.threshold
        
        return is_anomaly, anomaly_score, anomaly_type


class DetectionAgent:
    """Detection Agent - 공정 이상 탐지"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.detector = AnomalyDetectionModel(model_path)
        print("✅ Detection Agent 초기화 완료")
    
    def run(self, state: AgentState) -> AgentState:
        """이상 탐지 실행"""
        print(f"\n{'='*60}")
        print("🔍 Detection Agent 실행 중...")
        print(f"{'='*60}")
        
        # 이상 탐지
        is_anomaly, score, anomaly_type = self.detector.detect_anomaly(
            state['sensor_data']
        )
        
        # 상태 업데이트
        state['is_anomaly'] = is_anomaly
        state['anomaly_score'] = score
        state['anomaly_type'] = anomaly_type
        state['messages'].append(
            f"Detection: {'이상 감지' if is_anomaly else '정상'} "
            f"(Score: {score:.2f}, Type: {anomaly_type})"
        )
        
        print(f"결과: {'🚨 이상 감지' if is_anomaly else '✅ 정상'}")
        print(f"이상 유형: {anomaly_type}")
        print(f"신뢰도: {score:.1%}")
        
        return state


# ============================================================================
# 3. Retrieval Agent (RAG 기반 근거 검색)
# ============================================================================

class RAGSystem:
    """RAG 시스템 - 과거 이력, 매뉴얼 검색"""
    
    def __init__(self, knowledge_base_path: str = "./knowledge_base"):
        self.knowledge_base_path = knowledge_base_path
        self.embeddings = None
        self.vectorstore = None
        self.documents = []
        
        print("📚 RAG 시스템 초기화 중...")
        self._initialize_embeddings()
        self._load_knowledge_base()
    
    def _initialize_embeddings(self):
        """임베딩 모델 로드"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
            )
            print("✅ 임베딩 모델 로드 완료 (bge-m3)")
        except Exception as e:
            print(f"⚠️  임베딩 모델 로드 실패: {e}")
            # Fallback
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
    
    def _load_knowledge_base(self):
        """지식 베이스 로드 및 벡터화"""
        os.makedirs(self.knowledge_base_path, exist_ok=True)
        
        # 샘플 문서 생성 (실제로는 파일에서 로드)
        sample_docs = self._create_sample_documents()
        
        # 문서 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
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
            print(f"✅ 지식 베이스 로드 완료 ({len(splits)}개 청크)")
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
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """유사 문서 검색"""
        if self.vectorstore is None:
            return []
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        retrieved = []
        for doc, score in results:
            retrieved.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'similarity': float(1 - score)  # 거리를 유사도로 변환
            })
        
        return retrieved


class RetrievalAgent:
    """Retrieval Agent - RAG 기반 근거 검색"""
    
    def __init__(self, knowledge_base_path: str = "./knowledge_base"):
        self.rag = RAGSystem(knowledge_base_path)
        print("✅ Retrieval Agent 초기화 완료")
    
    def run(self, state: AgentState) -> AgentState:
        """RAG 검색 실행"""
        print(f"\n{'='*60}")
        print("📖 Retrieval Agent 실행 중...")
        print(f"{'='*60}")
        
        # 이상이 아니면 스킵
        if not state['is_anomaly']:
            state['messages'].append("Retrieval: 이상 없음, 검색 스킵")
            return state
        
        # 검색 쿼리 구성
        query = f"""
        설비: {state['equipment_id']}
        이상 유형: {state['anomaly_type']}
        센서 데이터: {state['sensor_data']}
        """
        
        # RAG 검색
        similar_cases = self.rag.search(query, k=3)
        
        # 컨텍스트 구성
        rag_context = "\n\n".join([
            f"[검색 결과 #{i+1}] (유사도: {case['similarity']:.1%})\n{case['content']}"
            for i, case in enumerate(similar_cases)
        ])
        
        # 상태 업데이트
        state['similar_cases'] = similar_cases
        state['rag_context'] = rag_context
        state['messages'].append(f"Retrieval: {len(similar_cases)}개 유사 사례 검색 완료")
        
        print(f"검색 완료: {len(similar_cases)}개 문서")
        for i, case in enumerate(similar_cases):
            print(f"  [{i+1}] {case['metadata'].get('type', 'Unknown')} "
                  f"(유사도: {case['similarity']:.1%})")
        
        return state


# ============================================================================
# 4. Action Agent (LoRA 모델 기반 조치 생성)
# ============================================================================

class LoRAInferenceEngine:
    """LoRA 파인튜닝 모델 추론 엔진"""
    
    def __init__(self, 
                 base_model_path: str = "Qwen/Qwen2.5-7B-Instruct",
                 lora_adapter_path: str = "./manufacturing_lora_output"):
        
        self.base_model_path = base_model_path
        self.lora_adapter_path = lora_adapter_path
        self.model = None
        self.tokenizer = None
        
        print(f"🤖 LoRA 모델 로딩 중...")
        self._load_model()
    
    def _load_model(self):
        """Base 모델 + LoRA 어댑터 로드"""
        try:
            # 토크나이저
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=True
            )
            
            # Base 모델
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # LoRA 어댑터 (학습 완료 후)
            if os.path.exists(self.lora_adapter_path):
                self.model = PeftModel.from_pretrained(
                    self.model, 
                    self.lora_adapter_path
                )
                print(f"✅ LoRA 어댑터 로드 완료: {self.lora_adapter_path}")
            else:
                print(f"⚠️  LoRA 어댑터 없음. Base 모델만 사용: {self.base_model_path}")
            
            self.model.eval()
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            raise
    
    def generate(self, 
                 instruction: str,
                 input_text: str,
                 max_new_tokens: int = 1024,
                 temperature: float = 0.7) -> str:
        """텍스트 생성"""
        
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response


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
        if not state['is_anomaly']:
            state['messages'].append("Action: 이상 없음, 조치 스킵")
            return state
        
        # 프롬프트 구성
        instruction = "당신은 제조 현장의 전문 설비 엔지니어입니다. 공정 이상 상황에 대해 원인을 분석하고, 구체적인 조치 가이드를 작성해주세요."
        
        input_text = f"""[공정 이상 이벤트]
설비: {state['equipment_id']}
발생시각: {state['timestamp']}
이상유형: {state['anomaly_type']}

[센서 데이터]
{self._format_sensor_data(state['sensor_data'])}

[RAG 검색 결과]
{state.get('rag_context', '검색 결과 없음')}

위 정보를 바탕으로 원인 분석과 조치 가이드를 작성하세요."""
        
        # LoRA 모델로 생성
        print("LoRA 모델 추론 중...")
        action_guide = self.lora_engine.generate(instruction, input_text)
        
        # 원인 후보 파싱 (간단한 휴리스틱)
        root_causes = self._parse_root_causes(action_guide)
        checklist = self._parse_checklist(action_guide)
        
        # 상태 업데이트
        state['root_causes'] = root_causes
        state['action_guide'] = action_guide
        state['checklist'] = checklist
        state['messages'].append("Action: 조치 가이드 생성 완료 (LoRA)")
        
        print(f"✅ 조치 가이드 생성 완료 ({len(action_guide)} 자)")
        print(f"   - 원인 후보: {len(root_causes)}개")
        print(f"   - 체크리스트: {len(checklist)}개")
        
        return state
    
    def _format_sensor_data(self, sensor_data: Dict) -> str:
        """센서 데이터 포맷팅"""
        lines = []
        for key, value in sensor_data.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)
    
    def _parse_root_causes(self, text: str) -> List[Dict]:
        """원인 후보 파싱"""
        causes = []
        lines = text.split('\n')
        
        for line in lines:
            if '1순위:' in line or '**1순위:' in line:
                cause = line.split(':')[1].strip().replace('**', '')
                causes.append({'rank': 1, 'cause': cause, 'probability': '높음'})
            elif '2순위:' in line or '**2순위:' in line:
                cause = line.split(':')[1].strip().replace('**', '')
                causes.append({'rank': 2, 'cause': cause, 'probability': '중간'})
            elif '3순위:' in line or '**3순위:' in line:
                cause = line.split(':')[1].strip().replace('**', '')
                causes.append({'rank': 3, 'cause': cause, 'probability': '낮음'})
        
        return causes if causes else [{'rank': 1, 'cause': '원인 분석 중', 'probability': '미정'}]
    
    def _parse_checklist(self, text: str) -> List[str]:
        """체크리스트 파싱"""
        checklist = []
        lines = text.split('\n')
        
        for line in lines:
            if line.strip().startswith('□'):
                item = line.strip()[2:].strip()
                checklist.append(item)
        
        return checklist if checklist else ['체크리스트 생성 중']


# ============================================================================
# 5. PM Recommendation Agent (예방보전 추천)
# ============================================================================

class PMRecommendationAgent:
    """PM Agent - 예방보전 추천"""
    
    def __init__(self):
        print("✅ PM Agent 초기화 완료")
    
    def run(self, state: AgentState) -> AgentState:
        """예방보전 추천"""
        print(f"\n{'='*60}")
        print("🔧 PM Agent 실행 중...")
        print(f"{'='*60}")
        
        # Health Score 계산 (간단한 규칙 기반)
        health_score, failure_risk = self._calculate_health_score(
            state['sensor_data'],
            state.get('is_anomaly', False)
        )
        
        # PM 추천
        pm_recommendations = self._generate_pm_recommendations(
            state['equipment_id'],
            health_score,
            failure_risk,
            state.get('anomaly_type', '정상')
        )
        
        # 상태 업데이트
        state['health_score'] = health_score
        state['failure_risk'] = failure_risk
        state['pm_recommendations'] = pm_recommendations
        state['messages'].append(
            f"PM: Health Score {health_score:.1%}, "
            f"고장 위험도 {failure_risk:.1%}"
        )
        
        print(f"✅ PM 분석 완료")
        print(f"   - Health Score: {health_score:.1%}")
        print(f"   - 고장 위험도: {failure_risk:.1%}")
        print(f"   - 추천 항목: {len(pm_recommendations)}개")
        
        return state
    
    def _calculate_health_score(self, sensor_data: Dict, is_anomaly: bool) -> tuple[float, float]:
        """설비 건강도 및 고장 위험도 계산"""
        
        # 기본 점수
        health_score = 0.85
        failure_risk = 0.15
        
        # 이상 발생 시 감점
        if is_anomaly:
            health_score -= 0.30
            failure_risk += 0.40
        
        # 센서 값 기반 조정
        if 'temperature' in sensor_data:
            temp = sensor_data['temperature']
            if temp > 220 or temp < 180:
                health_score -= 0.10
                failure_risk += 0.15
        
        if 'vibration' in sensor_data:
            vib = sensor_data['vibration']
            if vib > 2.0:
                health_score -= 0.15
                failure_risk += 0.20
        
        # 범위 제한
        health_score = max(0.0, min(1.0, health_score))
        failure_risk = max(0.0, min(1.0, failure_risk))
        
        return health_score, failure_risk
    
    def _generate_pm_recommendations(self, 
                                     equipment_id: str,
                                     health_score: float,
                                     failure_risk: float,
                                     anomaly_type: str) -> List[Dict]:
        """PM 추천 항목 생성"""
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
                'items': ['히터 저항값 측정', '냉각 시스템 점검', '단열재 교체'],
                'estimated_time': '3~5시간'
            })
        
        elif '압력' in anomaly_type:
            recommendations.append({
                'priority': 'HIGH',
                'action': '유압 시스템 점검',
                'items': ['펌프 성능 테스트', '씰 교체', '배관 청소'],
                'estimated_time': '4~6시간'
            })
        
        elif '진동' in anomaly_type:
            recommendations.append({
                'priority': 'HIGH',
                'action': '구동부 정밀 점검',
                'items': ['베어링 교체', '정렬 조정', '체결 토크 확인'],
                'estimated_time': '5~8시간'
            })
        
        return recommendations


# ============================================================================
# 6. Report Agent (8D 보고서 자동 생성 - LoRA)
# ============================================================================

class ReportAgent:
    """Report Agent - 8D 보고서 자동 생성 (LoRA)"""
    
    def __init__(self, lora_engine: LoRAInferenceEngine):
        self.lora_engine = lora_engine
        print("✅ Report Agent 초기화 완료")
    
    def run(self, state: AgentState) -> AgentState:
        """8D 보고서 생성"""
        print(f"\n{'='*60}")
        print("📄 Report Agent 실행 중... (LoRA 모델)")
        print(f"{'='*60}")
        
        # 이상이 아니면 간단한 보고서
        if not state['is_anomaly']:
            state['report_8d'] = self._generate_normal_report(state)
            state['messages'].append("Report: 정상 보고서 생성 완료")
            return state
        
        # 8D 보고서 생성 프롬프트
        instruction = "당신은 제조 현장의 품질 관리 전문가입니다. 8D Report를 작성해주세요."
        
        input_text = f"""[이상 상황 요약]
설비: {state['equipment_id']}
발생시각: {state['timestamp']}
이상유형: {state['anomaly_type']}

[원인 분석 결과]
{self._format_root_causes(state.get('root_causes', []))}

[조치 가이드]
{state.get('action_guide', '조치 가이드 없음')[:500]}...

[PM 추천사항]
- Health Score: {state.get('health_score', 0):.1%}
- 고장 위험도: {state.get('failure_risk', 0):.1%}
{self._format_pm_recommendations(state.get('pm_recommendations', []))}

위 정보를 바탕으로 8D Report (D1~D7)를 작성하세요."""
        
        # LoRA 모델로 생성
        print("8D Report 생성 중...")
        report_8d = self.lora_engine.generate(
            instruction, 
            input_text,
            max_new_tokens=1024
        )
        
        # 상태 업데이트
        state['report_8d'] = report_8d
        state['messages'].append("Report: 8D Report 생성 완료 (LoRA)")
        
        print(f"✅ 8D Report 생성 완료 ({len(report_8d)} 자)")
        
        return state
    
    def _generate_normal_report(self, state: AgentState) -> str:
        """정상 운전 보고서"""
        return f"""[정상 운전 보고서]

설비: {state['equipment_id']}
점검 시각: {state['timestamp']}
상태: 정상

센서 데이터:
{self._format_sensor_data(state['sensor_data'])}

Health Score: {state.get('health_score', 1.0):.1%}
다음 점검: 정기 PM 스케줄대로 진행
"""
    
    def _format_sensor_data(self, sensor_data: Dict) -> str:
        """센서 데이터 포맷팅"""
        return "\n".join([f"- {k}: {v}" for k, v in sensor_data.items()])
    
    def _format_root_causes(self, root_causes: List[Dict]) -> str:
        """원인 분석 포맷팅"""
        if not root_causes:
            return "원인 분석 없음"
        
        lines = []
        for cause in root_causes:
            lines.append(
                f"{cause['rank']}순위: {cause['cause']} "
                f"(확률: {cause['probability']})"
            )
        return "\n".join(lines)
    
    def _format_pm_recommendations(self, pm_recommendations: List[Dict]) -> str:
        """PM 추천사항 포맷팅"""
        if not pm_recommendations:
            return "추천사항 없음"
        
        lines = []
        for rec in pm_recommendations:
            lines.append(
                f"- [{rec['priority']}] {rec['action']}"
            )
        return "\n".join(lines)


# ============================================================================
# 7. Orchestrator (워크플로우 관리)
# ============================================================================

class ManufacturingAISystem:
    """전체 Multi-Agent 시스템 오케스트레이터"""
    
    def __init__(self,
                 lora_model_path: str = "./manufacturing_lora_output",
                 knowledge_base_path: str = "./knowledge_base"):
        
        print("=" * 80)
        print("🏭 AI 자율 운영 공정 시스템 초기화")
        print("=" * 80)
        
        # LoRA 엔진 초기화
        self.lora_engine = LoRAInferenceEngine(
            lora_adapter_path=lora_model_path
        )
        
        # Agents 초기화
        self.detection_agent = DetectionAgent()
        self.retrieval_agent = RetrievalAgent(knowledge_base_path)
        self.action_agent = ActionAgent(self.lora_engine)
        self.pm_agent = PMRecommendationAgent()
        self.report_agent = ReportAgent(self.lora_engine)
        
        # LangGraph 워크플로우 구성
        self.workflow = self._build_workflow()
        
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
        """이상 이벤트 처리 (전체 워크플로우 실행)"""
        
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
            result = self.workflow.invoke(initial_state)
            
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
            raise
    
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
# 8. 메인 실행 예제
# ============================================================================

def main():
    """메인 실행 함수"""
    
    # 시스템 초기화
    system = ManufacturingAISystem(
        lora_model_path="./manufacturing_lora_output",
        knowledge_base_path="./knowledge_base"
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
    with open("result_temperature_anomaly.json", "w", encoding="utf-8") as f:
        json.dump(result1, f, ensure_ascii=False, indent=2, default=str)
    
    print("\n✅ 결과 저장: result_temperature_anomaly.json")
    
    # 8D Report 출력
    print("\n" + "=" * 80)
    print("📄 생성된 8D Report")
    print("=" * 80)
    print(result1.get('report_8d', 'N/A'))
    
    
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
    os.makedirs("./knowledge_base", exist_ok=True)
    os.makedirs("./manufacturing_lora_output", exist_ok=True)
    
    print("""
    ⚠️  주의사항:
    1. LoRA 모델 학습이 먼저 완료되어야 합니다.
       → python train_lora.py
    
    2. 필요한 패키지 설치:
       → pip install langchain langgraph faiss-cpu sentence-transformers
    
    3. GPU 메모리 충분한지 확인 (최소 20GB 권장)
    """)
    
    # 실행
    main()