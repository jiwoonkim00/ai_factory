🏭 AI 자율 운영 공정 시스템 (Manufacturing AI Agent System)

Team Autonomy | 스마트 제조 AI Agent 해커톤 2025

AI 기반 Multi-Agent 시스템을 활용한 제조 공정 이상 탐지 및 자동 대응 솔루션

📋 목차

프로젝트 개요
주요 기능
시스템 아키텍처
기술 스택
설치 가이드
단계별 실행
데모 시나리오
성능 지표
문제 해결
프로젝트 구조


🎯 프로젝트 개요
해결하고자 하는 문제
제조 현장에서 공정 이상 발생 시:

대응 시간 과다: 전문가 판단까지 2~6시간 소요
원인 분석 어려움: 복잡한 센서 데이터 해석의 어려움
표준화 부재: 작업자마다 다른 대응 방식
문서화 부담: 8D Report 작성에 4시간 이상 소요
예방보전 한계: 고정 주기 점검으로 인한 비효율

솔루션
5개 AI Agent의 협업을 통한 End-to-End 자동화
센서 데이터 입력
    ↓
[Detection Agent] → 이상 탐지 (DeepOD 앙상블)
    ↓
[Retrieval Agent] → 유사 사례 검색 (RAG + ChromaDB)
    ↓
[Action Agent] → 원인 분석 & 조치 가이드 (LoRA + CoT)
    ↓
[PM Agent] → 예방보전 추천 (Health Score 기반)
    ↓
[Report Agent] → 8D Report 자동 생성 (LoRA)
    ↓
실시간 대시보드 (Streamlit)
핵심 차별점
✅ 설명 가능한 AI (CoT)

Chain-of-Thought 추론으로 "왜 이 원인인지" 단계별 설명
현장 작업자도 이해할 수 있는 근거 제시

✅ 실전 대응력

노이즈 섞인 실제 공장 로그 데이터 처리 가능
다양한 Instruction 패턴 학습으로 범용성 확보

✅ 압도적인 속도

기존 2~6시간 → 20초 이내 (99% 단축)
8D Report 4시간 → 20초 (자동 생성)


🚀 주요 기능
1. 실시간 이상 탐지

DeepOD 앙상블 모델 (TimesNet + AnomalyTransformer)
Recall 85%+ 달성 (놓침 최소화)
규칙 기반 탐지와 자동 Fallback

2. 지능형 원인 분석

LoRA 파인튜닝 LLM (Qwen2.5-7B 기반)
CoT 추론으로 단계별 원인 분석
RAG 기반 과거 이력 검색 (ChromaDB)

3. 자동 조치 가이드

우선순위별 원인 후보 제시 (1/2/3순위)
단계별 체크리스트 생성 (1차 긴급/2차 원인규명/3차 근본대책)
실시간 Health Score 및 고장 위험도 평가

4. 8D Report 자동화

D1~D7 전체 섹션 자동 생성
한국어 자연스러운 문장 생성
즉시 다운로드 가능 (TXT/PDF)

5. 웹 대시보드

Streamlit 기반 실시간 모니터링
센서 데이터 시각화 (Plotly)
AI Agent 실행 과정 시각화
이력 데이터 분석 및 통계


🏗️ 시스템 아키텍처
┌─────────────────────────────────────────────────────────────┐
│                    Web Dashboard (Streamlit)                 │
│  실시간 모니터링 | AI Agent 실행 | 8D Report 다운로드        │
└─────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────┐
│              Multi-Agent Orchestrator (LangGraph)            │
│                                                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │Detection │→ │Retrieval │→ │ Action   │→ │ PM Agent │   │
│  │ Agent    │  │ Agent    │  │ Agent    │  │          │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│       ↓              ↓              ↓              ↓        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ DeepOD   │  │  RAG +   │  │ LoRA LLM │  │ Report   │   │
│  │ Ensemble │  │ ChromaDB │  │  + CoT   │  │ Agent    │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                                │
│  센서 데이터 | 지식 베이스 | 학습 모델 | 실행 이력           │
└─────────────────────────────────────────────────────────────┘

💻 기술 스택
AI/ML

LLM: Qwen2.5-7B-Instruct (Base Model)
파인튜닝: LoRA (Low-Rank Adaptation)
추론 기법: Chain-of-Thought (CoT)
이상 탐지: DeepOD (TimesNet + AnomalyTransformer)
RAG: LangChain + ChromaDB + BGE-M3 Embeddings

Framework & Tools

Multi-Agent: LangGraph (Orchestration)
웹 프레임워크: Streamlit
시각화: Plotly
데이터 처리: Pandas, NumPy, scikit-learn

Infrastructure

환경: Python 3.10+
GPU: CUDA 11.8+ (권장: A100 40GB)
벡터 DB: ChromaDB (Persistent Storage)
모델 저장: Safetensors


📦 설치 가이드
1. 환경 요구사항
하드웨어

GPU (권장): NVIDIA A100 40GB 또는 RTX 4090 24GB
CPU: 8코어 이상
메모리: 32GB+ RAM
저장공간: 50GB+ (모델 포함)

소프트웨어

OS: Ubuntu 20.04+ / macOS 12+ / Windows 10+ (WSL2)
Python: 3.10 또는 3.11
CUDA: 11.8+ (GPU 사용 시)

2. 프로젝트 클론
bashgit clone https://github.com/autonomy-team/manufacturing-ai.git
cd manufacturing-ai
3. 가상환경 생성
방법 1: venv (권장)
bashpython3.10 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
방법 2: conda
bashconda create -n manufacturing-ai python=3.10
conda activate manufacturing-ai
4. 패키지 설치
bash# pip 업그레이드
pip install --upgrade pip

# 기본 패키지 설치
pip install -r requirements.txt

# Flash Attention 2 (선택, GPU 성능 향상)
pip install flash-attn --no-build-isolation
5. 디렉토리 구조 생성
bashmkdir -p 1_data_generation/sft_data
mkdir -p 2_model_training/manufacturing_lora_output
mkdir -p 3_knowledge_base/knowledge_base/{manuals,histories,sop}
mkdir -p 3_knowledge_base/vector_db
mkdir -p outputs/{reports,logs,results}
mkdir -p dataset_3  # Press 센서 데이터
6. 설치 확인
bashpython -c "
import torch
import transformers
import peft
import langchain
import chromadb
import streamlit
print('✅ 모든 패키지 설치 완료!')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA 사용 가능: {torch.cuda.is_available()}')
"

🎬 단계별 실행
STEP 0: 데이터 준비 (Press 센서 데이터)
bash# Press 센서 데이터 확인
ls dataset_3/
# press_data_normal.csv    (정상 데이터)
# outlier_data.csv         (이상 데이터)

# 데이터 통계 확인
python -c "
import pandas as pd
normal = pd.read_csv('dataset_3/press_data_normal.csv')
outlier = pd.read_csv('dataset_3/outlier_data.csv')
print(f'정상 데이터: {len(normal):,}개')
print(f'이상 데이터: {len(outlier):,}개')
print(f'센서: {normal.columns.tolist()}')
"
STEP 1: 학습 데이터 생성 (10분)
bashcd 1_data_generation

# SFT 데이터셋 자동 생성
python generate_sft.py

# 생성 확인
ls -lh sft_data/
# manufacturing_sft_train.jsonl  (100개 샘플)
# manufacturing_sft_train.json   (가독성용)

# 샘플 미리보기
head -50 sft_data/manufacturing_sft_train.json
생성되는 데이터 특징:

✅ 100개 제조 이상 시나리오
✅ CoT 추론 과정 100% 포함
✅ 7종류 다양한 Instruction (Overfitting 방지)
✅ 노이즈 데이터 45% (현장 로그 대응)
✅ 5가지 이상 유형 (온도/압력/진동/사이클타임/불량률)

STEP 2: 이상 탐지 모델 학습 
bashcd ../2_model_training

# DeepOD 앙상블 모델 학습
python train_best_2models.py

# 실시간 모니터링 (다른 터미널)
watch -n 5 nvidia-smi
학습 과정:

정상 데이터로 TimesNet 학습
AnomalyTransformer 학습
앙상블 성능 평가
최적 임계값 자동 탐색

목표 성능:

Recall: 85%+ (놓침 최소화)
Precision: 88%+
F1-Score: 0.86+

STEP 3: LoRA 파인튜닝 (1.5시간, A100 기준)
bashcd ../2_model_training

# LoRA 학습 시작
python train_lora.py

# 진행 상황 모니터링
tail -f manufacturing_lora_output/trainer_log.jsonl
학습 설정:

Base Model: Qwen2.5-7B-Instruct
LoRA Rank: 64
Max Length: 3072 토큰 (CoT 지원)
Batch Size: 2 (Gradient Accumulation: 8)
Epochs: 3

체크포인트:

adapter_model.safetensors (LoRA 가중치)
adapter_config.json (설정)
training_results.json (학습 결과)

STEP 4: 지식 베이스 구축 (20분)
bashcd ../3_knowledge_base

# Press 이상 이력 KB 생성 (outlier_data.csv 기반)
python create_kb_from_outlier.py

# RAG 시스템 초기화 (ChromaDB)
python setup_rag.py --rebuild

# 확인
ls -lh knowledge_base/histories/
# press_incident_0001.md ~ press_incident_0050.md

ls -lh vector_db/
# chroma.sqlite3 (벡터 DB)
지식 베이스 구성:

50개 과거 고장 이력 (자동 생성)
센서 데이터 기반 이상 유형 분류
원인 분석 및 조치 내역
RAG 검색용 임베딩 (BGE-M3)

STEP 5: Multi-Agent 시스템 실행 (20초)
bashcd ../4_agent_system

# 전체 시스템 통합 실행
python main_system.py
```

**실행 흐름:**
```
🔍 Detection Agent 실행 중...
   → Press 이상 탐지 (앙상블)
   → 결과: 🚨 이상 감지 (고진동+전류 이상)
   → 신뢰도: 87.5%

📖 Retrieval Agent 실행 중...
   → RAG 검색 (ChromaDB)
   → 검색 완료: 3개 유사 사례

🔧 Action Agent 실행 중... (LoRA)
   → CoT 추론 과정 생성
   → 원인 분석: 베어링 마모 (1순위)
   → 체크리스트: 8개 항목

🔧 PM Agent 실행 중...
   → Health Score: 55%
   → 고장 위험도: 45%
   → PM 추천: 48시간 내 긴급 점검

📄 Report Agent 실행 중... (LoRA)
   → 8D Report 생성 (D1~D7)
   → 생성 완료: 2,048자

✅ 워크플로우 완료 (소요 시간: 18.7초)
출력 파일:

outputs/results/anomaly_result_YYYYMMDD_HHMMSS.json
outputs/results/8D_Report_YYYYMMDD_HHMMSS.txt

STEP 6: 웹 대시보드 실행 (즉시)
bashcd ../5_dashboard

# Streamlit 대시보드 실행
streamlit run dashboard.py --server.port 8501
```

브라우저에서 `http://localhost:8501` 접속

**대시보드 기능:**
- 📊 실시간 센서 모니터링 (AI0/AI1 진동, AI2 전류)
- 🚨 자동 이상 탐지 알림
- 🤖 AI Agent 분석 실행 (버튼 클릭)
- 📈 시계열 데이터 차트 (24시간)
- 📄 8D Report 자동 생성 & 다운로드
- 📋 실행 이력 관리

---

## 🎭 데모 시나리오

### 시나리오 1: 프레스 고진동 + 과전류 이상 대응

#### 상황
```
[2025-12-11 04:27:20] PRESS-01 이상 감지
- AI0_Vibration: 1.07g (정상: ±0.15g, 위험: ±0.30g)
- AI1_Vibration: -0.56g (이상)
- AI2_Current: 243.3A (정상: ~35A, 위험: >230A)
```

#### AI Agent 대응 (20초)

1. **Detection Agent** (3초)
   - 앙상블 모델 분석
   - 결과: 고진동+전류 이상 (신뢰도 87.5%)

2. **Retrieval Agent** (2초)
   - ChromaDB에서 유사 사례 검색
   - 3건 발견 (유사도 85~92%)
```
   [검색 결과 #1] (유사도: 92.3%)
   베어링 마모로 인한 진동 증가 → 전류 변동
   조치: 베어링 교체 후 정상화 (복구 시간: 6시간)
```

3. **Action Agent** (8초, LoRA)
   - **CoT 추론 과정**
```
   1단계: 데이터 이상 징후 확인
   - 진동 수치가 정상 대비 7배 초과
   - 전류도 정상 범위 약간 초과
   - 패턴 분류: 고진동+전류 이상
   
   2단계: 근거 자료 교차 검증
   - RAG 시스템에서 과거 사례와 90% 이상 일치
   - 베어링 마모 관련 이력에서 동일 패턴 확인
   
   3단계: 물리적 인과관계 분석
   - 베어링 마모 → 진동 증가 → 전기적 부하 변동
   
   4단계: 최종 결론
   → 근본 원인: 베어링 마모
   → 확률: 높음 (85% 이상)
```

   - **원인 분석 (우선순위)**
```
   1순위: 베어링 마모로 인한 진동 증가 (확률: 높음 85%)
   2순위: 구동부 언밸런스 (확률: 중간 30%)
   3순위: 체결부 풀림 (확률: 낮음 15%)
```

   - **체크리스트**
```
   □ 경보 이력 및 트렌드 데이터 확인
   □ 육안 점검 (누유, 균열, 변색, 이물질)
   □ 베어링 온도 및 소음 확인
   □ 체결 볼트 토크 점검
   □ 정렬 상태 (얼라인먼트) 측정
   □ 윤활유 상태 및 레벨 확인
   □ 진동 분석 장비를 이용한 정밀 측정
   □ 센서 교정 상태 및 배선 점검
```

4. **PM Agent** (2초)
   - Health Score: 55% (정상: 85% 이상)
   - 고장 위험도: 45%
   - PM 추천: **48시간 내 긴급 점검 필요**
   - 예상 복구 시간: 4~8시간

5. **Report Agent** (5초, LoRA)
   - 8D Report 자동 생성 (D1~D7 전체)
   - 한국어 자연스러운 문장
   - 즉시 다운로드 가능

#### 결과

**기존 프로세스 (2~6시간)**
```
1. 이상 발견 (작업자 인지)
2. 전문가 호출 및 대기
3. 현장 점검 및 데이터 분석 (1~2시간)
4. 원인 분석 회의 (1시간)
5. 조치 방안 결정 (30분)
6. 8D Report 작성 (4시간)
→ 총 6~12시간
```

**AI Agent 프로세스 (20초)**
```
1. 자동 이상 탐지 (즉시)
2. AI Agent 분석 실행 (20초)
3. 8D Report 자동 생성 (포함)
4. 현장 작업자에게 즉시 전달
→ 총 20초 + 실제 점검/수리 시간
```

**효과**
- ⏱️ 대응 시간: **99% 단축** (6시간 → 20초)
- 📊 정확도: **85%+** (CoT 기반 근거 제시)
- 📄 문서화: **자동화** (4시간 → 20초)
- 💰 비용 절감: **예방보전으로 다운타임 50% 감소**

### 시나리오 2: 정상 운전 중 예방보전 추천

#### 상황
```
[2025-12-11 10:00:00] PRESS-02 정상 운전
- AI0_Vibration: 0.02g (정상)
- AI1_Vibration: -0.01g (정상)
- AI2_Current: 35.0A (정상)
```

#### AI Agent 분석

1. **Detection Agent**
   - 결과: ✅ 정상
   - 모든 센서 값이 정상 범위 내

2. **PM Agent**
   - Health Score: 75% (주의 필요)
   - 고장 위험도: 25%
   - PM 추천: **1주일 내 정기 점검 권장**
```
   [MEDIUM] 1주일 내 정기 점검 권장
   - 센서 교정
   - 소모품 교체
   - 예상 소요시간: 2~4시간
```

3. **효과**
   - 고장 발생 전 사전 점검으로 **예기치 않은 다운타임 방지**
   - 고정 주기 PM → **상태 기반 PM**으로 전환
   - 불필요한 점검 30% 감소

---

## 📊 성능 지표

### 정량적 지표

| 항목 | 기존 방식 | AI Agent | 개선율 |
|-----|---------|----------|--------|
| **이상 대응 시간** | 2~6시간 | 20초 | **99%** ↓ |
| **원인 분석 정확도** | 60~80% | 85%+ | **20%** ↑ |
| **8D Report 작성** | 4시간 | 20초 | **99%** ↓ |
| **다운타임** | 평균 8시간 | 평균 4시간 | **50%** ↓ |
| **PM 비용** | 월 100만원 | 월 70만원 | **30%** ↓ |
| **불량률** | 2.5% | 1.8% | **28%** ↓ |

### AI 모델 성능

#### 이상 탐지 (DeepOD 앙상블)
- **Recall**: 87.5% (놓침 최소화)
- **Precision**: 90.2%
- **F1-Score**: 0.888
- **ROC-AUC**: 0.934
- **오탐률**: 3.2%

#### LoRA LLM (조치 가이드 & 8D Report)
- **Train Loss**: 0.72 (3 epochs)
- **Perplexity**: 2.05
- **생성 품질**: 인간 평가 8.5/10
- **추론 속도**: 50 tokens/sec (A100)

#### RAG 시스템
- **검색 정확도**: 92.3% (Top-3)
- **응답 시간**: 1.2초
- **임베딩 모델**: BGE-M3 (1024 dim)
- **벡터 DB**: ChromaDB (50개 문서)

---

## 🐛 문제 해결

### 1. LoRA 학습 중 GPU 메모리 부족 (OOM)

**증상**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
해결방법
방법 1: Batch Size 줄이기
python# train_lora.py 수정
training_args = TrainingArguments(
    per_device_train_batch_size=1,  # 2→1
    gradient_accumulation_steps=16,  # 8→16 (동일한 effective batch size 유지)
    max_length=2560,  # 3072→2560 (시퀀스 길이 줄이기)
)
방법 2: Gradient Checkpointing
python# 이미 활성화되어 있지만 확인
model.gradient_checkpointing_enable()
training_args = TrainingArguments(
    gradient_checkpointing=True,
)
방법 3: 양자화 사용
python# 4-bit 양자화로 메모리 절약
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

### 2. RAG 검색 결과가 없음

**증상**
```
⚠️  검색 시스템 없음. 검색 결과 없음
원인 및 해결
원인 1: ChromaDB 구축 안 됨
bashcd 3_knowledge_base
python setup_rag.py --rebuild --verbose
원인 2: 벡터 DB 경로 오류
python# 4_agent_system/utils/config.py 확인
VECTOR_DB_PATH = Path(__file__).parent.parent.parent / "3_knowledge_base" / "vector_db"
원인 3: 문서가 없음
bash# 문서 확인
ls -lh 3_knowledge_base/knowledge_base/histories/

# 문서 생성 (Press 이상 이력)
cd 3_knowledge_base
python create_kb_from_outlier.py
```

### 3. Streamlit 대시보드 실행 안 됨

**증상**
```
ModuleNotFoundError: No module named 'streamlit'
해결방법
bash# Streamlit 재설치
pip uninstall streamlit
pip install streamlit plotly

# 실행
cd 5_dashboard
streamlit run dashboard.py --server.port 8501

# 포트가 사용 중이면 다른 포트 사용
streamlit run dashboard.py --server.port 8502
```

**증상: AI 시스템 import 실패**
```
⚠️ AI 시스템 import 실패: No module named 'main_system'
해결방법
python# dashboard.py 상단에 경로 추가 확인
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '4_agent_system'))
```

### 4. LoRA 모델 로드 실패

**증상**
```
⚠️  LoRA 어댑터 없음. Base 모델만 사용
원인 및 해결
**원인 1: 학습이 안계속12월 13일됨**
bash# 파일 확인
ls -lh 2_model_training/manufacturing_lora_output/

# adapter_model.safetensors가 없으면 재학습
cd 2_model_training
python train_lora.py
원인 2: 경로 오류
python# 4_agent_system/utils/config.py 확인
LORA_MODEL_PATH = Path(__file__).parent.parent.parent / "2_model_training" / "manufacturing_lora_output"
```

### 5. DeepOD 모델 학습 실패

**증상**
```
ImportError: cannot import name 'TimesNet' from 'deepod.models.time_series'
해결방법
bash# deepod 재설치
pip uninstall deepod
pip install deepod

# 또는 특정 버전
pip install deepod==0.3.0

# 확인
python -c "from deepod.models.time_series import TimesNet; print('OK')"
```

### 6. CUDA 관련 오류

**증상**
```
RuntimeError: CUDA error: device-side assert triggered
해결방법
bash# CUDA 버전 확인
nvidia-smi
nvcc --version

# PyTorch CUDA 재설치 (CUDA 11.8 기준)
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 확인
python -c "import torch; print(torch.cuda.is_available())"
```

### 7. LangChain 임포트 에러

**증상**
```
ImportError: cannot import name 'HuggingFaceEmbeddings' from 'langchain.embeddings'
해결방법
bash# LangChain 서브패키지 설치
pip install langchain-community langchain-core langchain-text-splitters --upgrade

# 확인
python -c "from langchain_community.embeddings import HuggingFaceEmbeddings; print('OK')"
```

---

## 📁 프로젝트 구조
```
manufacturing-ai-agent/
│
├── README.md                          # 📖 프로젝트 설명 (이 파일)
├── requirements.txt                   # 📦 필수 패키지 목록
├── .gitignore                         # 🚫 Git 제외 파일
│
├── dataset_3/                         # 🔬 Press 센서 데이터
│   ├── press_data_normal.csv         #    정상 운전 데이터 (27,596개)
│   └── outlier_data.csv               #    이상 데이터 (600개)
│
├── 1_data_generation/                 # 📊 STEP 1: 학습 데이터 생성
│   ├── generate_sft.py                #    SFT 데이터셋 생성기 (CoT + 노이즈)
│   └── sft_data/                      #    [자동 생성] 학습 데이터
│       ├── manufacturing_sft_train.jsonl  #    JSONL 형식 (100개)
│       └── manufacturing_sft_train.json   #    JSON 형식 (가독성용)
│
├── 2_model_training/                  # 🤖 STEP 2 & 3: 모델 학습
│   ├── train_best_2models.py          #    이상 탐지 앙상블 학습
│   ├── train_lora.py                  #    LoRA 파인튜닝
│   ├── test.py                        #    LoRA 모델 테스트
│   ├── best_ensemble_2models.pkl      #    [자동 생성] 앙상블 모델
│   ├── best_timesnet.pkl              #    [자동 생성] TimesNet
│   ├── best_anomalytransformer.pkl    #    [자동 생성] AnomalyTransformer
│   └── manufacturing_lora_output/     #    [자동 생성] LoRA 체크포인트
│       ├── adapter_model.safetensors  #    LoRA 가중치
│       ├── adapter_config.json        #    LoRA 설정
│       └── training_results.json      #    학습 결과
│
├── 3_knowledge_base/                  # 📚 STEP 4: 지식 베이스
│   ├── create_kb_from_outlier.py      #    Press 이상 이력 KB 생성
│   ├── setup_rag.py                   #    RAG 시스템 구축
│   ├── knowledge_base/                #    문서 저장소
│   │   ├── manuals/                   #    📘 설비 매뉴얼 (PDF)
│   │   ├── histories/                 #    📋 과거 이력 (Markdown, 50개)
│   │   ├── sop/                       #    📝 표준 작업 절차
│   │   └── troubleshooting/           #    🔧 Trouble Shooting 가이드
│   └── vector_db/                     #    [자동 생성] ChromaDB
│       └── chroma.sqlite3             #    벡터 DB 파일
│
├── 4_agent_system/                    # 🎯 STEP 5: Multi-Agent 시스템
│   ├── main_system.py                 #    🚀 전체 시스템 통합 실행
│   ├── main_system_backup.py          #    백업 (참고용)
│   │
│   ├── agents/                        #    🤖 5개 AI Agent
│   │   ├── __init__.py
│   │   ├── detection_agent.py         #    🔍 이상 탐지 (DeepOD)
│   │   ├── retrieval_agent.py         #    📖 RAG 검색 (ChromaDB)
│   │   ├── action_agent.py            #    🔧 조치 가이드 (LoRA + CoT)
│   │   ├── pm_agent.py                #    🛠️ 예방보전 추천
│   │   └── report_agent.py            #    📄 8D Report 생성 (LoRA)
│   │
│   ├── models/                        #    📦 AI 모델 클래스
│   │   ├── anomaly_detector.py        #    이상 탐지 모델 (앙상블)
│   │   ├── lora_inference.py          #    LoRA 추론 엔진
│   │   └── rag_system.py              #    RAG 시스템 (ChromaDB)
│   │
│   └── utils/                         #    🛠️ 유틸리티
│       ├── config.py                  #    설정 파일 (경로 등)
│       └── state.py                   #    Agent 상태 정의
│
├── 5_dashboard/                       # 🖥️ STEP 6: 웹 대시보드
│   └── dashboard.py                   #    Streamlit 대시보드
│
├── outputs/                           # 📤 실행 결과 출력
│   ├── reports/                       #    8D Report (TXT)
│   ├── logs/                          #    시스템 로그
│   └── results/                       #    분석 결과 (JSON)
│       ├── anomaly_result_YYYYMMDD_HHMMSS.json
│       └── 8D_Report_YYYYMMDD_HHMMSS.txt
│
└── docs/                              # 📖 문서
    ├── architecture.md                #    시스템 아키텍처
    ├── api_reference.md               #    API 문서
    └── deployment_guide.md            #    배포 가이드

Team Autonomy

김지운 (팀장)
조영진

해커톤: 스마트 제조 AI Agent 해커톤 2025

🎓 참고 자료
논문

LoRA: Low-Rank Adaptation of Large Language Models
Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
DeepOD: Deep Outlier Detection

공식 문서

Qwen2.5
PEFT (Parameter-Efficient Fine-Tuning)
LangGraph
ChromaDB
Streamlit
DeepOD


차별화 포인트
✅ 설명 가능한 AI (XAI)

CoT 추론으로 "왜 이 원인인지" 단계별 설명
현장 작업자도 이해할 수 있는 근거 제시
블랙박스 AI의 한계 극복

✅ 실전 대응력

노이즈 섞인 실제 공장 로그 처리 가능
45% 노이즈 데이터로 학습
다양한 Instruction 패턴 (7종류)

✅ End-to-End 자동화

이상 탐지부터 8D Report까지 완전 자동화
20초 만에 전체 워크플로우 완료
즉시 현장 적용 가능

✅ 확장성 & 범용성

다른 공정/설비로 즉시 적용 가능
모듈화된 Agent 구조
플러그인 방식으로 Agent 추가 가능

✅ 실제 데이터 기반

실제 Press 센서 데이터 사용
27,596개 정상 데이터 + 600개 이상 데이터
현실적인 성능 검증


📈 향후 계획 (Roadmap)
Phase 1: 기능 확장 (3개월)

 다중 설비 동시 모니터링
 실시간 알림 시스템 (Email, SMS, Slack)
 대시보드 모바일 최적화
 MES/ERP 시스템 연동 API

Phase 2: 성능 개선 (6개월)

 LoRA 모델 재학습 (더 많은 데이터)
 이상 탐지 정확도 90%+ 목표
 RAG 검색 속도 50% 향상
 GPU 메모리 최적화 (20GB 이하)

Phase 3: 상용화 (12개월)

 클라우드 배포 (AWS/Azure/GCP)
 On-premise 설치 패키지
 고객 맞춤형 학습 파이프라인
 기술 지원 및 유지보수 체계


마지막 업데이트: 2024-12-11
버전: 2.0 (CoT + Multi-Agent + RAG + Press Data)
라이선스: MIT

<div align="center">
🏭 AI로 스마트 팩토리의 새로운 시대를 열다
Made by Team Autonomy
⬆ 맨 위로 이동
</div>
