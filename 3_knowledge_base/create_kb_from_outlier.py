"""
outlier_data.csv 기반 프레스 설비 고장 이력 KB 생성 스크립트
- 입력:  ../data/outlier_data.csv
- 출력:  ./knowledge_base/histories/press_incident_XXXX.md

RAG에서 "과거 유사 사례"로 검색할 수 있는 문서들을 자동 생성한다.
"""

import os
import math
import random
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd


# -------- 경로 설정 --------
ROOT_DIR = Path(__file__).resolve().parent          # 3_knowledge_base/
DATA_PATH = ROOT_DIR.parent / "dataset_3" / "outlier_data.csv"
OUTPUT_DIR = ROOT_DIR / "knowledge_base" / "histories"


# -------- 고장 유형 / 원인 / 조치 템플릿 --------
FAULT_TEMPLATES = {
    "고진동 이상": {
        "cause_candidates": [
            "베어링 마모로 인한 진동 증가",
            "축 정렬 불량으로 인한 언밸런스",
            "풀림 볼트로 인한 구조 진동",
        ],
        "actions": [
            "베어링 상태 점검 및 필요 시 교체",
            "축 정렬(얼라인먼트) 측정 및 보정",
            "체결 볼트 토크 점검 및 재체결",
            "진동 값 정상화 여부 모니터링 (24시간)",
        ],
    },
    "전류 이상": {
        "cause_candidates": [
            "모터 과부하로 인한 과전류",
            "전원 계통 불안정",
            "인버터 설정값 이상",
        ],
        "actions": [
            "부하 조건 확인 및 과부하 요인 제거",
            "전원 라인 전압/전류 불균형 점검",
            "인버터 파라미터 재확인 및 로그 확인",
            "모터 절연 저항 측정",
        ],
    },
    "복합 이상(진동+전류)": {
        "cause_candidates": [
            "기계적 이상(베어링/축)과 전기적 이상이 동시에 발생",
            "로터 언밸런스로 인한 진동 및 전류 변동",
        ],
        "actions": [
            "진동/전류 트렌드 동시 분석",
            "베어링·축계 상태 점검",
            "전동기 및 인버터 상태 점검",
            "필요 시 샤프트 밸런싱 작업 수행",
        ],
    },
    "저전류 이상": {
        "cause_candidates": [
            "무부하 운전 또는 부하 전달 불량",
            "기계적 클러치/커플러 슬립",
        ],
        "actions": [
            "부하 연결 상태 점검 (커플러/클러치)",
            "공압/유압 계통 이상 여부 확인",
            "무부하 운전 조건인지 확인",
        ],
    },
    "기타 이상": {
        "cause_candidates": [
            "센서 노이즈 또는 일시적 이상",
            "환경 요인(온도, 전원 변동 등)",
        ],
        "actions": [
            "동일 조건 재가동 후 재현 여부 확인",
            "센서 배선 및 접지 상태 점검",
            "필요 시 센서 교체 및 교정",
        ],
    },
}


def classify_fault(row):
    """AI0/AI1 진동 + AI2 전류 값으로 고장 유형을 대략 분류"""
    vib0 = float(row["AI0_Vibration"])
    vib1 = float(row["AI1_Vibration"])
    curr = float(row["AI2_Current"])

    vib_level = max(abs(vib0), abs(vib1))

    # 임의 기준 (이상 데이터라 스케일이 클 수 있음, 필요하면 조정)
    high_vib = vib_level > 1.0
    very_high_curr = abs(curr) > 200
    low_curr = abs(curr) < 10

    if high_vib and very_high_curr:
        fault_type = "복합 이상(진동+전류)"
    elif high_vib:
        fault_type = "고진동 이상"
    elif very_high_curr:
        # 양/음 상관없이 전류 폭주로 간주
        fault_type = "전류 이상"
    elif low_curr:
        fault_type = "저전류 이상"
    else:
        fault_type = "기타 이상"

    template = FAULT_TEMPLATES[fault_type]
    cause = random.choice(template["cause_candidates"])
    actions = template["actions"]

    return fault_type, cause, actions, vib_level, curr


def make_timestamp(index: int) -> str:
    """간단히 인덱스 기반으로 가짜 발생 시각 생성"""
    base = datetime(2024, 1, 1, 8, 0, 0)
    dt = base + timedelta(minutes=15 * index)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def build_incident_markdown(idx: int, row, fault_info) -> str:
    """단일 이상 사건에 대한 Markdown 문서 생성"""
    fault_type, cause, actions, vib_level, curr = fault_info

    ts = make_timestamp(idx)
    incident_id = f"PRESS-{idx:04d}"

    vib0 = float(row["AI0_Vibration"])
    vib1 = float(row["AI1_Vibration"])
    curr = float(row["AI2_Current"])

    downtime_hours = random.randint(1, 6)

    md = f"""# 프레스 설비 이상 이력 #{idx:03d}

- 사건 ID: {incident_id}
- 설비: 프레스-1호기
- 발생 일시: {ts}
- 고장 유형: {fault_type}
- 추정 원인: {cause}
- 추정 복구 시간: 약 {downtime_hours}시간

## 센서 데이터 (이상 발생 시점)

- AI0_Vibration: {vib0:.4f}
- AI1_Vibration: {vib1:.4f}
- AI2_Current:  {curr:.4f}

## 이상 징후 요약

- 진동 레벨(최대 축 기준): {vib_level:.4f}
- 전류 값: {curr:.4f}
- 데이터 패턴 상, **{fault_type}** 패턴과 유사한 형태로 판단됨.

## 원인 분석 메모

1. 센서 데이터 상으로 {cause} 가능성이 높음.
2. outlier_data.csv 기준, 정상 구간과 비교하여 진동/전류 패턴이 크게 이탈.
3. 필요 시 실제 설비 로그, 작업 조건(하중, 속도), 최근 정비 이력과 추가 교차 검증 필요.

## 조치 내역 (예시)

"""  # end of f-string first part

    for i, act in enumerate(actions, start=1):
        md += f"{i}. {act}\n"

    md += f"""
## 예방 조치

- 동일 조건에서 최소 24시간 진동/전류 모니터링 수행
- 예방보전(PM) 점검표에 '{fault_type}' 관련 항목 추가
- 유사 패턴 재발 시 즉시 정지 및 AI Agent를 통한 원인 재분석

---
※ 이 문서는 RAG 기반 AI Agent가 참고하는 "과거 고장 이력" 예시입니다.
"""

    return md


def main(num_docs: int = 50):
    print("=== outlier_data.csv 기반 고장 이력 KB 생성 ===")
    print(f"입력 파일: {DATA_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없음: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # 필요 없는 인덱스 컬럼 제거
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    total_rows = len(df)
    print(f"- outlier 데이터 개수: {total_rows}")

    if total_rows == 0:
        raise ValueError("outlier_data.csv 에 데이터가 없습니다.")

    # 샘플링 (너무 많으면 num_docs개만)
    if total_rows <= num_docs:
        sampled = df.copy()
        print(f"- 전체 {total_rows}건을 사용합니다.")
    else:
        sampled = df.sample(n=num_docs, random_state=42).reset_index(drop=True)
        print(f"- {total_rows}건 중 {num_docs}건을 샘플링합니다.")

    # 출력 디렉토리 생성
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"- 출력 디렉토리: {OUTPUT_DIR}")

    stats = {}

    for i, row in sampled.iterrows():
        fault_info = classify_fault(row)
        fault_type = fault_info[0]
        stats[fault_type] = stats.get(fault_type, 0) + 1

        md_text = build_incident_markdown(i + 1, row, fault_info)
        out_path = OUTPUT_DIR / f"press_incident_{i+1:04d}.md"

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md_text)

    print("\n✅ KB 문서 생성 완료!")
    print(f"- 생성 문서 수: {len(sampled)}개")
    print("\n📊 고장 유형 분포:")
    for k, v in stats.items():
        print(f"  - {k}: {v}건")

    print("\n이제 setup_rag.py 에서 knowledge_base/histories/*.md 를 인덱싱하면 됩니다.")


if __name__ == "__main__":
    main()
