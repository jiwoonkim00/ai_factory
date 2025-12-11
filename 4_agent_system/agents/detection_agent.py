"""
4_agent_system/agents/detection_agent.py
Detection Agent - 이상 탐지

- DeepOD 모델(TimesNet / AnomalyTransformer / TranAD 등) + 규칙 기반 탐지 지원
- DeepOD 모델이 학습되지 않은 경우 자동으로 규칙 기반 탐지로 fallback
"""

import sys
import os

# 상위 디렉토리 import를 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Tuple

import pandas as pd

from models.anomaly_detector import AnomalyDetectionModel

try:
    # config가 있으면 중앙 설정 사용
    from utils.config import ANOMALY_MODEL_PATH, DETECTION_CONFIG
except ImportError:
    ANOMALY_MODEL_PATH = None
    DETECTION_CONFIG = {
        "threshold": 0.7,
        "seq_len": 50,
    }


class DetectionAgent:
    """Detection Agent - 공정 이상 탐지"""

    def __init__(
        self,
        model_type: str = "TimesNet",
        model_path: str = None,
        seq_len: int = None,
    ):
        """
        Args:
            model_type:
                - 'TimesNet'
                - 'AnomalyTransformer'
                - 'TranAD'
                - 'rule_based'  (완전 룰 기반만 사용)
            model_path: 학습된 DeepOD 모델 저장 경로
            seq_len: 시퀀스 길이 (DeepOD용)
        """

        # seq_len 기본값: config → 파라미터 → 50
        if seq_len is None:
            seq_len = DETECTION_CONFIG.get("seq_len", 50)

        # 모델 경로 기본값: config → 프로젝트 루트 기준
        if model_path is None:
            if ANOMALY_MODEL_PATH is not None:
                model_path = str(ANOMALY_MODEL_PATH)
            else:
                project_root = os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
                model_path = os.path.join(project_root, "2_model_training", "anomaly_model")

        self.model_type = model_type

        # DeepOD 모델 생성 (rule_based라도 객체는 만들어두되, 사용 여부는 따로 결정)
        self.detector = AnomalyDetectionModel(
            model_type=model_type,
            seq_len=seq_len,
            device="cuda",
            model_save_path=model_path,
        )

        # 학습된 모델을 쓸지 / 규칙 기반으로 갈지 결정
        self.use_learned_model = (
            self.detector.is_trained and self.model_type != "rule_based"
        )

        print("✅ Detection Agent 초기화 완료")
        if not self.detector.is_trained and self.model_type != "rule_based":
            print("⚠️  DeepOD 모델이 학습되지 않았습니다.")
            print("   → 현재 세션에서는 규칙 기반 탐지를 사용합니다.")
        elif self.model_type == "rule_based":
            print("ℹ️  설정에 의해 규칙 기반 탐지만 사용합니다.")
        else:
            print(f"🤖 DeepOD 모델 사용: {self.model_type}")

    # ------------------------------------------------------------------
    # 학습용 메서드 (DeepOD)
    # ------------------------------------------------------------------
    def train(self, train_data: pd.DataFrame, epochs: int = 10):
        """
        정상 데이터로 DeepOD 모델 학습

        Args:
            train_data: 정상 센서 데이터 (DataFrame)
                        예: ['temperature', 'pressure', 'vibration', 'cycle_time']
            epochs: 학습 에포크
        """
        return self.detector.train(train_data, epochs=epochs)

    # ------------------------------------------------------------------
    # 규칙 기반 탐지 로직
    # ------------------------------------------------------------------
    def _rule_based_press(self, sensor_data: Dict[str, float]) -> Tuple[bool, float, str]:
        """
        프레스 설비용 규칙 기반 탐지
        센서 키: AI0_Vibration, AI1_Vibration (g), AI2_Current (A)
        """

        v0 = float(sensor_data.get("AI0_Vibration", 0.0))
        v1 = float(sensor_data.get("AI1_Vibration", 0.0))
        cur = float(sensor_data.get("AI2_Current", 0.0))

        max_vib = max(abs(v0), abs(v1))

        # 대략적인 기준 (dataset_3 기준)
        # - 정상 진동: |v| ≲ 0.1 g
        # - 이상 진동: |v| ≳ 0.4 g
        # - 정상 전류: 30 ~ 45 A
        # - 이상 전류: <25 A 또는 >50 A
        vib_alarm = max_vib > 0.4
        cur_high = cur > 50.0
        cur_low = cur < 25.0

        # 스코어 계산 (0~1)
        vib_score = min(max_vib / 0.6, 1.0)  # 0.6g 이상이면 1.0 취급
        cur_score = 0.0
        if cur_high:
            cur_score = min((cur - 50.0) / 15.0, 1.0)  # 65A 이상이면 1.0
        elif cur_low:
            cur_score = min((25.0 - cur) / 10.0, 1.0)  # 15A 이하면 1.0

        # 진동 비중을 조금 더 크게
        score = max(vib_score * 0.7 + cur_score * 0.3, 0.0)
        score = min(score, 1.0)

        # anomaly_type 결정
        if vib_alarm and (cur_high or cur_low):
            anomaly_type = "고진동+전류 이상"
        elif vib_alarm:
            anomaly_type = "고진동 이상"
        elif cur_high:
            anomaly_type = "과전류 이상"
        elif cur_low:
            anomaly_type = "저전류 이상"
        else:
            anomaly_type = "정상"

        # 임계값은 config 기준
        threshold = DETECTION_CONFIG.get("threshold", 0.7)
        is_anomaly = score >= threshold and anomaly_type != "정상"

        return is_anomaly, float(score), anomaly_type

    def _rule_based_molding(self, sensor_data: Dict[str, float]) -> Tuple[bool, float, str]:
        """
        사출기/일반 공정용 간단 규칙 기반 탐지
        센서 키 예시: temperature, pressure, vibration, cycle_time
        """
        temp = float(sensor_data.get("temperature", 0.0))
        pressure = float(sensor_data.get("pressure", 0.0))
        vib = float(sensor_data.get("vibration", 0.0))
        cycle = float(sensor_data.get("cycle_time", 0.0))

        # 대략적인 기준 예시
        temp_score = 0.0
        if temp > 230:
            temp_score = min((temp - 230) / 20.0, 1.0)   # 250도면 1.0
        elif temp < 170:
            temp_score = min((170 - temp) / 30.0, 1.0)   # 140도면 1.0

        vib_score = min(vib / 2.0, 1.0)  # 진동이 2.0 이상이면 1.0
        cycle_score = 0.0
        if cycle > 60:
            cycle_score = min((cycle - 60) / 20.0, 1.0)

        pressure_score = 0.0  # 필요하면 여기에 규칙 추가 가능

        # 가장 심각한 쪽을 스코어로 사용
        score = max(temp_score, vib_score, cycle_score, pressure_score)

        if temp_score >= max(vib_score, cycle_score, pressure_score):
            anomaly_type = "온도 이상"
        elif vib_score >= max(temp_score, cycle_score, pressure_score):
            anomaly_type = "진동 이상"
        elif cycle_score >= max(temp_score, vib_score, pressure_score):
            anomaly_type = "사이클타임 지연"
        else:
            anomaly_type = "정상"

        threshold = DETECTION_CONFIG.get("threshold", 0.7)
        is_anomaly = score >= threshold and anomaly_type != "정상"

        return is_anomaly, float(score), anomaly_type

    def _rule_based_detection(self, sensor_data: Dict[str, float]) -> Tuple[bool, float, str]:
        """
        설비 타입에 따라 적절한 규칙 기반 탐지 함수 선택
        """
        if "AI0_Vibration" in sensor_data or "AI1_Vibration" in sensor_data:
            # 프레스 설비
            return self._rule_based_press(sensor_data)
        else:
            # 기본: 사출기/일반 공정
            return self._rule_based_molding(sensor_data)

    # ------------------------------------------------------------------
    # Multi-Agent용 실행 메서드
    # ------------------------------------------------------------------
    def run(self, state: dict) -> dict:
        """
        이상 탐지 실행 (Multi-Agent 시스템용)

        Args:
            state: AgentState 딕셔너리

        Returns:
            업데이트된 state
        """

        print(f"\n{'=' * 60}")
        if self.use_learned_model:
            print(f"🔍 Detection Agent 실행 중...")
            print(f"   모델: {self.model_type} (DeepOD)")
        else:
            print(f"🔍 Detection Agent 실행 중... (규칙 기반 모드)")
        print(f"{'=' * 60}")

        sensor_data = state["sensor_data"]

        # ---------------------------
        # 1) 사용할 모드 결정
        # ---------------------------
        if self.use_learned_model:
            # DeepOD 모델 사용
            is_anomaly, score, anomaly_type = self.detector.detect_anomaly(sensor_data)
        else:
            # 규칙 기반 탐지 사용
            is_anomaly, score, anomaly_type = self._rule_based_detection(sensor_data)

        # ---------------------------
        # 2) 상태 업데이트
        # ---------------------------
        state["is_anomaly"] = is_anomaly
        state["anomaly_score"] = score
        state["anomaly_type"] = anomaly_type

        threshold = getattr(self.detector, "threshold", DETECTION_CONFIG.get("threshold", 0.7))
        state["messages"].append(
            f"Detection: {'이상 감지' if is_anomaly else '정상'} "
            f"(Score: {score:.3f}, Type: {anomaly_type})"
        )

        # ---------------------------
        # 3) 로그 출력
        # ---------------------------
        print(f"결과: {'🚨 이상 감지' if is_anomaly else '✅ 정상'}")
        print(f"이상 유형: {anomaly_type}")
        print(f"신뢰도: {score:.3f} (임계값: {threshold:.3f})")

        return state


# ======================================================================
# 단독 테스트용
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Detection Agent 단독 테스트")
    print("=" * 60)

    # 예시 1: 프레스 이상 케이스
    agent = DetectionAgent(
        model_type="rule_based",  # 테스트는 규칙 기반으로 강제
        seq_len=50,
    )

    test_state = {
        "equipment_id": "PRESS-01",
        "timestamp": "2025-12-11 04:00:00",
        "sensor_data": {
            "AI0_Vibration": 1.07,   # 이상
            "AI1_Vibration": -0.56,  # 이상
            "AI2_Current": 243.3,    # 살짝 높은 값
        },
        "messages": [],
    }

    result = agent.run(test_state)

    print("\n최종 결과:")
    print(f"  이상 여부: {result['is_anomaly']}")
    print(f"  이상 유형: {result['anomaly_type']}")
    print(f"  스코어: {result['anomaly_score']:.3f}")
    print(f"  메시지 로그: {result['messages']}")
