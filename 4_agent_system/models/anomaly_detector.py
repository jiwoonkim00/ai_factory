# models/anomaly_detector.py
"""
AnomalyDetectionModel
- DeepOD 단일 모델(TimesNet, AnomalyTransformer, TranAD)
- DeepOD 2-모델 앙상블(ensemble)
- 규칙 기반(rule_based) 탐지
"""

import os
import pickle
from typing import Dict, Tuple

import numpy as np


class AnomalyDetectionModel:
    def __init__(
        self,
        model_type: str = "TimesNet",
        seq_len: int = 50,
        device: str = "cuda",
        model_save_path: str = None,
        threshold: float = 0.7,
    ):
        """
        Args:
            model_type: 'TimesNet', 'AnomalyTransformer', 'TranAD', 'ensemble', 'rule_based'
            seq_len: 시퀀스 길이 (현재는 단일 샘플에도 사용 가능)
            device: 'cuda' or 'cpu'
            model_save_path: 학습된 모델(.pkl) 경로 또는 디렉토리
            threshold: 이상 판정 임계값
        """

        self.model_type = model_type
        self.seq_len = seq_len
        self.device = device
        self.model_save_path = model_save_path
        self.threshold = threshold

        # DeepOD 관련
        self.models = None          # 단일 모델 or {name: model, ...}
        self.scaler = None
        self.feature_cols = None
        self.is_trained = False

        # 규칙 기반만 쓰는 경우
        if self.model_type == "rule_based":
            print("   → 규칙 기반 모드 (DeepOD 모델 로드 안 함)")
            return

        # DeepOD 체크포인트 로드
        ckpt_path = self._resolve_ckpt_path(model_save_path)
        if ckpt_path is None:
            print("⚠️  DeepOD 체크포인트를 찾을 수 없어 규칙 기반으로 fallback 합니다.")
            self.model_type = "rule_based"
            return

        try:
            with open(ckpt_path, "rb") as f:
                ckpt = pickle.load(f)
        except Exception as e:
            print(f"⚠️  모델 로드 실패: {e}")
            print("   → 규칙 기반으로 fallback 합니다.")
            self.model_type = "rule_based"
            return

        # ─────────────────────────────────────
        #  체크포인트 포맷 처리
        #  - ensemble: ckpt['models'] 에 두 모델 저장
        #  - single   : ckpt['model'] 또는 ckpt['models'][model_type]
        # ─────────────────────────────────────
        if self.model_type == "ensemble":
            if "models" in ckpt:
                self.models = ckpt["models"]              # dict: {"TimesNet": m1, "AnomalyTransformer": m2}
            elif "model" in ckpt:
                # 혹시 하나만 저장된 경우
                self.models = {"model": ckpt["model"]}
            else:
                print("⚠️  ensemble 체크포인트에 'models' 키가 없습니다. 규칙 기반으로 fallback 합니다.")
                self.model_type = "rule_based"
                return
        else:
            # 단일 모델 모드
            if "model" in ckpt:
                self.models = ckpt["model"]
            elif "models" in ckpt:
                # 여러 개 중에서 내가 원하는 타입 하나만 꺼내 쓰기
                self.models = ckpt["models"].get(self.model_type)
                if self.models is None:
                    print(f"⚠️  체크포인트에 {self.model_type} 모델이 없습니다. 규칙 기반으로 fallback 합니다.")
                    self.model_type = "rule_based"
                    return
            else:
                print("⚠️  체크포인트에 'model' 또는 'models' 키가 없습니다. 규칙 기반으로 fallback 합니다.")
                self.model_type = "rule_based"
                return

        # 나머지 메타 정보
        self.scaler = ckpt.get("scaler", None)
        self.feature_cols = ckpt.get("feature_cols", None)
        if "threshold" in ckpt:
            try:
                self.threshold = float(ckpt["threshold"])
            except Exception:
                pass

        self.is_trained = True
        print(
            f"✅ DeepOD {self.model_type} 모델 로드 완료 "
            f"(ckpt: {os.path.basename(ckpt_path)}, threshold={self.threshold:.3f})"
        )

    # ------------------------------------------------------------------
    # 내부 유틸
    # ------------------------------------------------------------------
    def _resolve_ckpt_path(self, model_save_path: str) -> str:
        """디렉토리/파일 여부에 따라 실제 .pkl 경로 반환"""
        if model_save_path is None:
            return None

        # 디렉토리면 가장 흔한 파일명 추정
        if os.path.isdir(model_save_path):
            # 예: 2_model_training/anomaly_model/best_model.pkl 형태
            candidates = [
                os.path.join(model_save_path, "best_model.pkl"),
                os.path.join(model_save_path, "model.pkl"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    return c
            return None

        # 파일이면 그대로 사용
        if os.path.exists(model_save_path):
            return model_save_path

        return None

    # ------------------------------------------------------------------
    #  규칙 기반 탐지 로직
    # ------------------------------------------------------------------
    def _rule_based_detect(self, sensor_data: Dict[str, float]) -> Tuple[bool, float, str]:
        """
        프레스용 간단 룰:
        - AI0_Vibration, AI1_Vibration: |값| > 0.15g  → 고진동
        - AI2_Current: < 220A or > 240A → 전류 이상
        """
        vib0 = float(sensor_data.get("AI0_Vibration", 0.0))
        vib1 = float(sensor_data.get("AI1_Vibration", 0.0))
        cur  = float(sensor_data.get("AI2_Current", 0.0))

        high_vib = (abs(vib0) > 0.15) or (abs(vib1) > 0.15)
        cur_abnormal = (cur < 220.0) or (cur > 240.0)

        if high_vib and cur_abnormal:
            return True, 1.0, "고진동+전류 이상"
        elif high_vib:
            return True, 0.9, "고진동 이상"
        elif cur_abnormal:
            return True, 0.8, "전류 이상"
        else:
            return False, 0.02, "정상"

    # ------------------------------------------------------------------
    #  DeepOD/ensemble 기반 이상 탐지
    # ------------------------------------------------------------------
    def _deepod_detect(self, sensor_data: Dict[str, float]) -> Tuple[bool, float, str]:
        if not self.is_trained or self.models is None or self.feature_cols is None:
            # 안전하게 룰 기반으로 fallback
            return self._rule_based_detect(sensor_data)

        # feature 순서에 맞게 벡터 구성
        x = np.array([[float(sensor_data.get(col, 0.0)) for col in self.feature_cols]], dtype=float)

        # 스케일러 적용
        if self.scaler is not None:
            x = self.scaler.transform(x)

        # 모델별 score 계산
        def _score_from_model(m, x_arr):
            # deepod는 보통 decision_function 사용
            if hasattr(m, "decision_function"):
                return float(m.decision_function(x_arr)[0])
            elif hasattr(m, "predict_confidence"):
                return float(m.predict_confidence(x_arr)[0])
            else:
                # 0/1 label만 있는 경우
                return float(m.predict(x_arr)[0])

        if self.model_type == "ensemble":
            # dict: {name: model}
            scores = []
            for name, m in self.models.items():
                try:
                    s = _score_from_model(m, x)
                    scores.append(s)
                except Exception:
                    continue

            if not scores:
                # 혹시 다 실패하면 룰로 fallback
                return self._rule_based_detect(sensor_data)

            score = float(np.mean(scores))
            anomaly_type = "프레스 이상 (앙상블)"
        else:
            # 단일 모델
            score = _score_from_model(self.models, x)
            anomaly_type = self.model_type

        is_anomaly = score >= self.threshold
        if not is_anomaly:
            anomaly_type = "정상"

        return is_anomaly, score, anomaly_type

    # ------------------------------------------------------------------
    #  공개 API
    # ------------------------------------------------------------------
    def train(self, train_df, epochs: int = 10):
        """
        (선택) DeepOD 학습용 – 지금은 rule_based + 사전 학습된 pkl을 쓰므로 필수는 아님.
        """
        raise NotImplementedError("현재 버전에서는 train() 대신 외부 스크립트에서 학습 후 pkl을 로드합니다.")

    def detect_anomaly(self, sensor_data: Dict[str, float]):
        """
        Args:
            sensor_data: {"AI0_Vibration": float, "AI1_Vibration": float, "AI2_Current": float, ...}

        Returns:
            (is_anomaly: bool, score: float, anomaly_type: str)
        """
        if self.model_type == "rule_based":
            return self._rule_based_detect(sensor_data)
        else:
            return self._deepod_detect(sensor_data)
