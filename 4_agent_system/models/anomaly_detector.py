"""
4_agent_system/models/anomaly_detector.py
간단하게 DeepOD 사용 (pip install deepod만 하면 끝!)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import pickle
import os

try:
    from deepod.models.time_series import TimesNet
    DEEPOD_AVAILABLE = True
except ImportError:
    DEEPOD_AVAILABLE = False
    print("⚠️  DeepOD 미설치: pip install deepod")


class AnomalyDetectionModel:
    """시계열 이상 탐지 모델 (DeepOD 직접 사용)"""
    
    def __init__(self, 
                 model_type: str = "TimesNet",
                 seq_len: int = 50,
                 device: str = 'cuda',
                 model_save_path: str = None):
        """
        Args:
            model_type: 'TimesNet', 'AnomalyTransformer', 'TranAD'
            seq_len: 시퀀스 길이
            device: 'cuda' or 'cpu'
            model_save_path: 모델 저장 경로 (None이면 자동 탐지)
        """
        # 경로 자동 설정
        if model_save_path is None:
            try:
                from ..utils.config import ANOMALY_MODEL_PATH
                model_save_path = str(ANOMALY_MODEL_PATH) + ".pkl"
            except ImportError:
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                model_save_path = os.path.join(project_root, "2_model_training", "anomaly_model.pkl")
        
        self.model_path = model_save_path
        self.model_type = model_type
        self.seq_len = seq_len
        self.device = device
        self.model = None
        self.threshold = 0.7
        self.is_trained = False
        
        # 학습된 모델 로드 시도
        if os.path.exists(self.model_path):
            try:
                self.load_model()
                print(f"✅ 학습된 모델 로드: {self.model_path}")
            except Exception as e:
                print(f"⚠️  모델 로드 실패: {e}")
                self._init_new_model()
        else:
            self._init_new_model()
    
    def _init_new_model(self):
        """새 모델 초기화"""
        if not DEEPOD_AVAILABLE:
            print("⚠️  DeepOD 없음. 규칙 기반 사용")
            return
        
        try:
            if self.model_type == "TimesNet":
                from deepod.models.time_series import TimesNet
                self.model = TimesNet(
                    seq_len=self.seq_len,
                    epochs=10,
                    batch_size=32,
                    device=self.device,
                    verbose=1
                )
            elif self.model_type == "AnomalyTransformer":
                from deepod.models.time_series import AnomalyTransformer
                self.model = AnomalyTransformer(
                    seq_len=self.seq_len,
                    epochs=10,
                    batch_size=32,
                    device=self.device,
                    verbose=1
                )
            elif self.model_type == "TranAD":
                from deepod.models.time_series import TranAD
                self.model = TranAD(
                    seq_len=self.seq_len,
                    epochs=10,
                    batch_size=32,
                    device=self.device,
                    verbose=1
                )
            else:
                print(f"⚠️  지원하지 않는 모델 타입: {self.model_type}")
                return
            
            print(f"✅ 새 {self.model_type} 모델 초기화 (학습 필요)")
        except Exception as e:
            print(f"⚠️  모델 초기화 실패: {e}")
            print("   규칙 기반 탐지 사용")
    
    def train(self, train_data, epochs: int = 20):
        """
        모델 학습 (정상 데이터만)
        
        Args:
            train_data: 
                - numpy array: (n_samples, seq_len, n_features) 형태
                - DataFrame: 시계열 데이터 (자동으로 시퀀스 생성)
            epochs: 학습 에포크
        """
        
        if not DEEPOD_AVAILABLE or self.model is None:
            print("⚠️  DeepOD 없음 또는 모델 미초기화. 학습 스킵")
            return
        
        print(f"\n{'='*60}")
        print(f"🎓 {self.model_type} 학습 시작")
        print(f"{'='*60}")
        
        # DataFrame인 경우 시계열 시퀀스로 변환
        if isinstance(train_data, pd.DataFrame):
            X_train = self._dataframe_to_sequences(train_data)
        elif isinstance(train_data, np.ndarray):
            X_train = train_data
        else:
            raise ValueError(f"지원하지 않는 데이터 타입: {type(train_data)}")
        
        # 데이터 형태 확인
        if len(X_train.shape) != 3:
            raise ValueError(f"시계열 데이터는 (n_samples, seq_len, n_features) 형태여야 합니다. 현재: {X_train.shape}")
        
        print(f"데이터 형태: {X_train.shape}")
        print(f"   - 샘플 수: {X_train.shape[0]}")
        print(f"   - 시퀀스 길이: {X_train.shape[1]}")
        print(f"   - 특징 수: {X_train.shape[2]}")
        
        # 학습
        self.model.epochs = epochs
        self.model.fit(X_train)
        
        # 임계값 자동 설정 (정상 데이터의 95% percentile)
        scores = self.model.decision_function(X_train)
        self.threshold = np.percentile(scores, 95)
        
        self.is_trained = True
        
        print(f"✅ 학습 완료")
        print(f"   임계값: {self.threshold:.4f}")
        print(f"   스코어 범위: [{scores.min():.4f}, {scores.max():.4f}]")
        
        # 자동 저장
        self.save_model()
    
    def _dataframe_to_sequences(self, df: pd.DataFrame) -> np.ndarray:
        """
        DataFrame을 시계열 시퀀스로 변환
        
        Args:
            df: DataFrame with sensor columns
        
        Returns:
            (n_samples, seq_len, n_features) 형태의 numpy 배열
        """
        # timestamp 제거
        if 'timestamp' in df.columns:
            df = df.drop(columns=['timestamp'])
        
        # 센서 컬럼만 선택
        sensor_cols = ['temperature', 'pressure', 'vibration', 'cycle_time']
        available_cols = [col for col in sensor_cols if col in df.columns]
        
        if not available_cols:
            raise ValueError("센서 컬럼을 찾을 수 없습니다.")
        
        data = df[available_cols].values  # (n_timesteps, n_features)
        
        # Sliding window로 시퀀스 생성
        sequences = []
        for i in range(len(data) - self.seq_len + 1):
            seq = data[i:i+self.seq_len]  # (seq_len, n_features)
            sequences.append(seq)
        
        return np.array(sequences)  # (n_samples, seq_len, n_features)
    
    def detect_anomaly(self, sensor_data: Dict[str, float]) -> Tuple[bool, float, str]:
        """
        이상 탐지
        
        Args:
            sensor_data: {'temperature': 200, 'pressure': 120, ...}
        
        Returns:
            (is_anomaly, score, anomaly_type)
        """
        
        if DEEPOD_AVAILABLE and self.is_trained:
            return self._deepod_detection(sensor_data)
        else:
            return self._rule_based_detection(sensor_data)
    
    def _deepod_detection(self, sensor_data: Dict) -> Tuple[bool, float, str]:
        """
        DeepOD 탐지
        
        주의: 단일 시점 데이터는 시계열 모델에 직접 사용할 수 없습니다.
        최근 시계열 히스토리가 필요합니다. 없으면 규칙 기반으로 폴백합니다.
        """
        # 단일 시점 데이터는 시계열 모델에 적합하지 않음
        # 실제 운영에서는 최근 seq_len 개의 시계열 데이터가 필요합니다.
        # 여기서는 간단히 규칙 기반으로 폴백
        
        print("⚠️  단일 시점 데이터는 시계열 모델에 부적합. 규칙 기반 탐지 사용")
        return self._rule_based_detection(sensor_data)
    
    def detect_anomaly_from_sequence(self, sequence_data: np.ndarray) -> Tuple[bool, float, str]:
        """
        시계열 시퀀스 데이터로 이상 탐지
        
        Args:
            sequence_data: (seq_len, n_features) 또는 (1, seq_len, n_features) 형태
        
        Returns:
            (is_anomaly, score, anomaly_type)
        """
        if not DEEPOD_AVAILABLE or not self.is_trained:
            return self._rule_based_detection({})
        
        # 형태 확인 및 변환
        if len(sequence_data.shape) == 2:
            # (seq_len, n_features) -> (1, seq_len, n_features)
            X = sequence_data.reshape(1, *sequence_data.shape)
        elif len(sequence_data.shape) == 3:
            X = sequence_data
        else:
            raise ValueError(f"잘못된 데이터 형태: {sequence_data.shape}")
        
        # 시퀀스 길이 확인
        if X.shape[1] != self.seq_len:
            raise ValueError(f"시퀀스 길이가 맞지 않습니다. 필요: {self.seq_len}, 제공: {X.shape[1]}")
        
        # 이상 스코어 계산
        scores = self.model.decision_function(X)
        score = float(scores[0])
        
        # 이상 판단
        is_anomaly = score > self.threshold
        
        # 이상 유형 (마지막 시점 데이터 사용)
        last_point = X[0, -1, :]  # 마지막 시점
        sensor_dict = {
            'temperature': last_point[0] if len(last_point) > 0 else 0,
            'pressure': last_point[1] if len(last_point) > 1 else 0,
            'vibration': last_point[2] if len(last_point) > 2 else 0,
            'cycle_time': last_point[3] if len(last_point) > 3 else 0,
        }
        anomaly_type = self._identify_type(sensor_dict, is_anomaly)
        
        return is_anomaly, score, anomaly_type
    
    def _rule_based_detection(self, sensor_data: Dict) -> Tuple[bool, float, str]:
        """규칙 기반 탐지 (Fallback)"""
        
        score = 0.0
        anomaly_type = "정상"
        
        if 'temperature' in sensor_data:
            temp = sensor_data['temperature']
            if temp > 230 or temp < 170:
                score = 0.9
                anomaly_type = "온도 이상"
        
        if 'pressure' in sensor_data:
            pressure = sensor_data['pressure']
            if pressure < 80 or pressure > 160:
                score = max(score, 0.85)
                anomaly_type = "압력 이상"
        
        if 'vibration' in sensor_data:
            vib = sensor_data['vibration']
            if vib > 2.5:
                score = max(score, 0.8)
                anomaly_type = "진동 이상"
        
        is_anomaly = score >= 0.7
        
        return is_anomaly, score, anomaly_type
    
    def _identify_type(self, sensor_data: Dict, is_anomaly: bool) -> str:
        """이상 유형 판단"""
        
        if not is_anomaly:
            return "정상"
        
        types = []
        
        if sensor_data.get('temperature', 0) > 230:
            types.append("온도 이상")
        if sensor_data.get('pressure', 0) < 80:
            types.append("압력 이상")
        if sensor_data.get('vibration', 0) > 2.5:
            types.append("진동 이상")
        
        return ", ".join(types) if types else "알 수 없는 이상"
    
    def save_model(self):
        """모델 저장"""
        
        if not DEEPOD_AVAILABLE or not self.is_trained or self.model is None:
            return
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'threshold': self.threshold,
                'model_type': self.model_type,
                'seq_len': self.seq_len
            }, f)
        
        print(f"✅ 모델 저장: {self.model_path}")
    
    def load_model(self):
        """모델 로드"""
        
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.threshold = data.get('threshold', 0.7)
        self.model_type = data.get('model_type', 'TimesNet')
        self.seq_len = data.get('seq_len', 50)
        self.is_trained = True