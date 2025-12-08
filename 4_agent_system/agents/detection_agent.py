"""
4_agent_system/agents/detection_agent.py
Detection Agent - 이상 탐지
"""

import sys
import os

# 상위 디렉토리 import를 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.anomaly_detector import AnomalyDetectionModel
import pandas as pd


class DetectionAgent:
    """Detection Agent - 공정 이상 탐지"""
    
    def __init__(self, 
                 model_type: str = "TimesNet",
                 model_path: str = None,
                 seq_len: int = 50):
        """
        Args:
            model_type: 'TimesNet', 'AnomalyTransformer', 'TranAD', 'rule_based'
            model_path: 학습된 모델 경로
            seq_len: 시퀀스 길이
        """
        
        if model_path is None:
            # 기본 경로 설정 (프로젝트 루트 기준)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(project_root, "2_model_training", "anomaly_model")
        
        self.detector = AnomalyDetectionModel(
            model_type=model_type,
            seq_len=seq_len,
            device='cuda',
            model_save_path=model_path
        )
        
        self.model_type = model_type
        
        print("✅ Detection Agent 초기화 완료")
        
        if not self.detector.is_trained and model_type != "rule_based":
            print("⚠️  DeepOD 모델이 학습되지 않았습니다.")
            print("   옵션 1: train() 메서드로 학습")
            print("   옵션 2: 규칙 기반 탐지 자동 사용")
    
    def train(self, train_data: pd.DataFrame, epochs: int = 10):
        """
        정상 데이터로 모델 학습
        
        Args:
            train_data: 정상 센서 데이터 (DataFrame)
                       columns: ['temperature', 'pressure', 'vibration', 'cycle_time']
            epochs: 학습 에포크
        """
        
        return self.detector.train(train_data, epochs=epochs)
    
    def run(self, state: dict) -> dict:
        """
        이상 탐지 실행 (Multi-Agent 시스템용)
        
        Args:
            state: AgentState 딕셔너리
        
        Returns:
            업데이트된 state
        """
        
        print(f"\n{'='*60}")
        print(f"🔍 Detection Agent 실행 중...")
        if self.model_type != "rule_based":
            print(f"   모델: {self.model_type} (DeepOD)")
        else:
            print(f"   모델: 규칙 기반")
        print(f"{'='*60}")
        
        # 센서 데이터 추출
        sensor_data = state['sensor_data']
        
        # 이상 탐지
        is_anomaly, score, anomaly_type = self.detector.detect_anomaly(sensor_data)
        
        # 상태 업데이트
        state['is_anomaly'] = is_anomaly
        state['anomaly_score'] = score
        state['anomaly_type'] = anomaly_type
        state['messages'].append(
            f"Detection: {'이상 감지' if is_anomaly else '정상'} "
            f"(Score: {score:.3f}, Type: {anomaly_type})"
        )
        
        # 결과 출력
        print(f"결과: {'🚨 이상 감지' if is_anomaly else '✅ 정상'}")
        print(f"이상 유형: {anomaly_type}")
        print(f"신뢰도: {score:.3f} (임계값: {self.detector.threshold:.3f})")
        
        return state


# 테스트 코드
if __name__ == "__main__":
    print("="*60)
    print("Detection Agent 단독 테스트")
    print("="*60)
    
    # Agent 초기화
    agent = DetectionAgent(
        model_type="rule_based",  # 빠른 테스트용
        seq_len=50
    )
    
    # 테스트 상태
    test_state = {
        'equipment_id': '사출기-2호기',
        'timestamp': '2024-12-08 10:00:00',
        'sensor_data': {
            'temperature': 235.0,  # 이상!
            'pressure': 120.0,
            'vibration': 1.0,
            'cycle_time': 50.0
        },
        'messages': []
    }
    
    # 실행
    result = agent.run(test_state)
    
    print(f"\n최종 결과:")
    print(f"  이상 여부: {result['is_anomaly']}")
    print(f"  이상 유형: {result['anomaly_type']}")
    print(f"  스코어: {result['anomaly_score']:.3f}")