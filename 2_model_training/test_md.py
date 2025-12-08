# ============================================================================
# 학습 스크립트 (2_model_training/test.py에서 사용)
# ============================================================================

def train_deepod_model():
    """간단한 학습 예제"""
    
    import numpy as np
    import pandas as pd
    
    print("="*60)
    print("🎓 DeepOD TimesNet 학습")
    print("="*60)
    
    # 1. 정상 데이터 생성
    n_samples = 2000
    train_data = pd.DataFrame({
        'temperature': np.random.normal(200, 5, n_samples),
        'pressure': np.random.normal(120, 3, n_samples),
        'vibration': np.random.normal(1.0, 0.2, n_samples),
        'cycle_time': np.random.normal(50, 2, n_samples)
    })
    
    print(f"정상 데이터: {len(train_data)} 샘플")
    
    # 2. 모델 학습
    detector = AnomalyDetectionModel()
    detector.train(train_data, epochs=5)
    
    # 3. 테스트
    print("\n" + "="*60)
    print("🔍 테스트")
    print("="*60)
    
    # 정상
    result1 = detector.detect_anomaly({
        'temperature': 200.0,
        'pressure': 120.0,
        'vibration': 1.0,
        'cycle_time': 50.0
    })
    print(f"정상 데이터: {'이상' if result1[0] else '✅ 정상'} (스코어: {result1[1]:.3f})")
    
    # 이상
    result2 = detector.detect_anomaly({
        'temperature': 245.0,  # 이상!
        'pressure': 120.0,
        'vibration': 1.0,
        'cycle_time': 50.0
    })
    print(f"이상 데이터: {'🚨 이상' if result2[0] else '정상'} (스코어: {result2[1]:.3f})")
    
    print("\n✅ 완료!")


if __name__ == "__main__":
    train_deepod_model()