"""
DeepOD TimesNet 시계열 이상 탐지 모델 학습 스크립트

사용법:
    python train_anomaly_detector.py --data_path data/sensor_data.csv --epochs 20

DeepOD TimesNet은 시계열 데이터를 (n_samples, seq_len, n_features) 형태로 받습니다.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import torch

# 프로젝트 루트 경로 추가
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "4_agent_system"))

from models.anomaly_detector import AnomalyDetectionModel


def get_device():
    """
    GPU 디바이스 자동 감지 및 설정 (A100 최적화)
    """
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✅ GPU 감지: {gpu_name}")
        print(f"   GPU 메모리: {gpu_memory:.1f} GB")
        
        # A100 감지
        if 'A100' in gpu_name:
            print(f"   🚀 A100 GPU 감지 - 최적화 모드 활성화")
            # CUDA 최적화 설정
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
    else:
        device = 'cpu'
        print(f"⚠️  GPU 없음 - CPU 모드")
    
    return device


def get_optimal_batch_size(device: str, seq_len: int, n_features: int, default_batch_size: int = 32):
    """
    GPU 메모리에 맞는 최적 배치 크기 계산
    """
    if device == 'cpu':
        return min(default_batch_size, 16)
    
    try:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # A100 40GB 기준 최적 배치 크기
        if gpu_memory_gb >= 40:  # A100 40GB
            # 메모리 여유가 있으므로 큰 배치 크기 사용
            if seq_len <= 50:
                optimal_batch = 128
            elif seq_len <= 100:
                optimal_batch = 64
            else:
                optimal_batch = 32
        elif gpu_memory_gb >= 24:  # A100 24GB or similar
            optimal_batch = 64
        elif gpu_memory_gb >= 16:  # V100 or similar
            optimal_batch = 32
        else:
            optimal_batch = 16
        
        print(f"   최적 배치 크기: {optimal_batch} (GPU 메모리: {gpu_memory_gb:.1f}GB 기준)")
        return optimal_batch
    except:
        return default_batch_size


def create_synthetic_time_series_data(
    n_samples: int = 5000,
    seq_len: int = 50,
    noise_level: float = 0.1
) -> np.ndarray:
    """
    시뮬레이션 시계열 데이터 생성
    
    Args:
        n_samples: 전체 샘플 수
        seq_len: 시퀀스 길이
        noise_level: 노이즈 레벨
    
    Returns:
        (n_samples, seq_len, n_features) 형태의 numpy 배열
    """
    print(f"📊 시뮬레이션 시계열 데이터 생성 중...")
    print(f"   - 샘플 수: {n_samples}")
    print(f"   - 시퀀스 길이: {seq_len}")
    print(f"   - 노이즈 레벨: {noise_level}")
    
    n_features = 4  # temperature, pressure, vibration, cycle_time
    
    # 전체 시계열 데이터 생성 (정상 패턴)
    total_length = n_samples + seq_len - 1
    
    # 정상 패턴: 주기적 변동 + 트렌드
    time_points = np.arange(total_length)
    
    # 각 센서별 정상 패턴
    data = np.zeros((total_length, n_features))
    
    # 1. 온도: 주기적 변동 (200°C 기준)
    data[:, 0] = 200 + 5 * np.sin(2 * np.pi * time_points / 100) + \
                 2 * np.sin(2 * np.pi * time_points / 50) + \
                 np.random.normal(0, 2, total_length)
    
    # 2. 압력: 주기적 변동 (120 bar 기준)
    data[:, 1] = 120 + 3 * np.sin(2 * np.pi * time_points / 80) + \
                 np.random.normal(0, 1.5, total_length)
    
    # 3. 진동: 안정적 패턴 (1.0 mm/s 기준)
    data[:, 2] = 1.0 + 0.1 * np.sin(2 * np.pi * time_points / 120) + \
                 np.random.normal(0, 0.15, total_length)
    
    # 4. 사이클타임: 안정적 패턴 (50초 기준)
    data[:, 3] = 50 + 1 * np.sin(2 * np.pi * time_points / 90) + \
                 np.random.normal(0, 1, total_length)
    
    # 노이즈 추가
    data += np.random.normal(0, noise_level, data.shape)
    
    # Sliding window로 시퀀스 생성
    sequences = []
    for i in range(n_samples):
        seq = data[i:i+seq_len]  # (seq_len, n_features)
        sequences.append(seq)
    
    X = np.array(sequences, dtype=np.float32)  # (n_samples, seq_len, n_features)
    
    print(f"✅ 데이터 생성 완료: {X.shape}, dtype: {X.dtype}")
    return X


def load_moldset_data(
    csv_path: str,
    seq_len: int = 50,
    use_label: bool = True,
    use_normal_only: bool = False,
    chunk_size: int = None
) -> tuple:
    """
    Moldset 데이터셋 로드 및 시계열 변환 (대용량 데이터 최적화)
    
    Args:
        csv_path: CSV 파일 경로
        seq_len: 시퀀스 길이
        use_label: Label 정보 사용 여부 (True면 PassOrFail 컬럼 처리)
        use_normal_only: 정상 데이터만 사용 (Label이 있을 때만 유효)
        chunk_size: 청크 크기 (대용량 파일 처리용, None이면 전체 로드)
    
    Returns:
        (sequences, labels) 형태의 tuple
        - sequences: (n_samples, seq_len, n_features) 형태의 numpy 배열
        - labels: (n_samples,) 형태의 numpy 배열 (use_label=False면 None)
    """
    print(f"📂 Moldset 데이터셋 로드: {csv_path}")
    
    # 파일 크기 확인
    file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    print(f"   파일 크기: {file_size_mb:.2f} MB")
    
    # 대용량 파일 처리 (100MB 이상)
    if file_size_mb > 100 and chunk_size is None:
        chunk_size = 50000  # 기본 청크 크기
        print(f"   ⚠️  대용량 파일 감지 - 청크 단위 처리 활성화")
    
    # CSV 파일 읽기 (청크 처리 또는 전체 로드)
    if chunk_size:
        print(f"   📖 청크 단위 로드 중... (청크 크기: {chunk_size:,}행)")
        chunks = []
        for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
            chunks.append(chunk)
            if (i + 1) % 10 == 0:
                print(f"      청크 {i+1} 로드 완료...")
        df = pd.concat(chunks, ignore_index=True)
        print(f"   ✅ 전체 데이터 로드 완료")
    else:
        df = pd.read_csv(csv_path)
    
    print(f"   원본 데이터 형태: {df.shape}")
    
    # Label 컬럼 확인 및 분리
    label_col = None
    if use_label:
        label_candidates = ['PassOrFail', 'label', 'Label', 'anomaly', 'Anomaly', 'target', 'Target']
        for col in label_candidates:
            if col in df.columns:
                label_col = col
                break
        
        if label_col:
            labels = df[label_col].values
            df = df.drop(columns=[label_col])
            print(f"   Label 컬럼 발견: {label_col}")
            print(f"   - 정상(0): {(labels == 0).sum()}개")
            print(f"   - 이상(1): {(labels == 1).sum()}개")
        else:
            print(f"   ⚠️  Label 컬럼 없음 (Unsupervised 학습)")
            labels = None
    else:
        labels = None
    
    # 인덱스 컬럼 제거 (첫 번째 컬럼이 인덱스인 경우)
    if df.columns[0] == 'Unnamed: 0' or df.columns[0] == '':
        df = df.drop(columns=[df.columns[0]])
    
    # 센서 컬럼 자동 매핑
    sensor_mapping = {
        '온도': [],
        '압력': [],
        '사이클타임': [],
        '기타': []
    }
    
    # 온도 센서 찾기
    temp_keywords = ['temperature', 'Temperature', 'Temp', 'temp', 'Barrel_Temperature', 
                     'Hopper_Temperature', 'Mold_Temperature']
    for col in df.columns:
        if any(kw in col for kw in temp_keywords):
            sensor_mapping['온도'].append(col)
    
    # 압력 센서 찾기
    pressure_keywords = ['pressure', 'Pressure', 'Press', 'press', 'Injection_Pressure', 
                         'Back_Pressure', 'Switch_Over_Pressure']
    for col in df.columns:
        if any(kw in col for kw in pressure_keywords):
            sensor_mapping['압력'].append(col)
    
    # 사이클타임 찾기
    cycle_keywords = ['Cycle_Time', 'cycle_time', 'CycleTime', 'cycle', 'Cycle']
    for col in df.columns:
        if any(kw in col for kw in cycle_keywords):
            sensor_mapping['사이클타임'].append(col)
            break  # 하나만 사용
    
    # 기타 센서 (속도, RPM, 시간 등)
    other_keywords = ['Speed', 'RPM', 'Time', 'Position', 'Position']
    for col in df.columns:
        if col not in sensor_mapping['온도'] and col not in sensor_mapping['압력'] and \
           col not in sensor_mapping['사이클타임']:
            if any(kw in col for kw in other_keywords):
                sensor_mapping['기타'].append(col)
    
    # 사용할 센서 컬럼 선택
    selected_cols = []
    
    # 온도: 평균값 사용 (여러 온도 센서가 있으면 평균)
    if sensor_mapping['온도']:
        if len(sensor_mapping['온도']) == 1:
            selected_cols.append(sensor_mapping['온도'][0])
            print(f"   ✅ 온도 센서: {sensor_mapping['온도'][0]}")
        else:
            # 여러 온도 센서의 평균 계산
            df['temperature_avg'] = df[sensor_mapping['온도']].mean(axis=1)
            selected_cols.append('temperature_avg')
            print(f"   ✅ 온도 센서: {len(sensor_mapping['온도'])}개 → 평균값 사용")
    else:
        print(f"   ⚠️  온도 센서 없음")
    
    # 압력: 최대값 또는 평균값 사용
    if sensor_mapping['압력']:
        if len(sensor_mapping['압력']) == 1:
            selected_cols.append(sensor_mapping['압력'][0])
            print(f"   ✅ 압력 센서: {sensor_mapping['압력'][0]}")
        else:
            # 여러 압력 센서의 평균 계산
            df['pressure_avg'] = df[sensor_mapping['압력']].mean(axis=1)
            selected_cols.append('pressure_avg')
            print(f"   ✅ 압력 센서: {len(sensor_mapping['압력'])}개 → 평균값 사용")
    else:
        print(f"   ⚠️  압력 센서 없음")
    
    # 사이클타임
    if sensor_mapping['사이클타임']:
        selected_cols.append(sensor_mapping['사이클타임'][0])
        print(f"   ✅ 사이클타임: {sensor_mapping['사이클타임'][0]}")
    else:
        print(f"   ⚠️  사이클타임 없음")
    
    # 기타 센서 (속도 등) - 숫자형만 추가 (문자열/날짜 제외)
    if sensor_mapping['기타']:
        numeric_count = 0
        for col in sensor_mapping['기타']:
            if numeric_count >= 2:  # 최대 2개만
                break
            # 숫자형 컬럼만 추가 (문자열/날짜 제외)
            try:
                # 숫자로 변환 가능한지 테스트
                test_val = pd.to_numeric(df[col], errors='coerce')
                if not test_val.isna().all():  # 모두 NaN이 아니면 숫자형
                    selected_cols.append(col)
                    print(f"   ✅ 추가 센서: {col}")
                    numeric_count += 1
                else:
                    print(f"   ⚠️  {col}: 숫자형이 아님 (제외)")
            except:
                print(f"   ⚠️  {col}: 숫자형이 아님 (제외)")
    
    if not selected_cols:
        raise ValueError("사용 가능한 센서 컬럼을 찾을 수 없습니다.")
    
    print(f"\n   최종 사용 컬럼 ({len(selected_cols)}개): {selected_cols}")
    
    # NaN 값 처리 및 숫자형 변환
    data = df[selected_cols].copy()
    
    # 모든 컬럼을 숫자형으로 변환 (문자열/날짜 제외)
    numeric_cols = []
    for col in data.columns:
        try:
            # 숫자로 변환 시도
            data[col] = pd.to_numeric(data[col], errors='coerce')
            # 변환 후 모두 NaN이 아니면 사용
            if not data[col].isna().all():
                numeric_cols.append(col)
            else:
                print(f"   ⚠️  {col}: 숫자 변환 실패, 제외")
        except:
            print(f"   ⚠️  {col}: 숫자 변환 실패, 제외")
    
    # 숫자형 컬럼만 사용
    if len(numeric_cols) == 0:
        raise ValueError("사용 가능한 숫자형 센서 컬럼이 없습니다.")
    
    data = data[numeric_cols]
    if len(numeric_cols) < len(selected_cols):
        print(f"   ⚠️  숫자형 컬럼만 사용: {len(numeric_cols)}개 (원래 {len(selected_cols)}개)")
    
    nan_before = data.isna().sum().sum()
    if nan_before > 0:
        print(f"   ⚠️  NaN 값 발견: {nan_before}개")
        # 전방 채우기 후 후방 채우기 (pandas 최신 버전 호환)
        try:
            data = data.ffill().bfill()
        except AttributeError:
            # 구버전 pandas 호환
            data = data.fillna(method='ffill').fillna(method='bfill')
        # 그래도 NaN이 있으면 0으로 채우기
        data = data.fillna(0)
        print(f"   ✅ NaN 값 처리 완료")
    
    # 정상 데이터만 사용 옵션
    if use_normal_only and labels is not None:
        normal_indices = labels == 0
        data = data[normal_indices]
        if labels is not None:
            labels = labels[normal_indices]
        print(f"   ✅ 정상 데이터만 사용: {len(data)}개 샘플")
    
    # 데이터를 numpy 배열로 변환 (float32로 명시적 변환)
    data_values = data.values.astype(np.float32)  # (n_timesteps, n_features)
    
    print(f"\n   데이터 전처리 완료: {data_values.shape}")
    print(f"   - 시계열 길이: {len(data_values)}")
    print(f"   - 특징 수: {data_values.shape[1]}")
    print(f"   - 데이터 타입: {data_values.dtype}")
    
    # Sliding window로 시계열 시퀀스 생성 (메모리 효율적)
    print(f"\n   🔄 Sliding Window로 시계열 변환 중... (seq_len={seq_len})")
    
    n_samples = len(data_values) - seq_len + 1
    n_features = data_values.shape[1]
    
    # 메모리 효율적인 시퀀스 생성
    print(f"   예상 시퀀스 수: {n_samples:,}개")
    
    # 대용량 데이터의 경우 청크 단위로 처리
    if n_samples > 100000:
        print(f"   ⚠️  대용량 데이터 감지 - 청크 단위 변환")
        chunk_size = 50000
        sequences_chunks = []
        labels_chunks = []
        
        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            chunk_sequences = []
            chunk_labels = []
            
            for i in range(start_idx, end_idx):
                seq = data_values[i:i+seq_len]  # (seq_len, n_features)
                chunk_sequences.append(seq)
                
                if labels is not None:
                    chunk_labels.append(labels[i + seq_len - 1])
            
            sequences_chunks.append(np.array(chunk_sequences, dtype=np.float32))
            if labels is not None:
                labels_chunks.append(np.array(chunk_labels, dtype=np.int64))
            
            print(f"      청크 변환 완료: {end_idx:,}/{n_samples:,}")
        
        X = np.concatenate(sequences_chunks, axis=0)
        y = np.concatenate(labels_chunks, axis=0) if labels is not None else None
    else:
        # 소규모 데이터는 일반 처리
        sequences = []
        sequence_labels = []
        
        for i in range(n_samples):
            seq = data_values[i:i+seq_len]  # (seq_len, n_features)
            sequences.append(seq)
            
            if labels is not None:
                sequence_labels.append(labels[i + seq_len - 1])
        
        X = np.array(sequences, dtype=np.float32)  # (n_samples, seq_len, n_features)
        y = np.array(sequence_labels, dtype=np.int64) if labels is not None else None
    
    print(f"   ✅ 시계열 변환 완료: {X.shape}")
    if y is not None:
        print(f"   - 정상 시퀀스: {(y == 0).sum()}개")
        print(f"   - 이상 시퀀스: {(y == 1).sum()}개")
    
    return X, y


def load_time_series_from_csv(
    csv_path: str,
    seq_len: int = 50,
    columns: list = None,
    use_moldset_format: bool = True
) -> np.ndarray:
    """
    CSV 파일에서 시계열 데이터 로드 (기존 함수와 호환성 유지)
    
    Args:
        csv_path: CSV 파일 경로
        seq_len: 시퀀스 길이
        columns: 사용할 컬럼 리스트 (None이면 자동 선택)
        use_moldset_format: Moldset 데이터셋 형식 자동 감지 및 사용
    
    Returns:
        (n_samples, seq_len, n_features) 형태의 numpy 배열
    """
    # Moldset 형식 자동 감지
    if use_moldset_format:
        # Moldset 데이터셋 특징 확인
        df_sample = pd.read_csv(csv_path, nrows=5)
        moldset_indicators = [
            'PassOrFail' in df_sample.columns,
            'Cycle_Time' in df_sample.columns,
            'Barrel_Temperature' in str(df_sample.columns),
            'Injection_Pressure' in str(df_sample.columns)
        ]
        
        if any(moldset_indicators):
            print(f"   🔍 Moldset 데이터셋 형식 감지")
            sequences, _ = load_moldset_data(
                csv_path=csv_path,
                seq_len=seq_len,
                use_label=False,  # Unsupervised 학습용
                use_normal_only=False
            )
            return sequences
    
    # 기존 로직 (일반 CSV 파일)
    print(f"📂 CSV 파일 로드: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # 컬럼 선택
    if columns is None:
        # 자동으로 센서 컬럼 찾기
        sensor_cols = ['temperature', 'pressure', 'vibration', 'cycle_time']
        columns = [col for col in sensor_cols if col in df.columns]
    
    if not columns:
        raise ValueError("센서 컬럼을 찾을 수 없습니다.")
    
    print(f"   사용 컬럼: {columns}")
    
    # 데이터 정규화 (선택사항) 및 float32 변환
    data = df[columns].values.astype(np.float32)
    
    # NaN 처리
    if np.isnan(data).any():
        data_df = pd.DataFrame(data)
        try:
            data = data_df.ffill().bfill().fillna(0).values.astype(np.float32)
        except AttributeError:
            # 구버전 pandas 호환
            data = data_df.fillna(method='ffill').fillna(method='bfill').fillna(0).values.astype(np.float32)
    
    # Sliding window로 시퀀스 생성
    sequences = []
    for i in range(len(data) - seq_len + 1):
        seq = data[i:i+seq_len]  # (seq_len, n_features)
        sequences.append(seq)
    
    X = np.array(sequences, dtype=np.float32)  # (n_samples, seq_len, n_features)
    
    print(f"✅ 데이터 로드 완료: {X.shape}")
    return X


def train_model(
    train_data: np.ndarray,
    model_type: str = "TimesNet",
    seq_len: int = 50,
    epochs: int = 20,
    batch_size: int = None,
    model_save_path: str = None,
    device: str = None
):
    """
    DeepOD 모델 학습 (A100 GPU 최적화)
    
    Args:
        train_data: (n_samples, seq_len, n_features) 형태의 학습 데이터
        model_type: 모델 타입 ('TimesNet', 'AnomalyTransformer', 'TranAD')
        seq_len: 시퀀스 길이
        epochs: 학습 에포크
        batch_size: 배치 크기 (None이면 자동 최적화)
        model_save_path: 모델 저장 경로
        device: 디바이스 ('cuda' 또는 'cpu', None이면 자동 감지)
    """
    print(f"\n{'='*80}")
    print(f"🎓 DeepOD {model_type} 모델 학습 시작")
    print(f"{'='*80}")
    print(f"   - 데이터 형태: {train_data.shape}")
    print(f"   - 시퀀스 길이: {seq_len}")
    print(f"   - 에포크: {epochs}")
    
    # GPU 디바이스 자동 감지
    if device is None:
        device = get_device()
    else:
        print(f"   디바이스: {device}")
    
    # 배치 크기 자동 최적화
    if batch_size is None:
        n_features = train_data.shape[2]
        batch_size = get_optimal_batch_size(device, seq_len, n_features)
    else:
        print(f"   배치 크기: {batch_size} (사용자 지정)")
    
    # 모델 경로 설정
    if model_save_path is None:
        model_save_path = str(project_root / "2_model_training" / f"anomaly_model_{model_type.lower()}.pkl")
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # 모델 초기화
    try:
        from deepod.models.time_series import TimesNet, AnomalyTransformer, TranAD
        
        if model_type == "TimesNet":
            model = TimesNet(
                seq_len=seq_len,
                epochs=epochs,
                batch_size=batch_size,
                device=device,
                verbose=1
            )
        elif model_type == "AnomalyTransformer":
            model = AnomalyTransformer(
                seq_len=seq_len,
                epochs=epochs,
                batch_size=batch_size,
                device=device,
                verbose=1
            )
        elif model_type == "TranAD":
            model = TranAD(
                seq_len=seq_len,
                epochs=epochs,
                batch_size=batch_size,
                device=device,
                verbose=1
            )
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
        
        print(f"✅ {model_type} 모델 초기화 완료")
        
    except ImportError:
        print("❌ DeepOD 미설치: pip install deepod")
        return
    
    # 학습 실행
    print(f"\n🚀 학습 시작...")
    start_time = datetime.now()
    
    # 데이터 형태 확인 및 변환
    print(f"   입력 데이터 형태: {train_data.shape}")
    print(f"   입력 데이터 dtype: {train_data.dtype}")
    
    # DeepOD TimesNet은 내부적으로 sliding window를 적용하므로
    # 원본 시계열 데이터 (n_timesteps, n_features) 형태를 기대합니다
    # 하지만 우리는 이미 windowing된 (n_samples, seq_len, n_features) 형태를 가지고 있습니다
    # 따라서 원본 시계열로 변환해야 합니다
    
    if len(train_data.shape) == 3:
        # (n_samples, seq_len, n_features) -> 원본 시계열로 변환
        # DeepOD TimesNet은 내부적으로 sliding window를 적용하므로 원본 시계열이 필요
        print(f"   ⚠️  Windowed 데이터 감지: {train_data.shape}")
        print(f"   원본 시계열로 변환 중...")
        
        n_samples, seq_len_actual, n_features = train_data.shape
        
        # 원본 시계열 복원: 첫 번째 시퀀스의 모든 시점 + 나머지 시퀀스의 마지막 시점만
        # 메모리 효율적인 방법
        original_length = seq_len_actual + (n_samples - 1)  # 첫 시퀀스 길이 + 나머지 시퀀스의 마지막 시점들
        train_data_original = np.zeros((original_length, n_features), dtype=np.float32)
        
        # 첫 번째 시퀀스: 모든 시점 복사
        train_data_original[:seq_len_actual] = train_data[0]
        
        # 나머지 시퀀스: 마지막 시점만 복사 (중복 제거)
        for i in range(1, n_samples):
            train_data_original[seq_len_actual + i - 1] = train_data[i, -1]
        
        print(f"   ✅ 원본 시계열 변환 완료: {train_data_original.shape}")
        print(f"   - 시계열 길이: {train_data_original.shape[0]}")
        print(f"   - 특징 수: {train_data_original.shape[1]}")
        
        train_data = train_data_original
    elif len(train_data.shape) == 2:
        # 이미 원본 시계열 형태 (n_timesteps, n_features)
        print(f"   ✅ 원본 시계열 형태 확인: {train_data.shape}")
        if train_data.dtype != np.float32:
            train_data = train_data.astype(np.float32)
    else:
        raise ValueError(f"지원하지 않는 데이터 형태: {train_data.shape}")
    
    print(f"   최종 데이터 형태: {train_data.shape}")
    print(f"   예상 형태: (n_timesteps, n_features)")
    
    model.fit(train_data)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"✅ 학습 완료 (소요 시간: {elapsed:.1f}초)")
    
    # 임계값 설정 (정상 데이터의 95% percentile)
    scores = model.decision_function(train_data)
    threshold = np.percentile(scores, 95)
    
    print(f"\n📊 학습 결과:")
    print(f"   - 임계값: {threshold:.4f}")
    print(f"   - 스코어 범위: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"   - 스코어 평균: {scores.mean():.4f}")
    
    # 모델 저장
    import pickle
    with open(model_save_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'threshold': threshold,
            'model_type': model_type,
            'seq_len': seq_len,
            'train_shape': train_data.shape
        }, f)
    
    print(f"✅ 모델 저장 완료: {model_save_path}")
    
    return model, threshold


def test_model(model, threshold, test_data: np.ndarray):
    """모델 테스트"""
    print(f"\n{'='*80}")
    print(f"🔍 모델 테스트")
    print(f"{'='*80}")
    
    scores = model.decision_function(test_data)
    predictions = scores > threshold
    
    print(f"   - 테스트 샘플: {len(test_data)}")
    print(f"   - 이상 탐지: {predictions.sum()}개 ({predictions.sum()/len(predictions)*100:.1f}%)")
    print(f"   - 정상: {(~predictions).sum()}개 ({(~predictions).sum()/len(predictions)*100:.1f}%)")
    
    return scores, predictions


def main():
    parser = argparse.ArgumentParser(description="DeepOD TimesNet 학습 스크립트")
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="CSV 파일 경로 (None이면 시뮬레이션 데이터 사용)"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="TimesNet",
        choices=["TimesNet", "AnomalyTransformer", "TranAD"],
        help="모델 타입"
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=50,
        help="시퀀스 길이"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="학습 에포크"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="배치 크기 (None이면 GPU 메모리에 맞게 자동 최적화, 기본값: A100 40GB 기준 128)"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5000,
        help="시뮬레이션 데이터 샘플 수 (CSV 사용 시 무시)"
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default=None,
        help="모델 저장 경로"
    )
    parser.add_argument(
        "--use_label",
        action="store_true",
        help="Label 정보 사용 (PassOrFail 등)"
    )
    parser.add_argument(
        "--use_normal_only",
        action="store_true",
        help="정상 데이터만 사용 (Label이 있을 때만 유효)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="디바이스 지정 (None이면 자동 감지)"
    )
    
    args = parser.parse_args()
    
    # GPU 디바이스 확인
    device = get_device() if args.device is None else args.device
    
    # 데이터 로드
    if args.data_path and os.path.exists(args.data_path):
        # Moldset 데이터셋 형식 자동 감지
        df_sample = pd.read_csv(args.data_path, nrows=5)
        is_moldset = any([
            'PassOrFail' in df_sample.columns,
            'Cycle_Time' in df_sample.columns,
            'Barrel_Temperature' in str(df_sample.columns)
        ])
        
        if is_moldset:
            print("🔍 Moldset 데이터셋 형식 감지")
            train_data, train_labels = load_moldset_data(
                csv_path=args.data_path,
                seq_len=args.seq_len,
                use_label=args.use_label,
                use_normal_only=args.use_normal_only
            )
            
            if train_labels is not None:
                print(f"\n📊 Label 통계:")
                print(f"   - 정상 시퀀스: {(train_labels == 0).sum()}개")
                print(f"   - 이상 시퀀스: {(train_labels == 1).sum()}개")
        else:
            train_data = load_time_series_from_csv(
                args.data_path,
                seq_len=args.seq_len,
                use_moldset_format=False
            )
    else:
        print("⚠️  CSV 파일 없음. 시뮬레이션 데이터 사용")
        train_data = create_synthetic_time_series_data(
            n_samples=args.n_samples,
            seq_len=args.seq_len
        )
    
    # 학습
    model, threshold = train_model(
        train_data=train_data,
        model_type=args.model_type,
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_save_path=args.model_save_path,
        device=device
    )
    
    # 테스트 (일부 데이터 사용)
    test_data = train_data[:1000]  # 처음 1000개 샘플로 테스트
    test_model(model, threshold, test_data)
    
    print(f"\n{'='*80}")
    print(f"✨ 모든 작업 완료!")
    print(f"{'='*80}")
    print(f"\n다음 명령어로 모델을 사용할 수 있습니다:")
    print(f"\n```python")
    print(f"from agent_system.models.anomaly_detector import AnomalyDetectionModel")
    print(f"")
    print(f"detector = AnomalyDetectionModel()")
    print(f"result = detector.detect_anomaly({{")
    print(f"    'temperature': 235.0,")
    print(f"    'pressure': 120.0,")
    print(f"    'vibration': 1.2,")
    print(f"    'cycle_time': 52.0")
    print(f"}})")
    print(f"```")
    print(f"\n📝 Moldset 데이터셋 사용 예시 (리눅스/A100):")
    print(f"python train_anomaly_detector.py \\")
    print(f"    --data_path dataset/moldset_labeled.csv \\")
    print(f"    --seq_len 50 \\")
    print(f"    --epochs 20 \\")
    print(f"    --use_label \\")
    print(f"    --use_normal_only \\")
    print(f"    --batch_size 128  # A100 40GB 최적화 (자동 감지 시 생략 가능)")


if __name__ == "__main__":
    main()

