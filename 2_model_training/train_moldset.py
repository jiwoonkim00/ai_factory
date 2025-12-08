#!/usr/bin/env python3
"""
Moldset 데이터셋 학습 실행 스크립트 (리눅스/A100 최적화)

사용법:
    python train_moldset.py
    python train_moldset.py --config all
    python train_moldset.py --config normal_only
"""

import os
import sys
import argparse
from pathlib import Path

# 프로젝트 루트 경로
project_root = Path(__file__).resolve().parent.parent
dataset_path = project_root / "dataset"
training_script = Path(__file__).resolve().parent / "train_anomaly_detector.py"

def check_gpu():
    """GPU 확인"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ GPU 감지: {gpu_name}")
            print(f"   GPU 메모리: {gpu_memory:.1f} GB")
            return True
        else:
            print("⚠️  GPU 없음 - CPU 모드로 실행됩니다")
            return False
    except ImportError:
        print("⚠️  PyTorch 미설치")
        return False

def check_dataset():
    """데이터셋 파일 확인"""
    datasets = {
        'labeled': dataset_path / "moldset_labeled.csv",
        'labeled_rg3': dataset_path / "moldset_labeled_rg3.csv",
        'labeled_cn7': dataset_path / "moldset_labeled_cn7.csv",
        'unlabeled_rg3': dataset_path / "moldset_unlabeled_rg3.csv",
        'unlabeled_cn7': dataset_path / "moldset_unlabeled_cn7.csv",
        'unlabeled': dataset_path / "unlabeled_data.csv",
        'labeled_data': dataset_path / "labeled_data.csv"
    }
    
    available = {}
    for name, path in datasets.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            available[name] = {'path': path, 'size_mb': size_mb}
            print(f"   ✅ {name}: {path.name} ({size_mb:.2f} MB)")
        else:
            print(f"   ❌ {name}: 파일 없음")
    
    return available

def train_labeled_normal_only():
    """정상 데이터만 사용하여 학습"""
    print("\n" + "="*80)
    print("1️⃣  정상 데이터만 사용 (Unsupervised 학습)")
    print("="*80)
    
    dataset_file = dataset_path / "moldset_labeled.csv"
    if not dataset_file.exists():
        print(f"❌ 데이터셋 파일 없음: {dataset_file}")
        return False
    
    cmd = [
        sys.executable,
        str(training_script),
        "--data_path", str(dataset_file),
        "--seq_len", "50",
        "--epochs", "20",
        "--use_label",
        "--use_normal_only",
        "--model_type", "TimesNet"
    ]
    
    print(f"실행 명령어: {' '.join(cmd)}")
    os.system(' '.join(cmd))
    return True

def train_labeled_all():
    """Label 정보 사용 (정상/이상 모두 포함)"""
    print("\n" + "="*80)
    print("2️⃣  Label 정보 사용 (정상/이상 모두 포함)")
    print("="*80)
    
    dataset_file = dataset_path / "moldset_labeled.csv"
    if not dataset_file.exists():
        print(f"❌ 데이터셋 파일 없음: {dataset_file}")
        return False
    
    cmd = [
        sys.executable,
        str(training_script),
        "--data_path", str(dataset_file),
        "--seq_len", "50",
        "--epochs", "20",
        "--use_label",
        "--model_type", "TimesNet"
    ]
    
    print(f"실행 명령어: {' '.join(cmd)}")
    os.system(' '.join(cmd))
    return True

def train_unlabeled():
    """Unlabeled 데이터로 학습"""
    print("\n" + "="*80)
    print("3️⃣  Unlabeled 데이터 사용")
    print("="*80)
    
    # 작은 파일부터 시도
    dataset_files = [
        dataset_path / "moldset_unlabeled_rg3.csv",
        dataset_path / "moldset_unlabeled_cn7.csv",
        dataset_path / "unlabeled_data.csv"
    ]
    
    dataset_file = None
    for f in dataset_files:
        if f.exists():
            dataset_file = f
            break
    
    if dataset_file is None:
        print("❌ Unlabeled 데이터셋 파일 없음")
        return False
    
    cmd = [
        sys.executable,
        str(training_script),
        "--data_path", str(dataset_file),
        "--seq_len", "50",
        "--epochs", "20",
        "--model_type", "TimesNet"
    ]
    
    print(f"실행 명령어: {' '.join(cmd)}")
    os.system(' '.join(cmd))
    return True

def train_custom(data_path, seq_len=50, epochs=20, use_label=False, use_normal_only=False, batch_size=None):
    """커스텀 학습"""
    cmd = [
        sys.executable,
        str(training_script),
        "--data_path", str(data_path),
        "--seq_len", str(seq_len),
        "--epochs", str(epochs),
        "--model_type", "TimesNet"
    ]
    
    if use_label:
        cmd.append("--use_label")
    if use_normal_only:
        cmd.append("--use_normal_only")
    if batch_size:
        cmd.extend(["--batch_size", str(batch_size)])
    
    print(f"실행 명령어: {' '.join(cmd)}")
    os.system(' '.join(cmd))

def main():
    parser = argparse.ArgumentParser(description="Moldset 데이터셋 학습 스크립트")
    parser.add_argument(
        "--config",
        type=str,
        default="normal_only",
        choices=["normal_only", "all", "unlabeled", "all_configs"],
        help="학습 설정 (normal_only: 정상 데이터만, all: Label 모두, unlabeled: Unlabeled, all_configs: 모든 설정)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="커스텀 데이터셋 경로"
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
        "--use_label",
        action="store_true",
        help="Label 정보 사용"
    )
    parser.add_argument(
        "--use_normal_only",
        action="store_true",
        help="정상 데이터만 사용"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="배치 크기 (None이면 자동 최적화)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 Moldset 데이터셋 학습 스크립트 (리눅스/A100 최적화)")
    print("="*80)
    
    # GPU 확인
    print("\n🔍 GPU 확인:")
    has_gpu = check_gpu()
    
    # 데이터셋 확인
    print("\n📂 데이터셋 확인:")
    available_datasets = check_dataset()
    
    if not available_datasets:
        print("\n❌ 사용 가능한 데이터셋이 없습니다!")
        return
    
    # 환경 변수 설정
    if has_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        print("\n⚙️  환경 변수 설정:")
        print(f"   CUDA_VISIBLE_DEVICES=0")
        print(f"   PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512")
    
    # 커스텀 데이터셋 사용
    if args.data_path:
        print(f"\n📊 커스텀 데이터셋 사용: {args.data_path}")
        train_custom(
            data_path=args.data_path,
            seq_len=args.seq_len,
            epochs=args.epochs,
            use_label=args.use_label,
            use_normal_only=args.use_normal_only,
            batch_size=args.batch_size
        )
        return
    
    # 설정에 따른 학습 실행
    if args.config == "normal_only":
        train_labeled_normal_only()
    elif args.config == "all":
        train_labeled_all()
    elif args.config == "unlabeled":
        train_unlabeled()
    elif args.config == "all_configs":
        print("\n🔄 모든 설정으로 학습 시작...\n")
        train_labeled_normal_only()
        train_labeled_all()
        train_unlabeled()
    
    print("\n" + "="*80)
    print("✅ 학습 완료!")
    print("="*80)
    print("\n학습된 모델 위치:")
    model_path = project_root / "2_model_training" / "anomaly_model_timesnet.pkl"
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  ✅ {model_path} ({size_mb:.2f} MB)")
    else:
        print(f"  ⚠️  {model_path} (아직 생성되지 않음)")

if __name__ == "__main__":
    main()

