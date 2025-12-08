"""
제조 공정 AI Agent LoRA 파인튜닝 스크립트 (고도화 버전)
A100 40GB 환경 최적화 (QLoRA 4-bit)

개선사항:
1. CoT가 포함된 긴 시퀀스 처리 (max_length 확장)
2. 노이즈 데이터 전처리 강화
3. 다양한 Instruction 형식 대응
4. 학습 안정성 개선 (Gradient clipping, Warmup)
"""

import os
import torch
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import numpy as np


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-7B-Instruct",
        metadata={"help": "모델 이름 또는 경로. Qwen2.5-7B-Instruct 또는 meta-llama/Meta-Llama-3.1-8B-Instruct"}
    )
    use_4bit: bool = field(
        default=True,
        metadata={"help": "4-bit quantization 사용 여부 (메모리 절약)"}
    )
    use_flash_attention: bool = field(
        default=True,
        metadata={"help": "Flash Attention 2 사용 여부 (A100에서 권장)"}
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default="../1_data_generation/sft_data/manufacturing_sft_train.jsonl",
        metadata={"help": "학습 데이터 경로"}
    )
    max_length: int = field(
        default=3072,  # 🆕 CoT 포함으로 증가 (2048 → 3072)
        metadata={"help": "최대 시퀀스 길이"}
    )
    truncation_strategy: str = field(
        default="right",
        metadata={"help": "길이 초과 시 truncation 방향"}
    )


@dataclass
class LoraArguments:
    lora_r: int = field(
        default=64,
        metadata={"help": "LoRA attention dimension"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha (스케일링)"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "LoRA를 적용할 모듈"}
    )


class ManufacturingDataProcessor:
    """제조 데이터 전처리 (CoT 및 노이즈 대응)"""
    
    def __init__(self, tokenizer, max_length: int = 3072):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stats = {
            "total": 0,
            "truncated": 0,
            "avg_length": 0,
            "max_observed": 0
        }
    
    def load_jsonl(self, data_path: str) -> Dataset:
        """JSONL 파일 로드"""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return Dataset.from_list(data)
    
    def clean_input(self, text: str) -> str:
        """🆕 노이즈 제거 (선택적) - 너무 과한 노이즈만 정리"""
        # 연속된 빈 줄 제거
        import re
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        return text.strip()
    
    def format_prompt_qwen(self, example: Dict) -> str:
        """Qwen2.5 ChatML 형식"""
        prompt = f"""<|im_start|>system
{example['instruction']}<|im_end|>
<|im_start|>user
{example['input']}<|im_end|>
<|im_start|>assistant
{example['output']}<|im_end|>"""
        return prompt
    
    def format_prompt_llama(self, example: Dict) -> str:
        """Llama 3.1 형식"""
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{example['instruction']}<|eot_id|><|start_header_id|>user<|end_header_id|>

{example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{example['output']}<|eot_id|>"""
        return prompt
    
    def format_prompt(self, example: Dict) -> str:
        """모델에 맞는 프롬프트 포맷 선택"""
        model_name = self.tokenizer.name_or_path.lower()
        
        if "llama" in model_name:
            return self.format_prompt_llama(example)
        else:  # Qwen 및 기타
            return self.format_prompt_qwen(example)
    
    def tokenize_function(self, examples: List[Dict]) -> Dict:
        """토크나이징 with 통계 수집"""
        prompts = []
        for ex in examples:
            # 입력 정리 (선택적)
            cleaned_input = self.clean_input(ex['input'])
            example_dict = {
                'instruction': ex['instruction'],
                'input': cleaned_input,
                'output': ex['output']
            }
            prompts.append(self.format_prompt(example_dict))
        
        # 토크나이징
        tokenized = self.tokenizer(
            prompts,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
            add_special_tokens=True  # 🆕 명시적으로 추가
        )
        
        # 통계 수집
        for input_ids in tokenized["input_ids"]:
            length = len(input_ids)
            self.stats["total"] += 1
            self.stats["avg_length"] += length
            self.stats["max_observed"] = max(self.stats["max_observed"], length)
            if length >= self.max_length:
                self.stats["truncated"] += 1
        
        # Labels = input_ids (causal LM)
        tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
        
        return tokenized
    
    def prepare_dataset(self, data_path: str) -> Dataset:
        """전체 데이터셋 준비"""
        dataset = self.load_jsonl(data_path)
        
        print(f"✅ 데이터셋 로드 완료: {len(dataset)}개 샘플")
        
        # 배치 단위로 토크나이징
        tokenized_dataset = dataset.map(
            lambda batch: self.tokenize_function(
                [{"instruction": inst, "input": inp, "output": out} 
                 for inst, inp, out in zip(batch["instruction"], batch["input"], batch["output"])]
            ),
            batched=True,
            batch_size=10,
            remove_columns=dataset.column_names,
            desc="토크나이징 중..."
        )
        
        # 통계 출력
        if self.stats["total"] > 0:
            avg_len = self.stats["avg_length"] / self.stats["total"]
            trunc_pct = (self.stats["truncated"] / self.stats["total"]) * 100
            
            print(f"\n📊 토크나이징 통계:")
            print(f"   - 평균 길이: {avg_len:.0f} 토큰")
            print(f"   - 최대 길이: {self.stats['max_observed']} 토큰")
            print(f"   - Truncation 발생: {self.stats['truncated']}개 ({trunc_pct:.1f}%)")
            
            if trunc_pct > 20:
                print(f"\n⚠️  경고: {trunc_pct:.0f}%의 샘플이 잘렸습니다!")
                print(f"   → max_length를 {int(self.stats['max_observed'] * 1.1)}로 증가 권장")
        
        return tokenized_dataset
    
    def print_sample(self, dataset: Dataset, idx: int = 0):
        """샘플 출력 (디버깅용)"""
        sample = dataset[idx]
        decoded = self.tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
        
        print(f"\n{'='*80}")
        print(f"📄 샘플 #{idx} (길이: {len(sample['input_ids'])} 토큰)")
        print(f"{'='*80}")
        print(decoded[:1000])
        if len(decoded) > 1000:
            print(f"\n... (중략, 총 {len(decoded)}자) ...\n")
            print(decoded[-500:])
        print(f"{'='*80}\n")


def setup_model_and_tokenizer(model_args: ModelArguments):
    """모델 및 토크나이저 설정"""
    
    # BitsAndBytes 설정 (4-bit quantization)
    if model_args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("✅ 4-bit Quantization 활성화")
    else:
        bnb_config = None
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
        use_fast=True  # 🆕 Fast tokenizer 사용
    )
    
    # 패딩 토큰 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"📝 토크나이저 정보:")
    print(f"   - Vocab size: {len(tokenizer)}")
    print(f"   - PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"   - EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"
    )
    
    # Gradient checkpointing (메모리 절약)
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    print(f"✅ 모델 로드 완료: {model_args.model_name_or_path}")
    print(f"   - Attention: SDPA (PyTorch Native)")
    print(f"   - 메모리 사용량: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    return model, tokenizer


def setup_lora(model, lora_args: LoraArguments):
    """LoRA 설정"""
    
    # k-bit 학습 준비
    model = prepare_model_for_kbit_training(model)
    
    # LoRA 구성
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print(f"\n✅ LoRA 설정 완료:")
    print(f"   - Rank: {lora_args.lora_r}")
    print(f"   - Alpha: {lora_args.lora_alpha}")
    print(f"   - Target modules: {len(lora_args.target_modules)}개")
    
    return model


def compute_metrics(eval_pred):
    """🆕 평가 지표 계산"""
    predictions, labels = eval_pred
    
    # Perplexity 계산 (간단 버전)
    # 실제로는 loss를 사용하는 게 더 정확
    return {}


def main():
    print("=" * 80)
    print("제조 공정 AI Agent LoRA 파인튜닝 (고도화 버전)")
    print("A100 40GB 최적화 + CoT + 노이즈 대응")
    print("=" * 80)
    
    # 인자 설정
    model_args = ModelArguments()
    data_args = DataArguments()
    lora_args = LoraArguments()
    
    # GPU 확인
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA를 사용할 수 없습니다. GPU 환경을 확인하세요.")
    
    print(f"\n🚀 GPU 정보:")
    print(f"   - Device: {torch.cuda.get_device_name(0)}")
    print(f"   - VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   - CUDA: {torch.version.cuda}")
    
    # 1. 모델 및 토크나이저 로드
    print("\n" + "=" * 80)
    print("1️⃣ 모델 로딩 중...")
    print("=" * 80)
    model, tokenizer = setup_model_and_tokenizer(model_args)
    
    # 2. LoRA 설정
    print("\n" + "=" * 80)
    print("2️⃣ LoRA 설정 중...")
    print("=" * 80)
    model = setup_lora(model, lora_args)
    
    # 3. 데이터 준비
    print("\n" + "=" * 80)
    print("3️⃣ 데이터 준비 중...")
    print("=" * 80)
    processor = ManufacturingDataProcessor(tokenizer, max_length=data_args.max_length)
    train_dataset = processor.prepare_dataset(data_args.data_path)
    
    # 샘플 출력 (첫 번째 샘플)
    processor.print_sample(train_dataset, idx=0)
    
    # Train/Eval 분리 (90:10)
    split_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"✅ 데이터 분할 완료:")
    print(f"   - 학습 샘플: {len(train_dataset)}개")
    print(f"   - 검증 샘플: {len(eval_dataset)}개")
    
    # 4. 학습 설정
    print("\n" + "=" * 80)
    print("4️⃣ 학습 설정...")
    print("=" * 80)
    
    output_dir = "./manufacturing_lora_output"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,  # 🆕 긴 시퀀스 대응 (4→2)
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,  # 🆕 Effective batch size = 16 유지
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_grad_norm=1.0,  # 🆕 Gradient clipping 추가
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        report_to="none",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,  # 🆕 데이터 로딩 병렬화
        dataloader_pin_memory=True,
        group_by_length=False,  # 🆕 길이별 그룹화 비활성화 (다양성 확보)
    )
    
    # Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True
    )
    
    # 5. Trainer 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # 6. 학습 시작
    print("\n" + "=" * 80)
    print("5️⃣ 학습 시작! 🚀")
    print("=" * 80)
    print(f"   - Epochs: {training_args.num_train_epochs}")
    print(f"   - Batch size: {training_args.per_device_train_batch_size}")
    print(f"   - Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"   - Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"   - Learning rate: {training_args.learning_rate}")
    print(f"   - Max sequence length: {data_args.max_length}")
    print(f"   - Total steps: {len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs}")
    print()
    
    # 학습 실행
    train_result = trainer.train()
    
    # 7. 모델 저장
    print("\n" + "=" * 80)
    print("6️⃣ 학습 완료! 모델 저장 중...")
    print("=" * 80)
    
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 학습 결과 저장
    with open(os.path.join(output_dir, "training_results.json"), "w") as f:
        json.dump(train_result.metrics, f, indent=2)
    
    print(f"✅ 모델 저장 완료: {output_dir}")
    print(f"   - LoRA 어댑터: {output_dir}/adapter_model.safetensors")
    print(f"   - 설정 파일: {output_dir}/adapter_config.json")
    print(f"   - 학습 결과: {output_dir}/training_results.json")
    
    # 최종 Loss 출력
    print(f"\n📊 최종 학습 지표:")
    print(f"   - Train Loss: {train_result.metrics.get('train_loss', 'N/A'):.4f}")
    print(f"   - Learning Rate: {train_result.metrics.get('train_runtime', 0) / 3600:.2f} hours")
    
    # 8. 추론 테스트
    print("\n" + "=" * 80)
    print("7️⃣ 추론 테스트 (CoT 생성 확인)")
    print("=" * 80)
    
    test_input = """[공정 이상 이벤트]
설비: 사출기-3호기
발생시각: 2024-12-08 09:15:00
이상유형: 온도 이상

[센서 데이터]
- 실린더 온도: 235°C (정상: 200°C)
- 임계값: 설정값 대비 ±15°C
- 증상: 제품 변형, 치수 불량

[RAG 검색 결과]
[과거 이력 #2023-11-20] 동일 증상으로 히터 고장 확인됨. 교체 후 정상화.
[설비 매뉴얼 3.2절] 온도 이상 발생 시 히터 성능 저하 가능성 높음.

위 정보를 바탕으로 원인 분석과 조치 가이드를 작성하세요."""
    
    messages = [
        {"role": "system", "content": "당신은 제조 현장의 전문 설비 엔지니어입니다."},
        {"role": "user", "content": test_input}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    print(f"   - Input 길이: {inputs['input_ids'].shape[1]} 토큰")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1  # 🆕 반복 방지
    )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    print("\n[테스트 응답]")
    print("-" * 80)
    print(response[:800])
    if len(response) > 800:
        print("\n... (중략) ...\n")
        print(response[-200:])
    print("-" * 80)
    
    # CoT 포함 여부 확인
    has_cot = "상황 분석" in response or "추론 과정" in response or "단계" in response
    print(f"\n{'✅' if has_cot else '⚠️ '} CoT 추론 과정 {'포함됨' if has_cot else '미포함 (추가 학습 필요)'}")
    
    print("\n" + "=" * 80)
    print("✨ 모든 작업 완료!")
    print("=" * 80)
    print(f"\n다음 명령어로 모델을 사용할 수 있습니다:")
    print(f"\n```python")
    print(f"from peft import PeftModel")
    print(f"from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"")
    print(f"base_model = AutoModelForCausalLM.from_pretrained('{model_args.model_name_or_path}')")
    print(f"model = PeftModel.from_pretrained(base_model, '{output_dir}')")
    print(f"tokenizer = AutoTokenizer.from_pretrained('{output_dir}')")
    print(f"```")
    print("=" * 80)


if __name__ == "__main__":
    main()