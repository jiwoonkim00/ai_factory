"""
4_agent_system/models/lora_inference.py
LoRA 파인튜닝 모델 추론 엔진
"""

import os
import torch
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ============================================================================
# LoRA 추론 엔진
# ============================================================================

class LoRAInferenceEngine:
    """LoRA 파인튜닝 모델 추론 엔진"""
    
    def __init__(self, 
                 base_model_path: str = "Qwen/Qwen2.5-7B-Instruct",
                 lora_adapter_path: str = None):
        """
        Args:
            base_model_path: Base 모델 경로 또는 HuggingFace 모델명
            lora_adapter_path: LoRA 어댑터 경로 (None이면 자동 탐지)
        """
        self.base_model_path = base_model_path
        
        # 경로 자동 설정
        if lora_adapter_path is None:
            from ..utils.config import LORA_MODEL_PATH
            lora_adapter_path = str(LORA_MODEL_PATH)
        
        self.lora_adapter_path = lora_adapter_path
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        print(f"🤖 LoRA 모델 로딩 중...")
        self._load_model()
    
    def _load_model(self):
        """Base 모델 + LoRA 어댑터 로드"""
        try:
            # 토크나이저
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=True
            )
            
            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Base 모델
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # LoRA 어댑터 (학습 완료 후)
            adapter_exists = os.path.exists(self.lora_adapter_path) and \
                           os.path.exists(os.path.join(self.lora_adapter_path, "adapter_config.json"))
            
            if adapter_exists:
                try:
                    self.model = PeftModel.from_pretrained(
                        self.model, 
                        self.lora_adapter_path
                    )
                    print(f"✅ LoRA 어댑터 로드 완료: {self.lora_adapter_path}")
                except Exception as e:
                    print(f"⚠️  LoRA 어댑터 로드 실패: {e}")
                    print(f"   Base 모델만 사용합니다.")
            else:
                print(f"⚠️  LoRA 어댑터 없음. Base 모델만 사용: {self.base_model_path}")
                print(f"   경로 확인: {self.lora_adapter_path}")
            
            self.model.eval()
            self.is_loaded = True
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            self.is_loaded = False
            raise
    
    def generate(self, 
                 instruction: str,
                 input_text: str,
                 max_new_tokens: int = 1024,
                 temperature: float = 0.7) -> str:
        """
        텍스트 생성
        
        Args:
            instruction: 시스템 프롬프트
            input_text: 사용자 입력 텍스트
            max_new_tokens: 최대 생성 토큰 수
            temperature: 생성 온도
        
        Returns:
            생성된 텍스트
        """
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            raise RuntimeError("모델이 로드되지 않았습니다. _load_model()을 먼저 실행하세요.")
        
        try:
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": input_text}
            ]
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.1
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return response
            
        except Exception as e:
            print(f"❌ 텍스트 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return f"[오류] 텍스트 생성 실패: {str(e)}"
