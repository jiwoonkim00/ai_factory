# test_lora_inference.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ✅ 여기를 네가 실제로 쓴 베이스 모델 이름으로 맞춰줘!
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"   # 예시
LORA_ADAPTER_DIR = "./manufacturing_lora_output"  # train_lora.py에서 저장한 경로


def load_model():
    print("🔹 Base 모델 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",          # A100이면 자동으로 GPU 할당
        torch_dtype=torch.float16,  # VRAM 절약
    )

    print("🔹 LoRA 어댑터 적용 중...")
    model = PeftModel.from_pretrained(
        base_model,
        LORA_ADAPTER_DIR,
    )

    model.eval()
    return tokenizer, model


def run_inference(tokenizer, model):
    # 🔧 테스트용 입력 (원하면 여기만 바꿔서 계속 시험해 보면 됨)
    instruction = (
        "당신은 제조 현장의 전문 설비 엔지니어입니다. "
        "아래 공정 이상 상황에 대해 원인을 분석하고, 체크리스트와 조치 가이드, "
        "8D Report 초안을 작성해주세요."
        "위 정보를 바탕으로 아래 형식을 **엄격히 지켜서** 답변하세요."

        "1) 🧠 상황 분석 및 추론 과정"
        "2) ✅ 원인 분석 결과 (우선순위 3개)"
        "3) 📝 우선 점검 체크리스트"
        "4) 🔧 단계별 조치 가이드 (1차/2차/3차)"
        "5) 📋 8D Report 초안 (D1~D7 항목 포함)"

    )

    input_text = f"""{instruction}

[공정 이상 이벤트]
설비: 사출기-2호기
발생시각: 2024-09-12 14:32:10
이상유형: 온도 이상

[센서 데이터]
- 실린더 온도: 245°C (정상: 200°C)
- 임계값: 설정값 대비 ±15°C
- 증상: 제품 변형, 치수 불량

[RAG 검색 결과]
[과거 이력 #2023-08-15] 동일 증상으로 냉각 시스템 막힘 확인됨. 해당 부품 교체 후 정상화.
[설비 매뉴얼 3.2절] 온도 이상 발생 시 냉각 시스템 막힘 가능성이 가장 높음. 즉시 점검 권장.

위 정보를 바탕으로 답변을 생성하세요.
"""

    print("\n🧾 ===== 입력 프롬프트 =====\n")
    print(input_text)

    inputs = tokenizer(
        input_text,
        return_tensors="pt"
    ).to(model.device)

    print("\n🤖 추론 중...\n")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=800,   # 출력 길이
            temperature=0.3,      # 랜덤성(낮을수록 보수적)
            top_p=0.9,
            do_sample=True,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n📄 ===== 모델 출력 =====\n")
    print(decoded)


def main():
    tokenizer, model = load_model()
    run_inference(tokenizer, model)


if __name__ == "__main__":
    main()
