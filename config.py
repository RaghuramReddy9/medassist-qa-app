HF_FULL_MODEL = "RaghuramReddyT/medassist-qa"   # your HF repo or Hf open source any model

SAFETY_PREFIX = (
    "You are a helpful medical information assistant. You are not a doctor. "
    "Provide general educational information and advise the user to consult a "
    "qualified healthcare professional for personal medical concerns.\n\n"
)

# If your HF repo is a LoRA adapter, set the base model you fine-tuned:
BASE_MODEL_FOR_LORA = "tiiuae/falcon-rw-1b"  # change if different
