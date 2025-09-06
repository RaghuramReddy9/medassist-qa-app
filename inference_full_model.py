from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config import HF_FULL_MODEL, SAFETY_PREFIX

# Classes used:
# - AutoTokenizer: picks the right tokenizer class for this model
# - AutoModelForCausalLM: loads a text-generation "brain" for causal LM
# - pipeline("text-generation"): convenience wrapper to generate answers

def build_prompt(q: str) -> str:
    return f"{SAFETY_PREFIX}User question: {q}\nAnswer:"

def load_pipeline():
    tok = AutoTokenizer.from_pretrained(HF_FULL_MODEL, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(HF_FULL_MODEL, trust_remote_code=True)
    return pipeline(
        "text-generation",
        model=mdl,
        tokenizer=tok,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.6,
        top_p=0.8,
    )

def ask(gen, q: str) -> str:
    out = gen(build_prompt(q), use_cache=False)[0]["generated_text"]
    # Only keep text after "Answer:"
    if "Answer:" in out:
        answer = out.split("Answer:")[-1].strip()
    else:
        answer = out.strip()

    # Add disclaimer to every response
    disclaimer = "\n\nNote: This information is for general education only, not a substitute for medical advice. Please consult a qualified healthcare professional for personal concerns."
    return answer + disclaimer


if __name__ == "__main__":
    gen = load_pipeline()
    for q in [
        "I have a sore throat and mild fever. What can I do at home?",
        "Is ibuprofen okay if I have stomach acidity?",
        "When should I see a doctor for chest pain?"
    ]:
        print("\nQ:", q)
        print("A:", ask(gen, q))
