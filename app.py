import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Config
MODEL_NAME = "RaghuramReddyT/medassist-qa"

# Few-shot prompt with examples
def build_prompt(user_q: str) -> str:
    return (
        "You are a medical FAQ bot. Answer clearly and briefly.\n\n"
        "Q: What is fever?\n"
        "A: Fever is a body temperature above 100.4°F (38°C), usually caused by an infection.\n\n"
        "Q: What are the symptoms of dehydration?\n"
        "A: Thirst, dry mouth, fatigue, dizziness, and headache.\n\n"
        f"Q: {user_q}\n"
        "A:"
    )

# Cache the pipeline so it loads only once
@st.cache_resource
def load_pipeline():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    return pipeline(
        "text-generation",
        model=mdl,
        tokenizer=tok,
        max_new_tokens=40,   # shorter answers for speed + less repetition
        do_sample=False,
        temperature=0.5,
        top_p=0.9,
    )

# Load model once
gen = load_pipeline()

# Streamlit UI
st.set_page_config(page_title="MedAssist-QA Bot", page_icon="🩺", layout="wide")
st.title("🩺 MedAssist-QA")
st.markdown(
    "<h4 style='color:gray;'>Fine-tuned Medical FAQ Assistant (Not medical advice)</h4>",
    unsafe_allow_html=True
)

# Input box
user_question = st.text_area(
    "Ask a medical question:", 
    height=120, 
    placeholder="e.g., What are the symptoms of dehydration?"
)

if st.button("Submit"):
    if user_question.strip():
        with st.spinner("Thinking..."):
            output = gen(build_prompt(user_question), use_cache=False)[0]["generated_text"]

            # Extract only answer part
            if "A:" in output:
                answer = output.split("A:")[-1].strip()
            else:
                answer = output.strip()

            # Show in styled card
            st.markdown(
                f"""
                <div style="padding: 20px; border-radius: 12px; background-color: #1E1E1E;">
                    <b style="color:#4CAF50; font-size:18px;">Answer:</b><br>
                    <span style="color:white; font-size:16px;">{answer}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Disclaimer
            st.markdown(
                "<p style='font-size:12px; color:gray;'>Note: This information is for general education only, not a substitute for medical advice. Please consult a qualified healthcare professional for personal concerns.</p>",
                unsafe_allow_html=True,
            )
