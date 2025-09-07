# MedAssist-QA (Fine-tuned Medical FAQ Bot)

This is a fine-tuned medical FAQ chatbot built using Hugging Face `transformers`.  
It runs as a CLI assistant and provides general healthcare information.  
⚠️ Not medical advice.  

## Features
- Loads fine-tuned Hugging Face model: [`RaghuramReddyT/medassist-qa`](https://huggingface.co/RaghuramReddyT/medassist-qa)
- **CLI mode** → interactive chat from terminal
- **Streamlit mode** → simple web app UI (`http://localhost:8501`)
- Auto-appends a **safety disclaimer**
- Clean UX with styled answers

## Example Usage
```bash
python chat.py
```
## Samples
```bash
You: What are the symptoms of dehydration?
Bot: Dry mouth, thirst, fatigue, dizziness, and headaches.

Note: This information is for general education only, not a substitute for medical advice. Please consult a qualified healthcare professional.
```
## Streamlit
```
streamlit run app.py

Then open browser → http://localhost:8501
```
## Tech stack
```
--Python 3.10+

--Hugging Face Transformers

--PyTorch

--PEFT (for LoRA support)

--Colorama (CLI polish)
```
## Structure
```
medassist-qa-app/
│── config.py
│── app.py
│── inference_full_model.py
│── chat.py
│── requirements.txt
│── chat_log.txt
```
