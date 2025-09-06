# MedAssist-QA (Fine-tuned Medical FAQ Bot)

This is a fine-tuned medical FAQ chatbot built using Hugging Face `transformers`.  
It runs as a CLI assistant and provides general healthcare information.  
⚠️ Not medical advice.  

## Features
- Loads custom fine-tuned Hugging Face model (`RaghuramReddyT/medassist-qa`)
- CLI chat loop with clean UX (colors, safety disclaimer)
- Logs all Q&A to `chat_log.txt`
- Auto-appends responsible disclaimer

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
│── inference_full_model.py
│── chat.py
│── requirements.txt
│── chat_log.txt
```
