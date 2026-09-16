from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-small"

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Downloading pretrained model...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("Model downloaded successfully!")