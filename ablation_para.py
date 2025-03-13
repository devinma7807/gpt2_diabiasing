import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import argparse
from paraphrase_plain import ParaphraseGPT
import numpy as np
import spacy
import pandas as pd
from collections import Counter

MODEL_PATH = "/Users/etsu/Desktop/CS 224N/final_project/cs224n_gpt/plain/4-5e-05-paraphrase_plain.pt"
device = "cpu"
saved = torch.load(MODEL_PATH, weights_only=False, map_location=device)

gpt2_eval_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
gpt2_eval_model.eval()
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

model = ParaphraseGPT(saved['args'])
model.load_state_dict(saved['model'])
model = model.to(device)
model.eval()

nlp = spacy.load("en_core_web_sm")

# Define paraphrase and non-paraphrase examples
paraphrase_pairs = [
    ("The cat sat on the mat.", "The feline rested on the rug."),
    ("She enjoys reading books.", "She likes to read novels."),
    ("The sun is shining brightly.", "It's a sunny day outside."),
    ("He quickly finished his homework.", "He completed his assignment in no time."),
    ("The dog barked loudly.", "The canine made a loud noise."),
    ("They traveled to Paris last summer.", "Last summer, they visited Paris."),
    ("I love drinking coffee in the morning.", "Every morning, I enjoy a cup of coffee."),
    ("She is very intelligent.", "She is highly smart."),
    ("The baby is sleeping peacefully.", "The infant is resting quietly."),
    ("The weather is cold today.", "It is chilly outside today.")
]

non_paraphrase_pairs = [
    ("The cat sat on the mat.", "The dog barked at the stranger."),
    ("She enjoys reading books.", "He dislikes going to the gym."),
    ("The sun is shining brightly.", "It is raining heavily."),
    ("He quickly finished his homework.", "She forgot to do her homework."),
    ("The dog barked loudly.", "Birds chirped in the morning."),
    ("They traveled to Paris last summer.", "She stayed home all summer."),
    ("I love drinking coffee in the morning.", "He prefers drinking tea at night."),
    ("She is very intelligent.", "He struggles with basic math."),
    ("The baby is sleeping peacefully.", "Cars honk loudly in traffic."),
    ("The weather is cold today.", "It is warm and sunny outside.")
]

# Function to format inputs for the model
def format_input(sentence1, sentence2):
    input_text = f'Is "{sentence1}" a paraphrase of "{sentence2}"? Answer "yes" or "no": '
    encoding = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
    return encoding["input_ids"], encoding["attention_mask"]

# Predict function
def predict_paraphrase(model, sentence1, sentence2):
    input_ids, attention_mask = format_input(sentence1, sentence2)
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        predicted_label = torch.argmax(logits, dim=-1).item()
    return predicted_label  # 1 = Paraphrase, 0 = Non-Paraphrase

# Evaluate model performance
correct = 0
total = len(paraphrase_pairs) + len(non_paraphrase_pairs)

for s1, s2 in paraphrase_pairs:
    prediction = predict_paraphrase(model, s1, s2)
    if prediction == 1:  # Should be paraphrase
        correct += 1

for s1, s2 in non_paraphrase_pairs:
    prediction = predict_paraphrase(model, s1, s2)
    if prediction == 0:  # Should be non-paraphrase
        correct += 1

accuracy = (correct / total) * 100

# Print results
print(f"\n🔹 **Paraphrase Detection Accuracy: {accuracy:.2f}%** 🔹\n")