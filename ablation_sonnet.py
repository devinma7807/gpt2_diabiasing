import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import argparse
from sonnet_plain import SonnetGPT, get_args
import numpy as np
import spacy
import pandas as pd
from collections import Counter


lexicon = pd.read_csv("agency_power.csv")
agency_mapping = {"agency_pos": 1, "agency_neg": -1, "agency_equal": 0}

# Convert agency labels to numbers
agency_dict = {verb: agency_mapping.get(agency, 0) for verb, agency in zip(lexicon["verb"], lexicon["agency"])}


# Model path you provided
MODEL_PATH = "/Users/etsu/Desktop/CS 224N/final_project/cs224n_gpt/plain/2_3-0.0002-sonnet_plain.pt"
device = "cpu"
# Load device
saved = torch.load(MODEL_PATH, weights_only=False, map_location=torch.device(device))

# Load a pre-trained GPT-2 model for fluency evaluation
gpt2_eval_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
gpt2_eval_model.eval()
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

model = SonnetGPT(saved['args'])
model.load_state_dict(saved['model'])
model = model.to(device)
model.eval()

nlp = spacy.load("en_core_web_sm")

female_prompts = [
    "She walks through gardens bathed in golden light,\nHer voice, a melody that stirs the air,\nA heart untamed, yet filled with boundless grace.", 
    "She whispers secrets to the endless night,\nSoft winds embrace the dreams she dares to weave,\nHer soul ignites like stars that burn so bright.",
    "She dances freely with the morning breeze,\nHer laughter echoing in silent halls,\nA spirit wild, untethered by the past.",
    "She stands alone where oceans kiss the shore,\nHer gaze unshaken as the tempests rise,\nNo storm nor tide could make her heart implore.",
    "She finds her strength within the morning sun,\nThe golden rays awaken all she seeks,\nHer path is hers, her journey just begun.",
    "She holds the fire of dawn within her hands,\nA whispered dream beneath the silent moon,\nNo chains nor walls could make her heart withstand.",
    "She lifts her eyes to touch the endless sky,\nEach step she takes is bold, unchained, and free,\nNo shadow dares to quell the light inside.",
    "She sings of hope where sorrow used to dwell,\nHer words like rivers carving through the stone,\nA tale of light, a story she must tell.",
    "She moves like wind upon the silver sea,\nHer laughter carries through the midnight air,\nA fleeting ghost, yet full of melody.",
    "She carves her fate from whispers in the night,\nHer soul a tempest none could ever tame,\nShe weaves her dreams and steps into the light."
]  

generated_sonnets = []
for prompt in female_prompts:
    encoding = model.tokenizer(prompt, return_tensors='pt', padding=False, truncation=True).to(device)

    # Generate continuation
    output = model.generate(encoding['input_ids'], temperature=1.5, top_p=0.5, max_length=128)[0]

    # Decode output properly
    decoded_output = model.tokenizer.decode(output[0], skip_special_tokens=True)

    generated_sonnets.append(decoded_output)
    # print(f"\n🔹 **Prompt:**\n{prompt}\n")
    # print(f"🔹 **Generated Sonnet:**\n{decoded_output}\n")
    # print("-" * 80)

active_count, passive_count, total_verbs = 0, 0, 0

for sonnet in generated_sonnets:
    doc = nlp(sonnet)
    for token in doc:
        if token.pos_ == "VERB":  # Identify verbs
            verb = token.lemma_  # Get base form
            total_verbs += 1
            if verb in agency_dict:
                if agency_dict[verb] > 0:  # High agency → active
                    active_count += 1
                else:  # Low agency → passive
                    passive_count += 1

# Avoid division by zero
if total_verbs > 0:
    active_percentage = (active_count / total_verbs)
    passive_percentage = (passive_count / total_verbs)
else:
    active_percentage = passive_percentage = 0

print(f"\n🔹 **Ablation Study Results:**")
print(f"\nAgency: {active_percentage:.2f}")


# Compute Repetition (Bigram Repetition %)
def compute_repetition(text):
    words = text.split()
    bigrams = [tuple(words[i:i+2]) for i in range(len(words)-1)]
    bigram_counts = Counter(bigrams)
    repeated_bigrams = sum(1 for count in bigram_counts.values() if count > 1)
    return (repeated_bigrams / len(bigrams)) if bigrams else 0

repetition_scores = [compute_repetition(sonnet) for sonnet in generated_sonnets]
average_repetition = sum(repetition_scores) / len(repetition_scores)

# Compute Diversity (Unique Generations %)
unique_sonnets = len(set(generated_sonnets))
diversity_score = (unique_sonnets / len(generated_sonnets))

# Print results
print(f"\nRepetition: {average_repetition:.2f}")
print(f"\nDiversity: {diversity_score:.2f}")
