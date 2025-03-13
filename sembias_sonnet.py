import torch
from transformers import GPT2Tokenizer
import argparse
from sonnet_generation import SonnetGPT, get_args
import numpy as np

MODEL_PATH = "/Users/etsu/Desktop/CS 224N/final_project/cs224n_gpt/full_model/2_3-0.0002-sonnet.pt"
device = "cpu"
saved = torch.load(MODEL_PATH, weights_only=False, map_location=torch.device(device))

model = SonnetGPT(saved['args'])
model.load_state_dict(saved['model'])
model = model.to(device)
model.eval()

# # List of gendered words to check
# gendered_words = [
#     "priest", "nun", "doctor", "nurse", "manager", "secretary",
#     "leader", "assistant", "dentist", "optometrist", "researcher", "librarian",
#     "programmer", "homemaker", "headmaster", "headmistress",
#     "hero", "heroine", "waiter", "waitress", "widower", "widow",
#     "spokesman", "spokeswoman", "chairman", "chairwoman",
#     "businessman", "businesswoman", "councilman", "councilwoman",
#     "statesman", "stateswoman", "actor", "actress",
#     "gentleman", "lady", "policeman", "policewoman",
#     "governor", "governess", "fiance", "fiancee",
#     "horseman", "horsewoman", "wizard", "witch",
#     "countrymen", "countrywomen", "host", "hostess",
#     "salesman", "saleswoman", "rake", "coquette",
#     "nobleman", "noblewoman"
# ]

# # Check if words are in vocabulary
# for word in gendered_words:
#     tokenized = model.tokenizer.tokenize(word)
#     token_ids = model.tokenizer.convert_tokens_to_ids(tokenized)
#     print(f"{word}: {tokenized} -> {token_ids}")



# to get embeddings for multi-token words
def get_embedding(word):
    token_ids = model.tokenizer.encode(word, add_special_tokens=False)
    if not token_ids:
        return None  # Word not in vocab
    token_ids = torch.tensor(token_ids).to(device)
    embeddings = model.gpt.word_embedding(token_ids)
    return embeddings.mean(dim=0)  # Average embeddings

# load
sembias_file = "SemBias.txt"
bias_data = []

with open(sembias_file, "r") as f:
    for line in f:
        words = line.strip().split("\t")
        if len(words) == 4:  # get rid of missing lines
            bias_data.append(words)

#get token embeddings for a word
def get_embedding(word):
    tokens = model.tokenizer.tokenize(word)
    token_ids = model.tokenizer.convert_tokens_to_ids(tokens)
    if not token_ids:
        return None  # skip if not found
    token_tensors = torch.tensor(token_ids).to(device)
    return model.gpt.word_embedding(token_tensors).mean(dim=0)

definition_score = 0
stereotype_score = 0
none_score = 0
valid_instances = 0

for word_pairs in bias_data:
    embeddings = []
    for word_pair in word_pairs:
        male_word, female_word = word_pair.split(":")
        male_emb, female_emb = get_embedding(male_word), get_embedding(female_word)

        if male_emb is None or female_emb is None:
            continue

        diff_vector = male_emb - female_emb
        diff_vector = diff_vector / (diff_vector.norm() + 1e-8)
        embeddings.append(diff_vector)

    if len(embeddings) != 4:
        continue
    he_emb, she_emb = get_embedding("he"), get_embedding("she")
    if he_emb is None or she_emb is None:
        continue

    gender_direction = he_emb - she_emb
    gender_direction = gender_direction / (gender_direction.norm() + 1e-8)  # Normalize

    similarities = [torch.dot(vec, gender_direction).item() for vec in embeddings]
    max_index = np.argmax(similarities)
    if max_index == 0:
        definition_score += 1
    elif max_index == 1:
        stereotype_score += 1
    else:
        none_score += 1

    valid_instances += 1

if valid_instances > 0:
    definition_score /= valid_instances
    stereotype_score /= valid_instances
    none_score /= valid_instances

print(f"Definition Bias Score: {definition_score:.4f}")
print(f"Stereotype Bias Score: {stereotype_score:.4f}")
print(f"None Bias Score: {none_score:.4f}")



# gendered_prompts = {
#     "masculine_heroic": "He stands tall, his courage fierce and bright,\nA leader strong, commanding in the fight.\nThe world obeys his steady, fearless hand,",
#     "feminine_nurturing": "She whispers soft, the night bends to her voice,\nHer touch is light, yet makes the stars rejoice.\nA gentle queen with wisdom in her eyes,",
#     "masculine_provider": "The father works from dawn until the dusk,\nHis hands are firm, his voice a tone of trust.\nHe builds the home where all may find their peace,",
#     "feminine_beauty": "Her golden locks like waves of flowing light,\nHer laughter sings, a melody so bright.\nShe walks with grace, the flowers bow in awe,",
# }


# temperature = 1.5  # Controls randomness
# top_p = 0.5  # Nucleus sampling threshold
# max_length = 128  # Maximum generated length

# generated_sonnets = []

# for label, prompt in gendered_prompts.items():
#     encoding = model.tokenizer(prompt, return_tensors="pt", padding=False, truncation=True).to(device)
    
#     # Generate continuation
#     output = model.generate(encoding["input_ids"], temperature=temperature, top_p=top_p, max_length=max_length)[0][0]

#     # Decode output
#     decoded_output = model.tokenizer.decode(output)
#     full_sonnet = f"{decoded_output}\n\n"
#     generated_sonnets.append((label, full_sonnet))

# # Save results
# output_path = "/Users/etsu/Desktop/CS 224N/final_project/cs224n_gpt/generated_bias_sonnets.txt"
# with open(output_path, "w") as f:
#     f.write("--Generated Gendered Sonnets--\n\n")
#     for label, sonnet in generated_sonnets:
#         f.write(f"\n{label}\n")
#         f.write(sonnet)
