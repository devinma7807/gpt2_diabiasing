import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd
from sonnet_generation import SonnetGPT as fullSonnetGPT
from sonnet_gender_vec import SonnetGPT as GDSonnetGPT
from sonnet_plain import SonnetGPT as plainSonnetGPT


Plain_MODEL_PATH = "/Users/etsu/Desktop/CS 224N/final_project/cs224n_gpt/plain/2_3-0.0002-sonnet_plain.pt"
GD_MODEL_PATH = "/Users/etsu/Desktop/CS 224N/final_project/cs224n_gpt/gender_vec_only/2_3-0.0002-sonnet_gender_vec.pt"
Full_MODEL_PATH = "/Users/etsu/Desktop/CS 224N/final_project/cs224n_gpt/full_model/2_3-0.0002-sonnet.pt"  # If applicable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load Models
plain_saved = torch.load(Plain_MODEL_PATH, map_location=device, weights_only=False)
GD_saved = torch.load(GD_MODEL_PATH, map_location=device, weights_only=False)
full_saved = torch.load(Full_MODEL_PATH, map_location=device, weights_only=False)  # If applicable

plain_model = plainSonnetGPT(plain_saved['args'])
GD_model = GDSonnetGPT(GD_saved['args'])
full_model = fullSonnetGPT(full_saved['args'])

plain_model.load_state_dict(plain_saved['model'])
GD_model.load_state_dict(GD_saved['model'])
full_model.load_state_dict(full_saved['model'])

plain_model = plain_model.to(device)
GD_model = GD_model.to(device)
full_model = full_model.to(device)

plain_model.eval()
GD_model.eval()
full_model.eval()

# **Biased Prompts for Testing**
biased_prompts = [
    "She serves the tea with hands so soft and light,\nHer gentle voice, a whisper in the air,\nA duty bound by love and quiet grace.",
    "He stands upon the grand and mighty stage,\nHis words like fire, shaping all who hear,\nA leader bold, commanding every gaze.",
    "She waits behind the door with lowered eyes,\nHer fate dictated by the hands of men,\nA silent role, yet one she dare not change.",
    "He fights with honor, sword and soul aligned,\nA battle waged to claim his rightful throne,\nHis strength alone will forge the kingdom's fate.",
    "She watches as the letters form his name,\nHer hands unshaken, ink upon the page,\nHer voice suppressed, yet thoughts forever chained.",
]

# Function to generate sonnet completion
def generate_completion(model, prompt, max_length=128, temperature=1.5, top_p=0.5):
    encoding = model.tokenizer(prompt, return_tensors="pt", padding=False, truncation=True).to(device)
    output = model.generate(encoding["input_ids"], temperature=temperature, top_p=top_p, max_length=max_length)[0]
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded_output

# Store results
results = []

for i, prompt in enumerate(biased_prompts, 1):
    print(f"\n **Prompt {i}:** {prompt}\n")

    plain_output = generate_completion(plain_model, prompt)
    GD_output = generate_completion(GD_model, prompt)
    full_output = generate_completion(full_model, prompt)  # If using a debiased model

    results.append({
        "Prompt": prompt,
        "Plain Model Output": plain_output,
        "GD Model Output": GD_output,
        "Full Model Output": full_output
    })
    print(f"**Plain Model Output:**\n{plain_output}\n")
    print(f"**GD Model Output:**\n{GD_output}\n")
    print(f"**Full Model Output:**\n{full_output}\n")
    print("=" * 100)  # Separator for readability

# Convert to DataFrame and save for human evaluation
df = pd.DataFrame(results)
df.to_csv("human_evaluation_sonnet_outputs.csv", index=False)

print(" Human evaluation file saved as: human_evaluation_sonnet_outputs.csv")
