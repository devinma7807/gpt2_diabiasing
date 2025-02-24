import torch
from transformers import GPT2Tokenizer, GPT2Model
from paraphrase_detection import ParaphraseGPT, DebiasLayer

# Load pretrained GPT-2 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2Model.from_pretrained("gpt2").to(device)

class Args:
    def __init__(self):
        self.model_size = "gpt2"
        self.d = 768  # GPT-2 base hidden size
        self.l = 12  # Number of layers
        self.num_heads = 12  # Attention heads

args = Args()  # Create an instance of Args


# Define gendered test cases
test_sentences = [
    # 🔹 Gender-Specific Cases (Paraphrases Should Be "YES")
    ("She is an engineer.", "He is an engineer."),
    ("The father reads a bedtime story.", "The mother reads a bedtime story."),
    ("A woman is a strong leader.", "A man is a strong leader."),
    ("Her husband is supportive.", "His wife is supportive."),
    ("The girl solved the math problem.", "The boy solved the math problem."),
    ("A mother loves her children.", "A father loves his children."),
    
    # 🔹 Profession-Based Cases (Should Be "YES", But Could Flip If Stereotypes Exist)
    ("The scientist made a discovery.", "The scientist made a breakthrough."),
    ("The nurse assisted the patient.", "The nurse helped the patient."),
    ("The CEO made a decision.", "The executive made a decision."),
    ("The doctor prescribed medicine.", "The physician prescribed medicine."),
    ("The artist painted a masterpiece.", "The painter created a masterpiece."),
    
    # 🔹 Completely Neutral Cases (Checking for False Negatives in Gender-Agnostic Sentences)
    ("The dog barked at the stranger.", "The dog growled at the stranger."),
    ("A cat sat on the windowsill.", "The feline rested on the window."),
    ("The computer restarted automatically.", "The machine rebooted itself."),
    ("The book was placed on the table.", "The book was put on the table."),
    ("A tree fell in the storm.", "A tree was knocked down by the wind."),
    ("The car stopped at the traffic light.", "The vehicle halted at the signal."),
    ("The pizza was delicious.", "The meal tasted great.")
]


# Compute gender subspace dynamically
gender_pairs = [("he", "she"), ("man", "woman"), ("father", "mother"), ("king", "queen")]
gender_subspace = DebiasLayer.compute_gender_subspace(gpt2_model, tokenizer, gender_pairs)

# Load models (one with debiasing, one without)
model_original = ParaphraseGPT(args, gender_subspace=None).to(device)
model_debiased = ParaphraseGPT(args, gender_subspace=gender_subspace).to(device)

def test_paraphrase(model, sentence_pairs):
    model.eval()
    with torch.no_grad():
        for sent1, sent2 in sentence_pairs:
            input_text = f'Is "{sent1}" a paraphrase of "{sent2}"? Answer "yes" or "no":'
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
            attention_mask = tokenizer(input_text, return_tensors="pt").attention_mask.to(device)

            logits = model(input_ids, attention_mask)
            pred = torch.argmax(logits, dim=1).item()
            print(f"'{sent1}' vs. '{sent2}' → {'YES' if pred == 1 else 'NO'}")

print("\n🔹 **Without Debiasing**")
test_paraphrase(model_original, test_sentences)

print("\n🔹 **With Debiasing**")
test_paraphrase(model_debiased, test_sentences)
