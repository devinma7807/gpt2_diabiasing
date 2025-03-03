import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import GPT2Tokenizer

class PowerTransformer(nn.Module):
    def __init__(self, hidden_dim, vocab_size, beta=5.0):
        """
        - Masked Reconstruction Training
        - Connotation Frames for Agency Control
        - Vocabulary Boosting at Decoding
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.beta = beta  # Controls strength of vocabulary boosting

        # Learnable transformation for modifying agency-related embeddings
        self.agency_matrix = nn.Parameter(torch.randn(vocab_size, hidden_dim))

        # Vocabulary boosting scores for controlling agency in token selection
        self.boosting_weights = nn.Parameter(torch.ones(vocab_size))

        # Define agency lexicon，need to find a new database for these
        self.agency_lexicon = {
            "positive": ["achieve", "lead", "create", "invent", "command", "decide"],
            "neutral": ["walk", "observe", "stand", "exist", "see"],
            "negative": ["obey", "suffer", "hesitate", "submit", "fail"]
        }
    
    def mask_and_reconstruct(self, input_ids, tokenizer, target_agency="neutral"):
        """
        Masks agency-related verbs and reconstructs them using controlled agency.
        """
        modified_ids = input_ids.clone()

        # Get vocabulary index for replacement verbs
        replacement_words = self.agency_lexicon[target_agency]
        replacement_ids = [tokenizer.convert_tokens_to_ids(w) for w in replacement_words if tokenizer.convert_tokens_to_ids(w) is not None]

        if len(replacement_ids) == 0: #if no replacement, use original words
            return modified_ids

        for i in range(input_ids.shape[0]):
            for j in range(input_ids.shape[1]):
                token = tokenizer.decode([input_ids[i, j]]).strip().lower()
                
                # Check if token is a verb in any agency category
                for agency, verbs in self.agency_lexicon.items():
                    if token in verbs:
                        modified_ids[i, j] = np.random.choice(replacement_ids)  # Replace with controlled verb
                        break

        return modified_ids

    def forward(self, embeddings, token_ids):
        """
        Modifies embeddings using PowerTransformer principles.
        """
        # Compute boosting scores for each token
        boost_factors = self.boosting_weights[token_ids]  # Shape: (batch, seq_len)
        boost_factors = boost_factors.unsqueeze(-1)  # Shape: (batch, seq_len, 1)
        
        # Modify embeddings based on learned agency matrix
        modified_embeddings = embeddings + self.beta * (boost_factors * self.agency_matrix[token_ids])
        return modified_embeddings
