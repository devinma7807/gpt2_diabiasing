import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import GPT2Tokenizer
import pandas as pd

agency_lexicon_data = pd.read_csv('./pt_agency_lexicons.csv')

class PowerTransformer(nn.Module):
    def __init__(self, hidden_dim, vocab_size, beta=5, activate=False):
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

        self.agency_lexicon ={
            "positive": list(agency_lexicon_data['positive'].dropna()),
            "neutral": list(agency_lexicon_data['neutral'].dropna()),
            "negative": list(agency_lexicon_data['negative'].dropna())
        } if activate else {
            "positive": [],
            "neutral": [],
            "negative": []
        } 

    def mask_and_reconstruct(self, input_ids, tokenizer, target_agency="neutral"):
        """
        Masks agency-related verbs and reconstructs them using controlled agency.
        """
        modified_ids = input_ids.clone()

        # Get vocabulary index for replacement verbs
        replacement_words = self.agency_lexicon[target_agency]
        eot = tokenizer.convert_tokens_to_ids('<|endoftext|>')
        replacement_ids = [tokenizer.convert_tokens_to_ids(w) for w in replacement_words if (tokenizer.convert_tokens_to_ids(w) is not None and tokenizer.convert_tokens_to_ids(w) != eot)]

        if len(replacement_ids) == 0:
            return modified_ids

        for i in range(input_ids.shape[0]):
            for j in range(input_ids.shape[1]):
                token = tokenizer.decode([input_ids[i, j]]).strip().lower()
                # Check if token is a verb in any agency category
                for agency, verbs in self.agency_lexicon.items():
                    if agency != target_agency and token in verbs:
                        modified_ids[i, j] = np.random.choice(replacement_ids)
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
