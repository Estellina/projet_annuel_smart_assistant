import torch
import torch.nn as nn
import re
import numpy as np
from tqdm import tqdm
from tokenizers import ByteLevelBPETokenizer

import torch
import torch.nn as nn
import math
from torch.utils.checkpoint import checkpoint

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.max_len = max_len
        # Create positional encoding buffer
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Dynamically handle sequences longer than max_len
        seq_len = x.size(1)
        if seq_len > self.max_len:
            # Truncate positional encoding if needed
            pe = self.pe[:, :seq_len]
        else:
            pe = self.pe[:, :self.max_len]

        x = x + pe[:, :x.size(1)]  # Only add positional encoding up to sequence length
        return self.dropout(x)


class TransformerSummarizer(nn.Module):
    def __init__(self, vocab_size, pad_id, d_model=768, nhead=12, num_layers=6,
                 dropout=0.2, use_checkpointing=False):
        super().__init__()
        self.pad_id = pad_id
        self.use_checkpointing = use_checkpointing

        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer models
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.output_layer = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def _run_encoder(self, src, src_key_padding_mask):
        return self.transformer.encoder(
            src,
            mask=None,
            src_key_padding_mask=src_key_padding_mask
        )

    def _run_decoder(self, tgt, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask):
        return self.transformer.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=None,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

    def forward(self, src_ids, tgt_ids, src_mask, tgt_mask):
        # Embeddings with positional encoding
        src = self.pos_encoder(self.embedding(src_ids))
        tgt = self.pos_encoder(self.embedding(tgt_ids))

        # Create decoder attention mask
        tgt_seq_len = tgt.size(1)
        tgt_attn_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)

        # Convert mask to boolean (True = ignore)
        src_key_padding_mask = (src_mask == 0)
        tgt_key_padding_mask = (tgt_mask == 0)

        # Run transformer with optional checkpointing
        if self.use_checkpointing and self.training:
            # Checkpoint encoder and decoder separately
            memory = checkpoint(
                self._run_encoder,
                src,
                src_key_padding_mask
            )
            output = checkpoint(
                self._run_decoder,
                tgt,
                memory,
                tgt_attn_mask,
                tgt_key_padding_mask,
                src_key_padding_mask
            )
        else:
            # Standard forward pass
            memory = self._run_encoder(src, src_key_padding_mask)
            output = self._run_decoder(
                tgt,
                memory,
                tgt_attn_mask,
                tgt_key_padding_mask,
                src_key_padding_mask
            )

        return self.output_layer(output)

    def generate_high_quality_summary(self, tokenizer, text: str, max_length: int = 100,
                                      max_source_length: int = 2048) -> str:
        """
        Enhanced summary generation with:
        - Key sentence extraction (TextRank)
        - Nucleus sampling decoding
        - Repetition prevention
        - Advanced post-processing
        """

        # 1. Enhanced Preprocessing
        def clean_text_for_generation(text):
            # Fix hyphenated words and line breaks
            text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
            text = re.sub(r'\n', ' ', text)

            # Remove citations and figure references
            text = re.sub(r'\([A-Za-z0-9,\s]+\s\d{4}\)', '', text)
            text = re.sub(r'(Figure|Table|Fig\.|Tab\.)\s?\d+', '', text)

            # Remove URLs and special characters
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            text = re.sub(r'[^\w\s.,;:?!\'"-]', '', text)

            # Reduce whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        # 2. Extract Key Sentences (Simple TextRank)
        def extract_key_sentences(text, num_sentences=5):
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
            word_freq = {}
            for sentence in sentences:
                for word in sentence.split():
                    word_freq[word] = word_freq.get(word, 0) + 1

            sentence_scores = {}
            for sentence in sentences:
                for word in sentence.split():
                    if word in word_freq:
                        sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_freq[word]
                sentence_scores[sentence] /= len(sentence.split())

            top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
            return " ".join([s[0] for s in top_sentences])

        # Preprocess and extract key content
        clean_text = clean_text_for_generation(text)
        key_text = extract_key_sentences(clean_text)

        # 3. Tokenize with truncation
        encoding = tokenizer.encode(key_text)
        source_tokens = encoding.ids[:max_source_length]
        source_ids = torch.tensor([source_tokens], dtype=torch.long, device=self.device)
        source_mask = torch.ones_like(source_ids, device=self.device)

        # 4. Nucleus Sampling (Top-p) Decoding
        def nucleus_sampling(logits, top_p=0.92, temperature=0.85):
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Remove tokens outside the nucleus
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')
            return torch.multinomial(torch.softmax(logits, dim=-1), 1).item()

        # 5. Generate with repetition prevention
        bos_id = tokenizer.token_to_id("<s>")
        eos_id = tokenizer.token_to_id("</s>")
        target_ids = torch.tensor([[bos_id]], dtype=torch.long, device=self.device)
        generated_tokens = []
        repetition_penalty = 1.5
        min_new_tokens = 20  # Minimum tokens before allowing EOS

        with torch.no_grad():
            for step in range(max_length):
                output = self(
                    source_ids,
                    target_ids,
                    source_mask,
                    torch.ones_like(target_ids))

                next_token_logits = output[0, -1, :]

                # Apply repetition penalty
                for token_id in set(generated_tokens[-10:]):  # Look at recent tokens
                    if next_token_logits[token_id] > 0:
                        next_token_logits[token_id] /= repetition_penalty

                # Generate next token with nucleus sampling
                next_token = nucleus_sampling(next_token_logits)

                # Prevent early stopping
                if next_token == eos_id and step < min_new_tokens:
                    next_token = torch.argmax(next_token_logits).item()

                # Stop at EOS
                if next_token == eos_id:
                    break

                # Update sequences
                generated_tokens.append(next_token)
                new_token = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
                target_ids = torch.cat([target_ids, new_token], dim=1)

        # 6. Post-processing
        summary_tokens = target_ids[0].tolist()
        summary_text = tokenizer.decode(summary_tokens)

        # Remove any remaining special tokens
        summary_text = re.sub(r'\[CITE\]|\[MATH\]|\[EQUATION\]', '', summary_text)

        # Remove repeated phrases
        sentences = summary_text.split('. ')
        unique_sentences = []
        for sentence in sentences:
            if sentence not in unique_sentences:
                unique_sentences.append(sentence)

        # Capitalize and format
        clean_summary = '. '.join(unique_sentences).strip()
        if clean_summary and clean_summary[-1] not in {'.', '!', '?'}:
            clean_summary += '.'
        if clean_summary and clean_summary[0].islower():
            clean_summary = clean_summary[0].upper() + clean_summary[1:]

        return clean_summary


def load_model(model_path: str, tokenizer_path: str, device: str = "auto"):
    """Load models with enhanced summary generation"""
    tokenizer = ByteLevelBPETokenizer(
        f"{tokenizer_path}/vocab.json",
        f"{tokenizer_path}/merges.txt"
    )
    tokenizer.add_special_tokens(["[CITE]", "[MATH]", "[EQUATION]"])

    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("<pad>")

    # Determine device
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
    else:
        device = "cpu"


    model = TransformerSummarizer(
        vocab_size,
        pad_id=pad_id,
        d_model=768,
        nhead=12,
        num_layers=6,
        dropout=0.0  # Disable dropout for inference
    )

    # Load weights
    checkpoint = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    state_dict = checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'models' in checkpoint:
        state_dict = checkpoint['models']

    # Remove 'module.' prefix if present
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    model.device = device  # Store device reference

    print(f"Model loaded on {device} from {model_path}")
    return model, tokenizer