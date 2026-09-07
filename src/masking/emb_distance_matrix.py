from numpy.typing import NDArray
import torch
from transformers import T5Tokenizer, T5EncoderModel
import numpy as np
from scipy.spatial.distance import pdist, squareform

def embed_sequence_t5_perresidue(sequence, tokenizer: T5Tokenizer, encoder: T5EncoderModel, device) -> torch.Tensor:
    # spaces between needed for tokenizer
    seq_spaced = " ".join(list(sequence))
    # tokenize
    tokenized = tokenizer(
        seq_spaced,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    # Get embeddings
    with torch.no_grad():
        output = encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = output.last_hidden_state  # (1, seq_len, hidden_dim)
    # Remove batch dimension
    embeddings = embeddings.squeeze(0)  # (seq_len, hidden_dim)
    # need to exclude the end of sequence token and any padding
    seq_len = len(sequence)
    embeddings = embeddings[:seq_len]  # (seq_len, hidden_dim)

    return embeddings.float()  # (seq_len, hidden_dim)

def build_distance_matrix(embeddings) -> np.ndarray:
    """
    returns symmetric (seq_len, seq_len) float32 distance matrix
    embeddings: shape (seq_len, dim)
    """
    arr: NDArray = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2:
      raise ValueError(f"expected 2D (seq_len, dim), got shape {arr.shape}")
    comp_distmatrix = pdist(X=arr, metric="cosine") # compressed dist matrix, cosine for angle instead of distance
    return squareform(comp_distmatrix).astype(np.float32) # return full dist matrix

def top_k_neighbors(distance_matrix: np.ndarray, k: int) -> list[dict]:
    """
    For each position, return the kNN excluding self
    """
    dm = distance_matrix.copy()
    # seq_len = dm.shape[0]
    # k = min(k, seq_len - 1)
    np.fill_diagonal(dm, np.inf) # exclude self 0->inf for diag

    return [
        {
            "position": i,
            "knn": [
                {
                    "position": int(j),
                    "dist": float(row[j])
                }
                for j in np.argsort(row)[:k] # first k in sorted row are nearest
            ],
        }
        for i, row in enumerate(dm)
    ]

if __name__ == "__main__":
    MODEL_NAME = "Rostlab/prot_t5_xl_half_uniref50-enc"
    DEVICE = "cuda"
    seq = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    TOKENIZER = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False, use_fast=False)
    ENCODER = T5EncoderModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    ENCODER.to(DEVICE)
    ENCODER.eval()

    result: torch.Tensor = embed_sequence_t5_perresidue(seq, TOKENIZER, ENCODER, DEVICE)
    embs: np.ndarray = result.cpu().numpy() # torch tensor
    distance_matrix: np.ndarray = build_distance_matrix(embs)
    topk = top_k_neighbors(distance_matrix, 5)
    print(topk)
