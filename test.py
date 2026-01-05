import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

def compute_correlation(test_seq, train_seqs):
    """
    Compute cosine similarity between the test sequence and each training sequence.
    Args:
        test_seq: Tensor of shape (seq_len, feature_dim)
        train_seqs: Tensor of shape (num_train, seq_len, feature_dim)
    Returns:
        similarities: Tensor of shape (num_train,), cosine similarity scores
    """
    test_seq = test_seq.flatten()  # Flatten test sequence to 1D
    train_seqs = train_seqs.view(train_seqs.size(0), -1)  # Flatten training sequences
    similarities = F.cosine_similarity(test_seq.unsqueeze(0), train_seqs, dim=1)
    return similarities

def aggregate_top_k(train_seqs, similarities, k):
    """
    Select top-k sequences based on similarity and aggregate them using avgpool.
    Args:
        train_seqs: Tensor of shape (num_train, seq_len, feature_dim)
        similarities: Tensor of shape (num_train,), similarity scores
        k: int, number of top sequences to select
    Returns:
        aggregated_seq: Tensor of shape (seq_len, feature_dim), aggregated sequence
    """
    top_k_indices = torch.topk(similarities, k=k, largest=True).indices
    top_k_seqs = train_seqs[top_k_indices]  # Shape: (k, seq_len, feature_dim)
    aggregated_seq = top_k_seqs.mean(dim=0)  # Average over top-k sequences
    return aggregated_seq

# Example usage
if __name__ == "__main__":
    # Simulate training and test data
    num_train = 100
    seq_len = 10
    feature_dim = 16
    k = 5

    train_seqs = torch.rand((num_train, seq_len, feature_dim))  # Training sequences
    test_seq = torch.rand((seq_len, feature_dim))  # Test sequence

    # Compute similarities
    similarities = compute_correlation(test_seq, train_seqs)

    # Aggregate top-k sequences
    aggregated_seq = aggregate_top_k(train_seqs, similarities, k)

    # Define encoder
    input_dim = feature_dim
    hidden_dim = 32
    output_dim = 16
    encoder = Encoder(input_dim, hidden_dim, output_dim)

    # Encode the aggregated sequence
    aggregated_seq_flat = aggregated_seq.flatten()  # Flatten to feed into encoder
    encoded_seq = encoder(aggregated_seq_flat)

    print("Encoded sequence:", encoded_seq)