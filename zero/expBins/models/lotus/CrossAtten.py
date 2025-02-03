import torch.nn.functional as F
import torch.nn as nn
import torch


class FeatureAggregate(nn.Module):
    def __init__(self, num_queries, feature_dim):
        """
        Args:
            num_queries (int): Number of fixed queries (K). This will determine how many feature vectors you output.
            feature_dim (int): Dimension of the per-point features (D).
        """
        super(FeatureAggregate, self).__init__()
        self.num_queries = num_queries
        self.feature_dim = feature_dim
        # Initialize K learnable query vectors of dimension D.
        self.queries = nn.Parameter(torch.randn(num_queries, feature_dim), requires_grad=True)

        # 拿到了query之后，再走一个mlp
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.LeakyReLU(0.02),
            nn.Linear(feature_dim * 2, feature_dim),

        )

    def forward(self, features):
        """
        Args:
            features (Tensor): Per-point features of shape (N, D)
        Returns:
            fixed_features (Tensor): Aggregated fixed features of shape (K, D)
        """
        N, D = features.shape
        if D != self.feature_dim:
            raise ValueError(f"Expected feature dimension {self.feature_dim} but got {D}")

        # Use queries directly (shape: (K, D))
        queries = self.queries  # shape: (K, D)

        # Compute the scaled dot-product between queries and features.
        # Transpose features to shape (D, N) so that the dot product is computed properly.
        scores = torch.mm(queries, features.t())  # Shape: (K, N)
        scores = scores / (self.feature_dim ** 0.5)  # Scale the scores

        # Compute attention weights with softmax along the N dimension.
        attn_weights = F.softmax(scores, dim=-1)  # Shape: (K, N)

        # Weighted sum of per-point features to get fixed features.
        fixed_features = torch.mm(attn_weights, features)  # Shape: (K, D)

        fixed_features = self.mlp(fixed_features)

        return fixed_features


# test_point_features = torch.randn(2, 100, 128)
# aggregator = FixedQueryAggregator(num_queries=5, feature_dim=128)
# fixed_features = aggregator(test_point_features)
# print(fixed_features.shape)  # Expected output: torch.Size([2, 5, 128])
