import torch.nn.functional as F
import torch.nn as nn
import torch


class FixedOutputTransformerDecoder(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=1,
        dim_feedforward=256,
        num_queries=3503,
        num_decoder_layers=1,
        dropout=0.1,
    ):
        """
        Args:
            embed_dim (int): The embedding dimension (d_model).
            num_heads (int): Number of attention heads.
            dim_feedforward (int): Hidden dimension in the feedforward network.
            num_queries (int): Fixed number of queries (defines the fixed output length).
            num_decoder_layers (int): Number of decoder layers to stack.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.num_queries = num_queries

        # Learnable fixed queries (each query is a vector of dimension embed_dim)
        self.query_embed = nn.Embedding(num_queries, embed_dim)

        # Build a Transformer Decoder with the specified number of layers.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Final linear layer to map each query’s output embedding to a scalar.
        # If you need a different output dimension, change the second argument.
        self.fc_out = nn.Linear(embed_dim, 1)

    def forward(self, memory, memory_key_padding_mask=None):
        """
        Args:
            memory (Tensor): Input features from the encoder.
                Shape: (num_points, batch_size, embed_dim)
            memory_key_padding_mask (Tensor, optional): A bool mask indicating padded positions
                in the memory (shape: (batch_size, num_points)).
                True values indicate positions that should be ignored.

        Returns:
            Tensor: Final output of shape (batch_size, num_queries).
        """
        batch_size = memory.size(1)
        # Create fixed queries: shape (num_queries, batch_size, embed_dim)
        queries = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)

        # Pass the fixed queries and the variable-length memory through the decoder.
        # The decoder will attend over the memory (ignoring padded positions if a key padding mask is provided).
        dec_output = self.decoder(
            tgt=queries,       # fixed queries (target)
            memory=memory,     # variable-length input features (encoder output)
            memory_key_padding_mask=memory_key_padding_mask,
        )
        # dec_output shape: (num_queries, batch_size, embed_dim)

        # Map each query’s output embedding to a scalar.
        output = self.fc_out(dec_output)  # Shape: (num_queries, batch_size, 1)
        output = output.squeeze(-1)       # Shape: (num_queries, batch_size)

        # Transpose so that batch is the first dimension: (batch_size, num_queries)
        output = output.transpose(0, 1)

        return output
