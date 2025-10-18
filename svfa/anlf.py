import torch
import torch.nn as nn
from mmfewshot.detection.models.utils.aggregation_layer import AGGREGATORS


@AGGREGATORS.register_module()
class Attention_min(nn.Module):
    def __init__(self, feature_dim):
        super(Attention_min, self).__init__()
        self.feature_dim = feature_dim

        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 1),  # Calculate attention score for each feature
            nn.Sigmoid()  # Scale scores to [0, 1]
        )

    def forward(self, feature1, feature2):

        # Calculate attention scores
        attention1 = self.attention(feature1)  # (batch_size, 1)
        attention2 = self.attention(feature2)  # (batch_size, 1)

        # Normalize weights
        alpha_sum = attention1 + attention2
        alpha1 = attention1 / alpha_sum
        alpha2 = attention2 / alpha_sum

        # Weighted fusion
        fused_feature = alpha1 * feature1 - alpha2 * feature2  # (batch_size, feature_dim)

        return fused_feature

@AGGREGATORS.register_module()
class Attention_mul(nn.Module):
    def __init__(self, feature_dim):
        super(Attention_mul, self).__init__()
        self.feature_dim = feature_dim

        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 1),  # Calculate attention score for each feature
            nn.Sigmoid()  # Scale scores to [0, 1]
        )

    def forward(self, feature1, feature2):

        # Calculate attention scores
        attention1 = self.attention(feature1)  # (batch_size, 1)
        attention2 = self.attention(feature2)  # (batch_size, 1)

        # Normalize weights
        alpha_sum = attention1 + attention2
        alpha1 = attention1 / alpha_sum
        alpha2 = attention2 / alpha_sum

        # Weighted fusion
        fused_feature = (alpha1 * feature1) * (alpha2 * feature2)  # (batch_size, feature_dim)

        return fused_feature
