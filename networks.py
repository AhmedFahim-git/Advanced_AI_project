import torch
from torch import nn
import torch.nn.functional as F


class UserFeatureNetwork(nn.Module):
    def __init__(
        self,
    ):
        super(UserFeatureNetwork, self).__init__()
        self.gru = nn.GRU(29, 128, num_layers=2, batch_first=True, dropout=0.1)
        self.linear1 = nn.Linear(128, 128)
        self.linear2 = nn.Linear(128, 256)

    def forward(self, x):
        x, _ = self.gru(x)
        x = x[:, -1]
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


class ContentRatingNetwork(nn.Module):
    def __init__(self, movie_features_base):
        super(ContentRatingNetwork, self).__init__()
        self.usernet = UserFeatureNetwork()
        self.linear1 = nn.Linear(28 + 256, 128)
        self.linear2 = nn.Linear(128, 1)
        self.movie_feature = movie_features_base
        self.num_movies = movie_features_base.shape[1]

    def forward(self, x1):
        batch_size = x1.shape[0]
        x1 = self.usernet(x1).unsqueeze(1).expand(-1, self.num_movies, -1)
        x2 = self.movie_feature.expand(batch_size, -1, -1)
        x = torch.cat([x1, x2], dim=-1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
