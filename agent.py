import torch
from torch import nn
import numpy as np
from networks import ContentRatingNetwork
import torch.optim as optim
from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = nn.MSELoss()


def cql_loss(q_values, current_action):
    """Computes the CQL loss for a batch of Q-values and actions."""
    logsumexp = torch.logsumexp(q_values, dim=1, keepdim=True)
    q_a = q_values.gather(1, current_action)

    return (logsumexp - q_a).mean()


class CQLAgent:
    def __init__(self, movie_features_base):
        self.tau = 0.55
        self.gamma = 0.8
        self.num_epochs = 2
        self.test_update_steps = 20

        self.DQN = ContentRatingNetwork(movie_features_base).to(device=device)
        self.test_DQN = ContentRatingNetwork(movie_features_base)
        self.test_DQN.load_state_dict(self.DQN.state_dict())
        self.test_DQN = self.test_DQN.to(device=device)

        self.optimizer = optim.Adam(self.DQN.parameters(), lr=1e-2)

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device=device)
        self.DQN.eval()
        with torch.no_grad():
            action_values = self.DQN(state)
        self.DQN.train()
        action = np.argmax(action_values.cpu().data.numpy(), axis=1)
        return action

    def offline_learn(self, my_dataloader):
        for i in range(self.num_epochs):
            running_loss = 0.0
            last_loss = 0.0

            for j, data in enumerate(my_dataloader, 1):
                # Every data instance is an input + label pair
                state, action, reward, next_state = data

                with torch.no_grad():
                    Q_target_next = self.test_DQN(next_state.to(device=device)).max(1)[
                        0
                    ]
                    Q_targets = reward.unsqueeze(-1).to(device=device) + (
                        self.gamma * Q_target_next
                    )

                Q_s_a = self.DQN(state.to(device=device))
                Q_expected = Q_s_a.gather(
                    1, action.unsqueeze(-1).unsqueeze(-1).to(device=device)
                ).squeeze(-1)

                cql1_loss = cql_loss(
                    Q_s_a, action.unsqueeze(-1).unsqueeze(-1).to(device=device)
                )

                bellman_error = loss_fn(Q_expected, Q_targets)

                q1_loss = 0.5 * cql1_loss + bellman_error

                running_loss += q1_loss.item()
                if j % 100 == 0:
                    last_loss = running_loss / 100
                    print("  batch {} loss: {}".format(j, last_loss))
                    running_loss = 0.0

                self.optimizer.zero_grad()
                q1_loss.backward()
                torch.nn.utils.clip_grad_value_(self.DQN.parameters(), 100)
                self.optimizer.step()

                if j % self.test_update_steps == 0:
                    new_state_dict = OrderedDict()
                    for k in self.DQN.state_dict():
                        new_state_dict[k] = (
                            self.tau * self.DQN.state_dict()[k]
                            + (1 - self.tau) * self.test_DQN.state_dict()[k]
                        )
                    self.test_DQN.load_state_dict(new_state_dict)
                    self.test_DQN = self.test_DQN.to(device=device)
