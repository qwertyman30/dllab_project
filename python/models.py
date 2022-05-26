import torch
from torch import nn
from torch.nn import functional as F


class Actor(nn.Module):
    """
    Basic dense actor network.
    The number of hidden units [400, 300] and
    activation functions [relu, relu, tanh] were chosen
    according to the TD3 paper.
    """

    def __init__(self, state_dim, action_dim, history_length):
        super(Actor, self).__init__()
        self.history_length = history_length
        self.linear1 = nn.Linear(state_dim * (history_length + 1), 400)
        self.linear2 = nn.Linear(400, 300)
        self.linear3 = nn.Linear(300, action_dim)

    def forward(self, state):
        """
        Takes in a state and returns the action.
        Activation function before the ouput is tanh
        because the continuous action space goes from -1 to 1
        """
        batch_size = state.size(0)
        state = state.view(batch_size, -1)
        a = F.relu(self.linear1(state))
        a = F.relu(self.linear2(a))
        a = torch.tanh(self.linear3(a))
        return a


class CNNActor(nn.Module):
    def __init__(self, state_dim, action_dim, history_length):
        super(CNNActor, self).__init__()
        self.history_length = history_length
        self.conv1 = nn.Conv1d(history_length + 1, 16, kernel_size=3, stride=2, padding=3 // 2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=3 // 2)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, padding=3 // 2)
        self.linear = nn.Linear(32 * state_dim // 2, action_dim)

    def forward(self, state):
        batch_size = state.size(0)
        state = state.view(batch_size, self.history_length + 1, -1)
        a = F.relu(self.conv1(state))
        a = F.relu(self.conv2(a))
        a = F.relu(self.conv3(a))
        a = torch.tanh(self.linear(a.view(batch_size, -1)))
        return a


class DoubleCritic(nn.Module):
    """
    Convenience class for TD3's twin critic networks,
    wraps two seperate critics in one class
    """

    def __init__(self, state_dim, action_dim, history_length):
        super(DoubleCritic, self).__init__()
        self.history_length = history_length
        self.linear1 = nn.Linear(state_dim * (history_length + 1) + action_dim, 400)
        self.linear2 = nn.Linear(action_dim + 400, 300)
        self.linear3 = nn.Linear(300, 1)

        self.linear4 = nn.Linear(state_dim * (history_length + 1) + action_dim, 400)
        self.linear5 = nn.Linear(action_dim + 400, 300)
        self.linear6 = nn.Linear(300, 1)

    def c1(self, state, action):
        batch_size = state.size(0)
        state = state.view(batch_size, -1)
        q = F.relu(self.linear1(torch.cat([state, action], 1)))
        q = F.relu(self.linear2(torch.cat([q, action], 1)))
        q = self.linear3(q)
        return q

    def c2(self, state, action):
        batch_size = state.size(0)
        state = state.view(batch_size, -1)
        q = F.relu(self.linear4(torch.cat([state, action], 1)))
        q = F.relu(self.linear5(torch.cat([q, action], 1)))
        q = self.linear6(q)
        return q

    def forward(self, state, action):
        """
        Takes in a state and an action and
        estimates the corrpesonding q value
        twice with each of the critics.
        """
        q1 = self.c1(state, action)
        q2 = self.c2(state, action)
        return q1, q2


class SingleCritic(nn.Module):
    """
    Convenience class for TD3's twin critic networks,
    wraps two seperate critics in one class
    """

    def __init__(self, state_dim, action_dim, history_length):
        super(SingleCritic, self).__init__()
        self.history_length = history_length
        self.linear1 = nn.Linear(state_dim * (history_length + 1) + action_dim, 400)
        self.linear2 = nn.Linear(action_dim + 400, 300)
        self.linear3 = nn.Linear(300, 1)

    def c1(self, state, action):
        batch_size = state.size(0)
        state = state.view(batch_size, -1)
        q = F.relu(self.linear1(torch.cat([state, action], 1)))
        q = F.relu(self.linear2(torch.cat([q, action], 1)))
        q = self.linear3(q)
        return q

    def forward(self, state, action):
        """
        Takes in a state and an action and
        estimates the corrpesonding q value
        twice with each of the critics.
        """
        q1 = self.c1(state, action)
        return q1


class CNNDoubleCritic(nn.Module):
    def __init__(self, state_dim, action_dim, history_length):
        super(CNNDoubleCritic, self).__init__()
        self.history_length = history_length
        self.conv1 = nn.Conv1d(history_length + 1, 16, kernel_size=3, stride=2, padding=3 // 2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=3 // 2)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, padding=3 // 2)
        self.linear1 = nn.Linear(32 * state_dim // 2 + 300, action_dim)
        self.linear2 = nn.Linear(action_dim, 300)

        self.conv4 = nn.Conv1d(history_length + 1, 16, kernel_size=3, stride=2, padding=3 // 2)
        self.conv5 = nn.Conv1d(16, 32, kernel_size=3, padding=3 // 2)
        self.conv6 = nn.Conv1d(32, 32, kernel_size=3, padding=3 // 2)
        self.linear3 = nn.Linear(32 * state_dim // 2 + 300, action_dim)
        self.linear4 = nn.Linear(action_dim, 300)

    def c1(self, state, action):
        batch_size = state.size(0)
        state = state.view(batch_size, self.history_length + 1, -1)
        q = F.relu(self.conv1(state))
        q = F.relu(self.conv2(q))
        q = F.relu(self.conv3(q))
        q_a = F.relu(self.linear2(action))
        q = self.linear1(torch.cat([q.view(batch_size, -1), q_a], 1))
        return q

    def c2(self, state, action):
        batch_size = state.size(0)
        state = state.view(batch_size, self.history_length + 1, -1)
        q = F.relu(self.conv4(state))
        q = F.relu(self.conv5(q))
        q = F.relu(self.conv6(q))
        q_a = F.relu(self.linear4(action))
        q = self.linear3(torch.cat([q.view(batch_size, -1), q_a], 1))
        return q

    def forward(self, state, action):
        q1 = self.c1(state, action)
        q2 = self.c2(state, action)
        return q1, q2
