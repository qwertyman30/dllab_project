import numpy as np
import torch


class Memory:
    """
    Memory storage for experience replay, as it is used in many popular
    off-policy algorithms.
    Using preallocated numpy arrays for storing the states, actions,
    rewards, next_states and final flags leads to huge performance improvements
    over standard python lists.
    """

    def __init__(self, state_dim, action_dim, max_size, history_length, device="cuda"):
        self.max_size = max_size
        self.states = np.zeros((max_size, state_dim), dtype=np.float32) \
            if history_length == 0 \
            else np.zeros((max_size, history_length + 1, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32) \
            if history_length == 0 \
            else np.zeros((max_size, history_length + 1, state_dim))
        self.final = np.zeros(max_size, dtype=np.float32)
        self.idx = 0
        self.size = 0
        self.device = torch.device(device)

    def add(self, state, action, reward, next_state, final):
        """
        Adds a new transition to the memory. Adds 1 to the current size of the memory.
        We have to use an index pointer here because we preallocate the
        numpy arrays in the init, so we cannot just append at the end like with
        regular python lists. We set the index to zero once we reached our max memory size
        so we always overwrite the oldest transition the next time we add a transition.
        """
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.final[self.idx] = final
        self.idx = 0 if self.idx >= self.max_size - 1 else self.idx + 1
        self.size = self.max_size if self.size >= self.max_size else self.size + 1

    def sample(self, batch_size):
        """
        Sample #batch_size transitions from the memory using a uniform distribution.
        For convenience we already transform the sampled transitions to torch tensors
        and load them to the cpu or gpu.
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        final = self.final[indices]
        return torch.tensor(states, dtype=torch.float32, requires_grad=False).to(self.device), \
               torch.tensor(actions, dtype=torch.float32,
                            requires_grad=False).to(self.device), \
               torch.tensor(rewards, dtype=torch.float32,
                            requires_grad=False).to(self.device), \
               torch.tensor(next_states, dtype=torch.float32,
                            requires_grad=False).to(self.device), \
               torch.tensor(final, dtype=torch.float32,
                            requires_grad=False).to(self.device)
