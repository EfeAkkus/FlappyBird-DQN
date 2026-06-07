# # Define memory for Experience Replay
# from collections import deque
# import random


# class ReplayMemory:
    
#     def __init__(self, maxlen, seed=None):
#         self.memory = deque([], maxlen=maxlen)

#         # Optional seed for reproducibility
#         if seed is not None:
#             random.seed(seed)

#     def append(self, transition):
#         self.memory.append(transition)

#     def sample(self, sample_size):
#         return random.sample(self.memory, sample_size)

#     def __len__(self):
#         return len(self.memory)

from collections import deque
import random

class ReplayMemory():
    """
    Experience Replay buffer to store and sample transitions for DQN training.
    Breaks the correlation between consecutive samples, stabilizing the network.
    """
    def __init__(self, maxlen, seed=None):
        self.memory = deque([], maxlen=maxlen)

        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        """
        Stores a transition (state, action, next_state, reward, done).
        """
        self.memory.append(transition)

    def sample(self, sample_size):
        """
        Randomly samples a batch of transitions from memory.
        """
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)