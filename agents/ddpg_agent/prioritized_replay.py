import torch
import numpy as np
import random
from collections import namedtuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SumTree(object):
    '''
    SumTree for efficiently performing weighted sampling. 

    Adapted from https://pylessons.com/CartPole-PER/
    '''

    # Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    def __init__(self, capacity):
        # Number of leaf nodes (final nodes) that contains experiences
        self.capacity = capacity

        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema below
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)

        self.data_pointer = 0  # Pointer to the next leave to update.

        # Contains the experiences (so the size of data is capacity)
        self.data = [None]*capacity

    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + (self.capacity - 1)

        """ tree:
                    0
                   / \
                  0   0
                 / \ / \
        tree_index  0 0  0  We fill the leaves from left to right
        """

        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update(tree_index, priority)

        # Add 1 to data_pointer
        self.data_pointer += 1

        # If we're above the capacity, we go back to first index (we overwrite)
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        # this method is faster than the recursive loop
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]  # Returns the root node

    @property
    def maximum_priority(self):
        return np.max(self.tree[-self.capacity:])  # Returns the root node

    def __len__(self):
        """Return the current size of internal memory."""
        return np.sum(~(self.tree[-self.capacity:] == 0))


class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples.
       Leverages a SumTree for efficiently sampling."""

    def __init__(self,
                 buffer_size,
                 batch_size,
                 seed,
                 per_epsilon: float = None,
                 per_alpha: float = None,):
        """Initialize a PrioritizedReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.tree = SumTree(buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.per_epsilon = per_epsilon or 0.01
        self.per_alpha = per_alpha or 0

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        maximum_priority = self.tree.maximum_priority + \
            self.per_epsilon  # TODO use clipped abs error?
        if maximum_priority == 0:
            maximum_priority = 1
        self.tree.add(maximum_priority, e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""

        experiences = []
        indices = []
        priorities = []
        # We divide the priority into buckets and sample from each of those
        segments = self.tree.total_priority/self.batch_size
        values = []
        for i in range(self.batch_size):
            value = random.uniform(i*segments, (i+1)*segments)
            leaf_index, priority, data = self.tree.get_leaf(value)

            experiences.append(data)
            indices.append(leaf_index)
            priorities.append(priority)
            values.append(value)

        try:
            states = torch.from_numpy(
                np.vstack([e.state for e in experiences if e is not None])).float().to(device)
            actions = torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])).float().to(device)
            rewards = torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
            next_states = torch.from_numpy(np.vstack(
                [e.next_state for e in experiences if e is not None])).float().to(device)
            dones = torch.from_numpy(np.vstack(
                [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        except:
            import pdb;pdb.set_trace()
        return indices, torch.Tensor(priorities), (states, actions, rewards, next_states, dones)

    def update_batches(self, indices, errors):

        for index, error in zip(indices, errors.detach().numpy()):
            self.tree.update(
                index, (abs(error)+self.per_epsilon)**self.per_alpha)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.tree)
