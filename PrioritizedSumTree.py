import numpy as np

class PrioritizedSumTree:
    """
    In a SARST replay memory, there should be some what of choosing better examples from memory than
    by doing so randomly. One way to do this is by using a Prioritized Experience Replay.
    
    In the prioritized summation tree, we efficiently store an O(log n) search mechanism that can be
    used to retrieve an example given a valid probability distribution. The sum tree stores the priority
    values for each of the S-A-R-S'-T indices in the array. At the root, there is the sum of all priorities,
    and the leaves are the priorities of the individual SARST additions. This way, when a new element is added
    to the replay memory, it can be added at the specified leaf index and have its sums propagated up the tree
    as an O(log N) operation. It also means that when a value is randomly sampled between 0 and the sum of the
    priorities (the root node), we can trace down the tree and will get to the corresponding leaf with the correct
    probability, that is to say, the value of the leaf divided by the total value of all leaves (the root node).
    
    This is done by recursively searching through the children of a node until reaching a leaf. Notice the recursive
    _get method and how it chooses to go left or right.
    
    Also notice that we need to keep track of a pointer of where we are in the sum tree so that each leaf has a life
    just as long as any other leaf.
    
    
    """
    def __init__(self, size):
        #Creates a tree with 2*size-1 total nodes, and size leaves nodes
        self.size = size
        
        #a binary tree of with n leaf nodes will have 2n-1 total nodes
        self.tree = np.zeros(2*size-1, dtype=np.float32)
        
        #we need a pointer to store which values need to be overwritten so they go in order
        self.pointer_idx = 0
    
    def sum_priorities(self):
        return self.tree[0]

    def _propagate_sums(self, tree_idx, val_to_add):
        
        #get the parent of the current node, and add the new value
        parent_idx = (tree_idx-1) // 2
        self.tree[parent_idx] += val_to_add

        #keep adding up the tree until reaching the root
        if parent_idx != 0:
            self._propagate_sums(parent_idx, val_to_add)

    def _get(self, tree_idx, search):
        
        #index the children of the index in question
        left_child_idx = 2*tree_idx + 1
        right_child_idx = 2*tree_idx + 2

        #don't search further, found it
        if left_child_idx >= self.tree.shape[0]:
            return tree_idx

        if search <= self.tree[left_child_idx]:
            return self._get(left_child_idx, search)
        else:
            return self._get(right_child_idx, search-self.tree[left_child_idx])
       
    def get(self):
        #get a random number between 0 and the total sum of the priorities
        r = np.random.uniform(low=0., high=self.sum_priorities())
        #print("get randomly chose %.4f" % r)
        leaf_idx = self._get(0, r)

        return leaf_idx, self.tree[leaf_idx]

    def _update_pointer_idx(self):
        #increment the current pointer that we use to add values to the tree
        self.pointer_idx += 1
        if self.pointer_idx >= self.size:
            self.pointer_idx = 0
    
    def add(self, priority_val):
        #adds a value to the tree, and updates the pointer index
        add_idx = self.pointer_idx + self.size - 1

        self.update_tree(add_idx, priority_val)

        self._update_pointer_idx()

    def update_tree(self, idx, priority_val):
        #updates the tree sums by finding the new value to add, and then propagates this upward
        to_add = priority_val - self.tree[idx]

        self.tree[idx] = priority_val
        self._propagate_sums(idx, to_add)