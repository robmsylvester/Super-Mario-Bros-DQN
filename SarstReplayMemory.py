#!/usr/bin/env
import numpy as np
from PrioritizedSumTree import PrioritizedSumTree

#TODO - implement Tiered memories similar to cache levels in a processor
#TODO - try out some of the smarter replay choosing methods on my scribbles of junk

class SarstReplayMemory:
    """This memory holds five numpy arrays, each of which store the state, action, reward, next_state, and is_terminal
    boolean for a series of observations in a reinforcement learner. It works by sampling a batch of corresponding indexes
    from each of the four (s,a,r,s,t) arrays. This sampling is done by using a prioritized sum tree, which will pick
    samples that have high rewards associated with them under the assumption that they are more important.
    """
    def __init__(self, capacity, state_size, prioritize=True, priority_epsilon=0.001, priority_alpha=0.5):
        """
        Args:
            Capacity - int - How many samples to hold in the array. Samples all exist for exactly the same amount of time
            State_Size - numpy array shape tuple -  specifying the dimensions of the input
            prioritize - Boolean, whether to use the prioritized sum tree or to just randomly pick
            priority epsilon - float - small number. the smoothing param to add to the reward priority calculation if priortize is true
            priority alpha - float - between 0-1, the exponent used in the priority calculatiopn if prioritize is true
        """
        #Some book-keeping so we know how big the memory is
        self.memory_capacity = capacity #max size
        self.memory_size = 0
        self.memory_pointer_index = 0
        self.prioritize = prioritize
        self.total_priority_sum = 0.
        
        if prioritize:
            self.priority_epsilon = priority_epsilon
            self.priority_alpha = priority_alpha
            self.priority_tree = PrioritizedSumTree(capacity)
        
        #Create the SARS memory
        new_state_size = [self.memory_capacity] + [s for s in state_size]
        self.states = np.zeros(shape=new_state_size, dtype=np.uint8) #colors have to be 0-255
        self.actions = np.zeros(shape=(self.memory_capacity), dtype=np.uint8) #can't have more than 256 actions
        self.rewards = np.zeros(shape=(self.memory_capacity), dtype=np.float32)
        self.next_states = np.zeros(shape=new_state_size, dtype=np.uint8)
        
        #We also should know if this next state is a terminal state because algorithms depend on this if/else check
        self.terminal_flags = np.zeros(shape=(self.memory_capacity), dtype=np.bool)

        
    def add_to_memory(self, state, action, reward, next_state, next_state_is_terminal):
        """
        Adds new value to s, a, r, s, and t arrays when new observation is made.
        Write all the information to the current pointer in memory
        """
        self.states[self.memory_pointer_index] = state
        self.actions[self.memory_pointer_index] = action
        self.rewards[self.memory_pointer_index] = reward
        self.next_states[self.memory_pointer_index] = next_state
        self.terminal_flags[self.memory_pointer_index] = next_state_is_terminal
        
        #if using a prioritized memory, need to add it the sample to that as well
        if self.prioritize:
                    
            #calculates a priority value for a given reward, and adds it to the tree. Note we NEED absolute value
            #for not just the correctness of the sum tree but also, intuitively, negative rewards are just as
            #important as positive ones
            sample_priority = (abs(reward) + self.priority_epsilon)**self.priority_alpha
            self.priority_tree.add(sample_priority)


        #we have to increment the memory pointer index so we write to the correct spot
        #note that the prioritized sum tree does this on its own, which is why we dont have it explicitly done above
        self.memory_pointer_index = (self.memory_pointer_index + 1) % self.memory_capacity
        
        #also update the memory size as a safety check against batches that are larger than the memory size
        self.memory_size = min(self.memory_size+1, self.memory_capacity)

        
    def get_batch_sample(self, batch_size):
        """
        Returns a numpy array of batch_size samples complete with s,a,r,s,t. This is to be fed into a neural network
        such that the network can compute a one-hot dot product with the action that it cares about to see how
        accurate the network is
        """
        
        if self.memory_size < batch_size:
            #TODO - might be a better idea to just keep sampling from them anyway and repeat samples
            raise ValueError, "Cannot read a batch of %d samples when memory only has %d samples stored" % (batch_size, self.memory_size)
        
        chosen_sarst_indexes = []
        
        #either use the prioritized sum tree to return samples based on their reward values so that more important ones
        #are chosen, or do it randomly
        if self.prioritize:
            while len(chosen_sarst_indexes) < batch_size:
                chosen_idx, relative_priority = self.priority_tree.get()
                #print("chose replay idx %d with priority %.4f" % (chosen_idx, relative_priority))
                
                #the tree starts at 0, therefore the leaves start at half of the tree size
                #we need to convert to the index of the sarst memory by subtracting the capcacity+1
                chosen_idx -= self.memory_capacity-1
                #print("this chosen reward should read %.4f\n" % (abs(self.rewards[chosen_idx]) + self.priority_epsilon)**self.priority_alpha)
                chosen_sarst_indexes.append(chosen_idx)
        else:
            #choose randomly
            while len(chosen_sarst_indexes) < batch_size:
                chosen_sarst_indexes.append(np.random.randint(low=0, high=self.memory_size))

        states = self.states[chosen_sarst_indexes]
        actions = self.actions[chosen_sarst_indexes]
        rewards = self.rewards[chosen_sarst_indexes]
        next_states = self.next_states[chosen_sarst_indexes]
        terminal_flags = self.terminal_flags[chosen_sarst_indexes]

        return states, actions, rewards, next_states, terminal_flags
   