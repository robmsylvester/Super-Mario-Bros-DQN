
# coding: utf-8

import numpy as np
import tensorflow as tf
import gym
from collections import deque
import matplotlib.pyplot as plot
import tensorflow as tf
import ops
from SarstReplayMemory import SarstReplayMemory

#TODO - add more rewards to py file
#TODO - add frequency visualization to replay memory priority selections
#TODO - add more frames to state

class MarioDQNAgent():
    
    def __init__(self,
                 gym_environment,
                 tf_session,
                 gamma,
                 state_size,
                 actions,
                 custom_reward_dict={},
                 memory_capacity=10000,
                 num_episodes=10,
                 batch_size=32,
                 report_frequency=100):
       
        self.env = gym_environment #a gym environment
        self.session = tf_session # a tensorflow session
        self.state_size = state_size #size of environment state space.
        self.actions = actions #a list of list of NES buttons, each of which is binary and of length 6
        self.num_actions = len(actions) #actions available at any state.
        self.num_episodes = num_episodes #how many levels to attempt, either dying or beating them.
        self.batch_size = batch_size #when updating deep q network, how many sarst samples to randomly pull
        self.gamma = gamma
        self.target_network_update_frequency = 2000 #how many minibatch steps before we set the target network weights to the prediction network weights
        self.cur_state = None
        
        #The Deep Network will have a prediction network and a target network.
        self.network_inputs = {}
        self.minibatches_run = 0
        
        #The size of the SarstReplayMemory class
        self.memory_capacity = memory_capacity #as per the dqn network paper, go with 10k

        #All learning rate configuration goes here
        self.learning_rate_init = 0.0003 #0.00025 used in atari paper
        self.learning_rate_decay = 0.99999
        self.learning_rate_decay_steps = 5

        #and some exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.015
        self.epsilon_decay = 0.9999
        
        #we wont clip the loss, but we will clip gradients themselves.
        self.clip_gradients_enabled = False
        self.min_gradient = -10.
        self.max_gradient = 10. #gradients of more than this will be clipped to this maximum in optimizer's minimize() call
        
        #use dropout on the weights
        self.dropout_keep_probability = 1.0 #layer 1 and 2 outputs pass through dropout
        
        valid_custom_reward_keys = ['life', 'coins', 'time', 'player_status', 'score']
        for key in custom_reward_dict.keys():
            assert key in valid_custom_reward_keys, "Custom reward dictionary requires you define keys that are defined in the nes info dict returned by evn.step(). These are defined in the nes_env package available at https://github.com/ppaquette/gym-super-mario/tree/master/ppaquette_gym_super_mario"
        self.custom_reward_dict = custom_reward_dict
        
        #These will hold episode reward information
        self.episode_count = 0
        self.total_episode_reward = 0.
        self.episode_rewards = np.zeros(shape=(self.num_episodes), dtype=np.float32)
        self.total_iterations = 0
        self.episode_iterations = 0
        
        #openAI tests convergence by examining rewards per 100 episodes, so we should try this as well.
        self.last_100_episode_rewards = deque([])
        self.all_episode_rewards = []
        
        #How often we get an update printed to screen about how things are looking and what param values are
        self.report_frequency = 100
        
        #instantiate a new blank replay memory to store state, action, reward, new_state, new_state_is_terminal arrays
        self.replay_memory = SarstReplayMemory(self.memory_capacity,
                                              self.state_size)
        
        print("Initialized SARST Replay Memory")
        
        #Let's also make our agent a prediction network, and a target network
        #construct the deep q network, which initializes all the placeholders for all variables in the tf graph
        self.deep_q_network()
    
    def deep_q_network(self):
        
        #Construct the two networks with identical architectures
        self.build_network('prediction_network')
        self.build_network('target_network', prediction_network=False)
        
        #Create tensorflow placeholder variables and map functions that we can call in the session
        #to easily copy over the prediction network parameters to the target network parameters
        self.build_network_copier()
        
        #Build global step decayer for the learning rate exponential decay function
        with tf.variable_scope('global_step'):
            self.global_step = tf.placeholder(tf.int32, name="global_step")
        
        #create the optimizer in the model
        self.run_optimizer()
        
        #initialize all these variables, mostly with xavier initializers
        init_op = tf.global_variables_initializer()
        
        #Ready to train
        self.session.run(init_op)
        
        print("Initialized Deep Q Network")
    
    
    #This function will be used to build both the prediction network as well as the target network
    def build_network(self, scope_name, prediction_network=True):
        
        net_shape = [None] + [s for s in self.state_size]
        print net_shape
        
        with tf.variable_scope(scope_name):
            self.network_inputs[scope_name] = tf.placeholder(tf.float32, shape=net_shape, name=scope_name+"_inputs")

            #The first layer is an 7x7 convolutional_layer with stride 2, ReLU
            #This reduces 256x224x1 to 128x112x8
            conv1 = ops.conv(self.network_inputs[scope_name],
                            8,
                            kernel=[7,7],
                            strides=[2,2],
                            w_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            name="conv1")
            conv1 = tf.nn.relu(conv1)
            
            #Then a 3x3 max pool stride 2, reduces to 64x56x8
            conv1_pool = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
            
            #Then a 5x5 convolutional layer with stride 2 to reduce to 32x28x16
            conv2 = ops.conv(conv1_pool,
                            16,
                            kernel=[5,5],
                            strides=[2,2],
                            w_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            name="conv2")
            conv2 = tf.nn.relu(conv2)
            
            #Then a 3x3 max pool stride 2, reduces to 16x14x16
            conv2_pool = tf.nn.max_pool(conv2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
            
            #Then a 3x3 convolution stride 2, reduces to 8x7x32. no pool after this
            conv3 = ops.conv(conv2_pool,
                            32,
                            kernel=[3,3],
                            strides=[2,2],
                            w_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            name="conv3")
            conv3 = tf.nn.relu(conv3)
            
            #Hook this to a fully-connected layer
            conv3 = tf.contrib.layers.flatten(conv3)
            with tf.variable_scope('fc1') as scope:
                w1 = tf.get_variable('fcw1', shape=[conv3.get_shape()[1], 256],
                                                   initializer=tf.contrib.layers.xavier_initializer())
                b1 = tf.get_variable('fcb1', shape=[256],
                                                    initializer=tf.constant_initializer(0.0))
                fc1_out = tf.matmul(conv3, w1) + b1
                fc1_out = tf.nn.relu(fc1_out)
                fc1_out = tf.nn.dropout(fc1_out, keep_prob=self.dropout_keep_probability)
            
            #Hook this to a final fully-connected layer
            with tf.variable_scope('fc2') as scope:
                w2 = tf.get_variable('fcw2', shape=[256, self.num_actions],
                                                   initializer=tf.contrib.layers.xavier_initializer())
                b2 = tf.get_variable('fcb2', shape=[self.num_actions],
                                                    initializer=tf.constant_initializer(0.0))
                if prediction_network:
                    self.q_predictions = tf.matmul(fc1_out, w2) + b2
                    self.max_predict_q_action = tf.argmax(self.q_predictions, axis=1)
                else:
                    self.q_targets = tf.matmul(fc1_out, w2) + b2

    
    def build_network_copier(self):
        
        #Tensorflow needs to create copy operations for parameters for which we can call eval()
        #When we call eval, we can pass a feed dictionary of the copied parameters
        with tf.variable_scope('copy_weights'):
            self.copied_parameters = {}
            self.copy_parameters_operation = {}

        for idx, (predict_parameter, target_parameter) in enumerate(zip(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='prediction_network'),
                                                                      tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_network'))):
            
            #sanity check that parameters are the same shape
            assert predict_parameter.get_shape().as_list() == target_parameter.get_shape().as_list(), "Networks parameters must be the same shape"
            
            input_shape = predict_parameter.get_shape().as_list()
            self.copied_parameters[predict_parameter.name] = tf.placeholder(tf.float32, shape=input_shape, name="copier_%d"%idx)
            self.copy_parameters_operation[predict_parameter.name] = target_parameter.assign(self.copied_parameters[predict_parameter.name])
              
    def restore_checkpoint(self):
        pass
    
   
        
    def copy_prediction_parameters_to_target_network(self):
        #print("\nCopying prediction network parameters to target network")
        for param in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='prediction_network'):
            self.copy_parameters_operation[param.name].eval({ self.copied_parameters[param.name] : param.eval() })
        
    
    def choose_action(self, state):
        
        def _choose_to_explore():
            #returns true with a probability equal to that of epsilon
            return True if np.random.rand() < self.epsilon else False
        
        def _choose_random_action():
            r = np.random.randint(low=0, high=self.num_actions) #high is 1 above what can be picked
            return self.actions[r]
        
        if _choose_to_explore():
            return _choose_random_action()
        else:
            #returns a tensor with the single best q action evaluated from prediction network
            #[0] returns this single value from the tensor
            a = self.max_predict_q_action.eval({self.network_inputs['prediction_network'] : [state]})[0]
            return self.actions[a]
    
    def clip_gradients(self, gradient):
        # Clip gradients to min_grad and max_grad before.
        # Used before optimizer.minimize() function
        # 
        # Parameters
        # gradient (int) - single value from gradient list returned from a tuple of (gradient, variable) 
        #               - from a call to tf.optimizer.compute_gradients(loss_function)
        # min_grad, max_grad (float) - min and max gradients
        #
        # Returns
        # - tensorflow gradient after being clipped
        #
        
        if gradient is None:
            return gradient #this is necessary for initialization in tensorflow or it throws an error
        return tf.clip_by_value(gradient, self.min_gradient, self.max_gradient)
    
    def run_optimizer(self):
        with tf.variable_scope('optimizer'):
            
            #target y variables for use in the loss function in algorithm 1
            self.target_y = tf.placeholder(tf.float32, shape=[None], name="target_y")
            
            #chosen actions for use in the loss function in algorithm 1
            self.chosen_actions = tf.placeholder(tf.int32, shape=[None], name="chosen_actions")
            
            #convert the chosen actions to a one-hot vector.
            self.chosen_actions_one_hot = tf.one_hot(self.chosen_actions,
                                                   self.num_actions,
                                                   on_value=1.0,
                                                   off_value=0.0,
                                                   axis=None,
                                                   dtype=None,
                                                   name="chosen_actions_one_hot")
            
            #The q value is that of the dot product of the prediction network
            #with the one-hot representation of the chosen action. this gives us a single chosen action
            #because reduce_sum will add this up over each of the indexes, all but one of which are non-zero
            self.predict_y = tf.reduce_sum(self.chosen_actions_one_hot * self.q_predictions,
                                         axis=1, #reduce along the second axis because we have batches
                                         name="predict_y")
            
            #Implement mean squared error between the target and prediction networks as the loss
            self.loss = tf.reduce_mean(tf.square(tf.subtract(self.target_y, self.predict_y)), name="loss")
            #self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.predict_y,
            #                                                                   logits=self.target_y,
            #                                                                   name="loss"))
            
            self.learning_rate = tf.train.exponential_decay(self.learning_rate_init,
                                                           self.global_step,
                                                           self.learning_rate_decay_steps,
                                                           self.learning_rate_decay)
            
            #and pass it to the optimizer to train on this defined loss function
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                name="adam")
            
            
            #Clip the gradients if the option is enabled by breaking the optimizer's minimize() into a (get, clip, apply) set of operations
            if self.clip_gradients_enabled:
                gradients_and_variables = self.opt.compute_gradients(self.loss)
                clipped_gradients = [(self.clip_gradients(grad), var) for grad, var in gradients_and_variables]
                self.optimizer = self.opt.apply_gradients(clipped_gradients) #this increments global step
            else:
                self.optimizer = self.opt.minimize(self.loss)

            
    def run_minibatch(self):
        #print("Running minibatch. Sarst memory contains %d entries" % self.replay_memory.memory_size)
        
        #These are of size batch_size, not single values
        state, action, reward, state_prime, state_prime_is_terminal = self.replay_memory.get_batch_sample(self.batch_size)

        #get the logits from the target network for this resulting state
        q_value_state_prime = self.q_targets.eval({self.network_inputs['target_network'] : state_prime})
        
        #the max logit is the max action q value
        max_q_value_state_prime = np.max(q_value_state_prime, axis=1)
        
        #the state_prime_is_terminal * 1  converts [True, False, True] to [1,0,1].
        # Subtracting this from 1 effectively eliminates the entire term, leaving just reward for terminal states
        target_y = reward + (self.gamma * max_q_value_state_prime * (1 - (state_prime_is_terminal*1)))
        
        
        #Now that the terms are in place, run a session
        _, self.report_predictions, lr, self.one_hot_actions, self.final_predictions, self.report_loss = self.session.run([self.optimizer, self.q_predictions,                                        self.learning_rate, self.chosen_actions_one_hot, self.predict_y, self.loss], {
            self.network_inputs['prediction_network'] : state,     # it'll need the states possibly
            self.chosen_actions : action, #and definitely the actions
            self.target_y : target_y,     #and the targets in the optimizer
            self.global_step : self.total_iterations #and update our global step. TODO, maybe this should be self.global_step. make sure isn't incremented twice with minimize() call
        })
        
        #if self.episode_iterations == 0 and self.episode_count % 5 == 0:
        #    print "Episode %d\t\tLearning Rate: %.9f\t\tEpsilon: %.6f" % (self.episode_count, lr, self.epsilon)
        
        self.minibatches_run += 1
        
    def _update_last_100_rewards(self, deq, to_add):
        if len(deq) == 100:
            _ = deq.popleft()
        deq.append(to_add)
    
    def start_new_episode(self):
        self._update_last_100_rewards(self.last_100_episode_rewards, self.total_episode_reward)
        self.episode_rewards[self.episode_count] = self.total_episode_reward
        self.episode_count += 1
        self.episode_iterations = 0
        self.total_episode_reward = 0.
        
        r = np.random.randint(32) #returns a level between 0 (level 1-1) and 31 (level 8-4)
        self.env.change_level(r)
    
    def train(self):
        mean_episode_rewards = []
        mean_episode_rewards_last_100 = []
        
        def run_report():
            print("\n***********Report*************")
            print("\tEpisode: %d" % self.episode_count)
            print("\tEpisode Reward: %.4f" % self.total_episode_reward)
            print("\tEpisode Steps: %.4f" % self.episode_iterations)
            print("\tTotal iterations: %d" % self.total_iterations)
            print("\tMemory Size: %d" % self.replay_memory.memory_size)
            print("\tBatch Loss: %.4f" % self.report_loss)
            print("\tEpsilon: %.4f" % self.epsilon)
        
        #Load the emulator
        #If this doesn't work, see:
        #https://github.com/ppaquette/gym-super-mario/issues/6
        self.env.reset()
        
        #Unlock all levels
        #we do not want to deal with actually playing the game straight through, so we will skip all over
        #the place by picking a new level each time Mario dies
        self.env.locked_levels = [False] * 32
        
        for episode in range(self.num_episodes):
            
            episode_terminated = False
            
            self.cur_state = None
            #self.previous_state_info = None
        
            #last 100 episode check?
            
            while not episode_terminated:
                
                self.env.render()
                
                #First frame - do nothing
                if self.cur_state is None:
                    action = self.actions[0] #noop defined in action_space.py
                else:
                    #The first thing we need to do is select an action
                    action = self.choose_action(self.cur_state)
            
                #Then we act!
                state_prime, reward, done, state_info = self.env.step(action)
                #print("Action: %s\tDiscrete:%d\tReward: %.4f" % (str(action), self.actions.index(action), reward))
                
                #Then we store the image as grayscale using the ITU-R 601-2 luma transform
                #This will allow us to save 3x on the SARST memory consumption, plus save in learning
                #the convolutional channels. It also enables us to pass multiple images to learn velocity
                #and acceleration of objects easier.
                state_prime = np.dot(state_prime[...,:3], [0.299, 0.587, 0.114])
                state_prime = np.expand_dims(state_prime, axis=2)
                
        
                #Then we store away what happened, unless we are in the first stage
                if self.cur_state is not None:
                    self.replay_memory.add_to_memory(self.cur_state,
                                                    self.actions.index(action), #SARST has access no access to button maps
                                                    reward,
                                                    state_prime,
                                                    done)
                    
                #Then we update the current state after safely storing it
                self.cur_state = state_prime
                
                #then run an actual neural network trainer on that replay memory
                if self.replay_memory.memory_size >= self.batch_size:
                    self.run_minibatch()
                
                #increment counters for decayed variables
                self.total_iterations += 1
                self.episode_iterations += 1
                
                #decay agent exploration 
                self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)
                
                #update the target network, however often you choose to
                if self.total_iterations % self.target_network_update_frequency == 0:
                    self.copy_prediction_parameters_to_target_network()
                
                #The rewards get updated for the episode
                self.total_episode_reward += reward
                
                if done:
                    #print("episode %d finished in %d iterations with reward %.4f" % (self.episode_count+1, self.episode_iterations, self.total_episode_reward))
                    print("ran total of %d minibatches so far" % self.minibatches_run)
                    if self.episode_count % self.report_frequency == 0:
                        run_report()
                    episode_terminated=True
                    self.all_episode_rewards.append(self.total_episode_reward)
                    self.start_new_episode()
        
        #nore more episodes, close the monitor
        print "Mean episode rewards per %d timesteps are: \n %s" % (self.report_frequency, str(mean_episode_rewards[:self.episode_count]))
        #print self.all_episode_rewards
        self.env.close()
            #if done start new episode

