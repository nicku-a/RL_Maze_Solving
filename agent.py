import numpy as np
import torch

import copy
import collections
import random
import time


class Agent:

    # Function to initialise the agent
    def __init__(self):
        # Record time started to decay epsilon and episode length with time
        self.start_time = time.time()
        # Set the episode length
        self.episode_length = 500
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # Track number of episodes
        self.num_episodes_taken = 0
        # Track the number of steps the agent has taken within an episode
        self.steps_in_episode = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        # Create deep Q-network
        self.dqn = DQN()
        # Create experience replay buffer
        self.buffer = ReplayBuffer()
        # Boolean value to track whether greedy policy reaches goal, so we can stop training
        self.done_training = False

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.steps_in_episode == self.episode_length:
            # Change variables for the start of the next episode
            self.num_episodes_taken += 1
            self.steps_in_episode = 0
            if self.num_episodes_taken % 15 == 0:
                self.episode_length = 100
            else:
                # Decrease episode length linearly from 500 to 100 over 10 minutes
                self.episode_length = int(-2/3 * self._get_time_passed() + 501)
            return True
        else:
            return False

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        # Check greedy policy every 15 episodes
        if self.num_episodes_taken % 15 == 0 or self.done_training:
            action = self.get_greedy_action(state)
        else:
            action = self.get_epsilon_greedy_action(state)
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Store the action; this will be used later, when storing the transition
        self.action = action
        return action

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def discrete_action_to_continuous(self, discrete_action):
        if discrete_action == 0:
            continuous_action = np.array([0, 0.02], dtype=np.float32)
        if discrete_action == 1:
            continuous_action = np.array([0.02, 0], dtype=np.float32)
        if discrete_action == 2:
            continuous_action = np.array([0, -0.02], dtype=np.float32)
        # if discrete_action == 3:
        #     continuous_action = np.array([np.cos(np.deg2rad(45))*0.019999, np.cos(np.deg2rad(45))*0.019999], dtype=np.float32)
        # if discrete_action == 4:
        #     continuous_action = np.array([np.cos(np.deg2rad(45))*0.019999, -np.cos(np.deg2rad(45))*0.019999], dtype=np.float32)
        return continuous_action

    # Function to convert continuous action (as used by the environment) to a discrete action (as used by a DQN).
    def continuous_action_to_discrete(self, continuous_action):
        if np.all(continuous_action == np.array([0, 0.02], dtype=np.float32)):
            discrete_action = 0
        if np.all(continuous_action == np.array([0.02, 0], dtype=np.float32)):
            discrete_action = 1
        if np.all(continuous_action == np.array([0, -0.02], dtype=np.float32)):
            discrete_action = 2
        # if np.all(continuous_action == np.array([np.cos(np.deg2rad(45))*0.019999, np.cos(np.deg2rad(45))*0.019999], dtype=np.float32)):
        #     discrete_action = 3
        # if np.all(continuous_action == np.array([np.cos(np.deg2rad(45))*0.019999, -np.cos(np.deg2rad(45))*0.019999], dtype=np.float32)):
        #     discrete_action = 4
        return discrete_action

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):

        # Increase number of steps taken in individual episode
        self.steps_in_episode += 1

        # Check if reached goal
        if self.num_episodes_taken % 15 == 0:
            if distance_to_goal < 0.03 and self.steps_in_episode <= 100:
                self.done_training = True
                # print('Reached goal executing greedy policy!')
            # elif self.steps_in_episode == 100:
            #     print('Failed to reach goal. Distance to goal: ', distance_to_goal)

        # Convert the distance to a reward
        reward = -(distance_to_goal**0.3)

        # Create a transition
        transition = (self.state, self.continuous_action_to_discrete(self.action), reward, next_state)

        # Mini-batch size
        bs = int(self.episode_length/2)

        # Add transition to buffer
        self.buffer.update_buffer(transition)

        # Train Q-network - but not during greedy episodes
        if self.num_steps_taken >= bs and not (self.done_training or self.num_episodes_taken % 15 == 0):
            self.dqn.train_q_network(transition, self.buffer, bs)

        # Update target network every 60 steps - but not during greedy episodes
        if self.num_steps_taken % 60 == 0 and not (self.done_training or self.num_episodes_taken % 15 == 0):
            self.dqn.update_target_network()

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):

        # Use Q-network to get predictions
        predictions = self.dqn.q_network.forward(torch.unsqueeze(torch.tensor(state), 0).float())[0]
        opt_action = int(torch.argmax(predictions, 0))

        # Convert discrete prediction to continuous action
        continuous_action = self.discrete_action_to_continuous(opt_action)

        return continuous_action

    # Function used to get time passed to decay epsilon and episode length
    def _get_time_passed(self):
        return time.time() - self.start_time

    # Function to get the epsilon-greedy action for a particular state
    def get_epsilon_greedy_action(self, state):

        # Set exploration parameter epsilon. Epsilon decays from 1 to 0 over 10 minutes.
        epsilon = -1/600 * self._get_time_passed() + 1

        predictions = self.dqn.q_network.forward(torch.unsqueeze(torch.tensor(state), 0).float())[0]
        opt_action = int(torch.argmax(predictions, 0))

        # Epsilon-greedy policy
        if random.random() > epsilon:
            discrete_action = opt_action
        else:
            discrete_action = random.choice([0, 1, 2])

        # Convert discrete prediction to continuous action
        continuous_action = self.discrete_action_to_continuous(discrete_action)

        return continuous_action


# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the
    # dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This has two hidden layers, each with 100 units.
        self.lin_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.lin_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output.
    def forward(self, input):
        lin_1_output = torch.nn.functional.relu(self.lin_1(input))
        lin_2_output = torch.nn.functional.relu(self.lin_2(lin_1_output))
        output = self.output_layer(lin_2_output)
        return output


# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=3)
        self.target_network = copy.deepcopy(self.q_network)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each
        # gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.01)

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a
    # transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, transition, buffer, bs):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Sample mini batch from replay buffer of size n
        mini_batch = buffer.rand_mini_batch(bs)
        # Compute loss
        loss = self._calculate_batch_loss(mini_batch)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network
        # parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()

    def update_target_network(self):
        self.target_network = copy.deepcopy(self.q_network)

    def _calculate_batch_loss(self, mini_batch):
        # Convert minibatch into separate tensors of states, actions, rewards and next states
        mini_batch_states = []
        mini_batch_actions = []
        rewards = []
        mini_batch_next_states = []
        for i in range(len(mini_batch)):
            mini_batch_states.append(mini_batch[i][0])
            mini_batch_actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            mini_batch_next_states.append(mini_batch[i][3])
        rewards_tensor = torch.tensor(rewards)
        mini_batch_states_tensor = torch.tensor(mini_batch_states)
        mini_batch_actions_tensor = torch.tensor(mini_batch_actions)
        mini_batch_next_states_tensor = torch.tensor(mini_batch_next_states)

        # # Double Deep QN - too slow to learn because more complex
        # max_targ_next, indices = torch.max(self.target_network.forward(mini_batch_next_states_tensor), 1)
        # max_next_state_q_values = self.q_network.forward(mini_batch_next_states_tensor.float()).gather(dim=1,
        #                                                                                                index=indices.
        #                                                                                        long().unsqueeze(-1)).\
        #     squeeze(-1)

        # Use target network to predict the maximum next state Q-values
        max_next_state_q_values, _ = torch.max(self.target_network.forward(mini_batch_next_states_tensor), 1)
        max_next_state_q_values = max_next_state_q_values.detach()

        # Use Q-network to predict current state Q-values
        predictions = self.q_network.forward(mini_batch_states_tensor).gather(dim=1, index=mini_batch_actions_tensor.
                                                                              long().unsqueeze(-1)).squeeze(-1)

        # Bellman equation
        return torch.nn.MSELoss()(rewards_tensor + 0.98 * max_next_state_q_values, predictions)


class ReplayBuffer:

    # The class initialisation function.
    def __init__(self):
        self.buffer = collections.deque(maxlen=5000)

    # Function to add a new transition to the buffer
    def update_buffer(self, transition):
        self.buffer.append(transition)

    # Function to randomly sample minibatch from experience replay buffer
    def rand_mini_batch(self, bs):
        mini_batch = random.sample(self.buffer, bs)
        return mini_batch
