import time
import numpy as np

import matplotlib.pyplot as plt
import torch

from random_environment_edited import Environment
from agent import Agent


# Main entry point
if __name__ == "__main__":

    # This determines whether the environment will be displayed on each each step.
    # When we train your code for the 10 minute period, we will not display the environment.
    display_on = True

    # Create a random seed, which will define the environment
    random_seed = int(time.time())
    np.random.seed(random_seed)

    print(random_seed)

    # Create a random environment
    environment = Environment(magnification=500)

    # Create an agent
    agent = Agent()

    # Get the initial state
    state = environment.init_state

    # Determine the time at which training will stop, i.e. in 10 minutes (600 seconds) time
    start_time = time.time()
    end_time = start_time + 600

    episode = 0

    losses = []
    average_losses = []

    # fig = plt.gcf()
    # fig.show()
    # fig.canvas.draw()

    # Train the agent, until the time is up
    while time.time() < end_time:

        # If the action is to start a new episode, then reset the state
        if agent.has_finished_episode():
            time_passed = time.time()-start_time
            predicted_num_ep = int(episode * 600/(int(time_passed)+1e-3))
            print('Episodes completed/predicted: {}/{}'.format(episode, predicted_num_ep))
            mins = int(np.floor(time_passed/60))
            secs = int((time.time()-start_time) % 60)
            print('Time passed: {} minutes {} seconds'.format(mins, secs))
            state = environment.init_state
            episode += 1
            # average_losses.append(np.mean(losses))
            # plt.plot(range(episode), average_losses)
            # plt.yscale('log')
            # plt.ylabel('Loss')
            # plt.xlabel('Episodes')
            # fig.canvas.draw()
            # plt.pause(0.0001)
            episode_loss = 0
            losses = []
        # Get the state and action from the agent
        action = agent.get_next_action(state)
        # Get the next state and the distance to the goal
        next_state, distance_to_goal = environment.step(state, action)
        # Return this to the agent
        loss = agent.set_next_state_and_distance(next_state, distance_to_goal)
        losses.append(loss)
        # Set what the new state is
        state = next_state

        # Optionally, show the environment
        if display_on:
            environment.show(state)

    policy = []
    # Test the agent for 100 steps, using its greedy policy
    state = environment.init_state
    policy.append(state)
    has_reached_goal = False
    for step_num in range(100):
        action = agent.get_greedy_action(state)
        next_state, distance_to_goal = environment.step(state, action)
        policy.append(next_state)
        # The agent must achieve a maximum distance of 0.03 for us to consider it "reaching the goal"
        if distance_to_goal < 0.03:
            has_reached_goal = True
            break
        state = next_state

        # Optionally, show the environment
        if display_on:
            environment.show(state)

    # Print out the result
    if has_reached_goal:
        print('Reached goal in ' + str(step_num) + ' steps.')
    else:
        print('Did not reach goal. Final distance = ' + str(distance_to_goal))

    environment.draw_policy(policy)
