from environment import Robot
import numpy as np
import itertools
import matplotlib
import matplotlib.style
import pandas as pd
import sys
from collections import defaultdict
import plotting_r as plotting

matplotlib.style.use('ggplot')

SPEED = 1

def createEpsilonGreedyPolicy(Q, epsilon, num_actions):
    """
    Creates an epsilon-greedy policy based
    on a given Q-function and epsilon.

    Returns a function that takes the state
    as an input and returns the probabilities
    for each action in the form of a numpy array
    of length of the action space(set of possible actions).
    """
    def policyFunction(state):

        action_probabilities = np.ones(num_actions,dtype=float) * epsilon / num_actions

        best_action = np.argmax(Q[state])
        action_probabilities[best_action] += (1.0 - epsilon)
        return action_probabilities

    return policyFunction

def qLearning(env, num_episodes, discount_factor=1.0, alpha=0.6, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control.
    Finds the optimal greedy policy while improving
    following an epsilon-greedy policy
    """
    # Action value function
    # A nested dictionary that maps
    # state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(4))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths = np.zeros(num_episodes),
        episode_rewards = np.zeros(num_episodes)
        )

    # Create an epsilon greedy policy function
    # appropriately for environment action space
    policy = createEpsilonGreedyPolicy(Q, epsilon, 4)

    # For every episode
    for ith_episode in range(num_episodes):

        # Reset the environment and pick the first action
        state = env.reset_sim()
        state = tuple(state['proxy_sensor'][0])
        for t in itertools.count():

            # get probabilities of all actions from current state
            action_probabilities = policy(state)

            # choose action according to
            # the probability distribution
            action = np.random.choice(np.arange(len(action_probabilities)),p = action_probabilities)
            if action == 0:
                action = [SPEED, SPEED]
            elif action == 1:
                action = [0, SPEED]
            elif action == 2:
                action = [SPEED, 0]
            elif action == 3:
                action = [0,0]
            else:
                raise Exception("Invalid action!")

            # take action and get reward, transit to next state
            next_state, reward, done = env.step(action)

            next_state = tuple(next_state['proxy_sensor'][0])
            reward = reward['proxy_sensor']

            # Update statistics
            stats.episode_rewards[ith_episode] += reward
            stats.episode_lengths[ith_episode] = t

            # TD Update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            # done is True if episode terminated
            if done:
                break

            state = next_state

    return Q, stats

if __name__ == '__main__':
    env = Robot()
    Q, stats = qLearning(env, 1000)
