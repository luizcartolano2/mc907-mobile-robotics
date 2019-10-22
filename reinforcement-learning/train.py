import ast
import logging
from environment import Robot
import numpy as np
import itertools
import matplotlib
import matplotlib.style
import pandas as pd
import sys
from collections import defaultdict
import plotting_r as plotting
import json
matplotlib.style.use('ggplot')

SPEED = 0.7
logging.basicConfig(filename='reinforcement-learning.log', filemode='w', level=logging.DEBUG)


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

def qLearning(env, num_episodes, discount_factor=1, alpha=0.01, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control.
    Finds the optimal greedy policy while improving
    following an epsilon-greedy policy
    """
    # Action value function
    # A nested dictionary that maps
    # state -> (action -> action-value).
    # import pdb; pdb.set_trace()

    try:
        with open('model.txt', 'r') as f:
            json_file = json.load(f)
        Q = {ast.literal_eval(k): np.array(ast.literal_eval(v)) for k, v in json_file.items()}
    except:
        Q = defaultdict(lambda: np.zeros(3))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths = np.zeros(num_episodes),
        episode_rewards = np.zeros(num_episodes)
        )

    # Create an epsilon greedy policy function
    # appropriately for environment action space
    policy = createEpsilonGreedyPolicy(Q, epsilon, 3)

    # For every episode
    for ith_episode in range(num_episodes):

        # Reset the environment and pick the first action
        state = env.reset_sim()
        state = tuple(state['proxy_sensor'][0])
        logging.debug('Ith_Episode: {}'.format(ith_episode))
        for t in itertools.count():

            logging.debug('\tt_episode: {}'.format(t))

            # get probabilities of all actions from current state
            action_probabilities = policy(state)
            logging.debug('\t\taction_probabilities: {}'.format(action_probabilities))

            # choose action according to
            # the probability distribution
            action = np.random.choice(np.arange(len(action_probabilities)), p=action_probabilities)
            logging.debug("\t\tActions: {}".format(action))
            if action == 0:
                action_env = [1.5, 1.5]
            elif action == 1:
                action_env = [0.5, 1.5]
            elif action == 2:
                action_env = [1.5, 0.5]
            else:
                raise Exception("Invalid action!")

            # take action and get reward, transit to next state
            next_state, reward, done = env.step(action_env)

            next_state = tuple(next_state['proxy_sensor'][0])
            reward = reward['proxy_sensor']
            logging.debug("\t\tReward: {}".format(reward))

            # Update statistics
            stats.episode_rewards[ith_episode] += reward
            stats.episode_lengths[ith_episode] = t

            # TD Update
            best_next_action = np.argmax(Q[next_state])
            logging.debug("\t\tBest Next Action: {}".format(best_next_action))
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            logging.debug("\t\tTD Target: {}".format(td_target))
            td_delta = td_target - Q[state][action]
            logging.debug("\t\tTD Delta: {}".format(td_delta))
            Q[state][action] += alpha * td_delta

            # done is True if episode terminated
            if done:
                break

            state = next_state
        epsilon -= 0.001

    env.destroy_sim()

    return Q, stats

if __name__ == '__main__':
    env = Robot()

    Q, stats = qLearning(env, 1)

    # save the learned model
    with open('model.txt', 'w') as f:
        json.dump({str(k): str(tuple(v)) for k, v in Q.items()}, f)

    plotting.plot_episode_stats(stats)
