from agents import DDPG_Agent as Agent
import unityagents
from collections import deque
import pickle
import numpy as np
from typing import Collection


class AgentTrainer():
    ''' Skeleton adapted from Udacity exercise sample code.'''
    def __init__(self,
                 env: unityagents.environment.UnityEnvironment,
                 max_t: int = 1000,
                 max_n_episodes: int = 3000,
                 target_score: float =  0.5
                 ):
        self.env = env
        self.max_t = max_t
        self.max_n_episodes = max_n_episodes
        self.brain_name = env.brain_names[0]
        self.target_score = target_score

    def env_step(self, action):

        env_info = self.env.step(action)[self.brain_name]
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        # see if episode has finished
        done = env_info.local_done[0]

        return next_state, reward, done

    def train_agent(self,
                    agents: Collection[Agent],
                    hyperparams):
        """Deep Q-Learning.

        Params
        ======
            agent:Agent
        """
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores

        solved = False

        for i_episode in range(1, self.max_n_episodes+1):
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            for agent in agents:
                agent.reset()
            # get the current states
            states = env_info.vector_observations
            scores_episode = np.zeros(2)
            for t in range(self.max_t):
                actions = np.vstack((agents[0].act(states[0]),agents[1].act(states[1])))
                env_info = self.env.step(actions)[self.brain_name]           # send all actions to tne environment
                next_states = env_info.vector_observations         # get next state (for each agent)
                rewards = env_info.rewards                         # get reward (for each agent)
                dones = env_info.local_done                        # see if episode finished
                scores_episode += env_info.rewards                         # update the score (for each agent)

                for agent, state,action, reward, next_state, done in zip(agents,
                                                                   states,
                                                                   actions,  
                                                                   rewards,      
                                                                   next_states,
                                                                   dones):
                    agent.step(state, action, reward, next_state, done)
                states= next_states
                if np.any(dones):
                    break

            scores_window.append(scores_episode.mean())       # save most recent score
            scores.append(scores_episode.mean())              # save most recent score
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                scores_window_mean = np.mean(scores_window)
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                    i_episode, scores_window_mean))

            if np.mean(scores_window) >= self.target_score and not solved:
                solved = True
                print('Env solved in {:d} episodes! Avg Score: {:.2f}'.format(
                    i_episode-100, np.mean(scores_window)))
                break

        self.save_scores( hyperparams, f'''./{hyperparams['description']}''',scores)
        return agent

    def save_scores(self, hyperparams, filename,scores):
        obj = {'scores': scores,
               'hyperparams': hyperparams}

        with open(filename, 'wb') as f:
            pickle.dump(obj, f)
