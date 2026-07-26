import os
os.environ["SDL_AUDIODRIVER"] = "dummy"

import torch
import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml

print('GPU Available:', torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent:
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.replay_memory_size = hyperparameters['replay_memory_size']  # size of replay memory
        self.mini_batch_size    = hyperparameters['mini_batch_size']     # size of the training data set sampled from the replay memory
        self.epsilon_init       = hyperparameters['epsilon_init']        # 1 = 100% random actions
        self.epsilon_decay      = hyperparameters['epsilon_decay']       # epsilon decay rate
        self.epsilon_min        = hyperparameters['epsilon_min']         # minimum epsilon value


    def run(self, is_training=True, render=False):
        # env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.actions_space.n

        rewards_per_episode = []

        policy_dqn = DQN(num_states, num_actions).to_device(device)

        if is_training:
            memory = ReplayMemory(10000)

        for episode in itertools.count():
            state, _ = env.reset()
            terminated = False
            episode_reward = 0.0

            while not terminated:
                # Next action:
                # (feed the observation to your agent here)
                action = env.action_space.sample()

                # Processing:
                new_state, reward, terminated, _, info = env.step(action)

                # accumulate reward
                episode_reward += reward

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))

                # Move to new state
                state = new_state

            rewards_per_episode.append(episode_reward)