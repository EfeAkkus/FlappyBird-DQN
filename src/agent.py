# import os

# #use SDL_AUDIODRIVER=dummy flappy_bird_gymnasium for terminal
# os.environ["SDL_AUDIODRIVER"] = "dummy"
# import torch
# import flappy_bird_gymnasium
# import gymnasium
# from dqn import DQN
# from expeirence_replay import ReplayMemory
# import itertools
# import yaml
# import random
# from torch import nn

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# class Agent():

    
#     def __init__(self, hyperparameter_set):
#         with open("hyperparameters.yml", "r") as file:
#             all_hyperparameter_sets = yaml.safe_load(file)
#             hyperparameters = all_hyperparameter_sets[hyperparameter_set]

#         # print(hyperparameters)

#         self.replay_memory_size = hyperparameters["replay_memory_size"]  # size of replay memory
#         self.mini_batch_size = hyperparameters["mini_batch_size"]        # size of the training data set sampled from the replay memory
#         self.epsilon_init = hyperparameters["epsilon_init"]              # 1 = 100% random actions
#         self.epsilon_decay = hyperparameters["epsilon_decay"]            # epsilon decay rate
#         self.epsilon_min = hyperparameters["epsilon_min"]                # minimum epsilon value
#         self.discount_factor_g = hyperparameters["discount_factor_g"]  
#         self.learning_rate_a = hyperparameters["learning_rate_a"]  
#         self.network_sync_rate = hyperparameters["network_sync_rate"]
        
#         self.loss_fn=nn.MSELoss()
#         self.optimizer = None
        
        
#     def run(self , is_training = True , render = False):
        
#         #env = gymnasium.make("FlappyBird-v0",render_mode="human",use_lidar=False)
#         env = gymnasium.make("CartPole-v1",render_mode="human" if render else None)

#         num_states = env.observation_space.shape[0]
#         num_actions = env.action_space.n
#         rewards_per_episode= []
#         epsilon_history = []
        
#         policy_dqn = DQN(num_states, num_actions).to(device)
        
#         if is_training:
#             memory = ReplayMemory(self.replay_memory_size)
#             epsilon = self.epsilon_init
#             target_dqn = DQN(num_states, num_actions).to(device)
#             target_dqn.load_state_dict(policy_dqn.state_dict())
            
#             step_count = 0
            
#             #Policy network optimizer "Adam"
            
#             self.optimizer = torch.optim.Adam(policy_dqn.parameters() , lr=self.learning_rate_a)

#         for episode in itertools.count():
#             state, _ = env.reset()
#             state = torch.tensor(state , dtype = torch.float , device = device)

#             terminated = False
#             episode_reward = 0.0
            
            
#             while  not terminated:
#                 # Next action:
#                 if is_training and random.random() < epsilon:
#                     action = env.action_space.sample()
#                 else :
#                     with torch.no_grad():
#                         action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

#                 # Processing:
#                 new_state, reward, terminated, truncated, info = env.step(action.item())
                
#                 episode_reward = episode_reward + reward
                
#                 new_state = torch.tensor(new_state , dtype = torch.float , device = device)
#                 reward = torch.tensor(reward , dtype = torch.float , device = device)
                
                
#                 if is_training:
#                     memory.append((state , action , new_state , reward , terminated))
#                     step_count +=1
#                 state = new_state
                
#             rewards_per_episode.append(episode_reward)
            
#             epsilon = max(epsilon * self.epsilon_decay , self.epsilon_min)
#             epsilon_history.append(epsilon)
            
#             #If enough experience has been collected
#             if len(memory) > self.mini_batch_size:
                
#                 #Sample from memory
                
#                 mini_batch = memory.sample(self.mini_batch_size)
                
#                 self.optimize(mini_batch , policy_dqn , target_dqn)
                
#                 #Copy policy network to target network after certain number of steps
#                 if step_count > self.network_sync_rate :
#                     target_dqn.load_state_dict(policy_dqn.state_dict())
#                     step_count = 0
            
    
#     def optimize(self, mini_batch, policy_dqn, target_dqn):
#         # Transpose the list of experiences and separate each element
#         states, actions, new_states, rewards, terminations = zip(*mini_batch)

#         # Stack tensors to create batch tensors
#         states = torch.stack(states)
#         actions = torch.stack(actions)
#         new_states = torch.stack(new_states)
#         rewards = torch.stack(rewards)
#         terminations = torch.tensor(terminations).float().to(device)

#         with torch.no_grad():   
#             # Calculate target Q values
#             target_q = rewards + (1 - terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]

#         # Calculate Q values from current policy
#         current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

#         # Compute loss for the whole minibatch
#         loss = self.loss_fn(current_q, target_q)

#         # Optimize the model
#         self.optimizer.zero_grad()   # Clear gradients
#         loss.backward()              # Compute gradients
#         self.optimizer.step()        # Update network parameters
    
    
    
# print("Device:", device)

        
# if __name__ == "__main__":
#     agent = Agent("cartpole1")
#     agent.run(is_training = True ,render=True)


import os
import random
import yaml
import argparse
import itertools
from datetime import datetime

import numpy as np
import gymnasium as gym
import flappy_bird_gymnasium 

import torch
from torch import nn
from tensorboardX import SummaryWriter

from experience_replay import ReplayMemory
from dqn import DQN

os.environ['SDL_AUDIODRIVER'] = 'dummy'
# os.environ['SDL_VIDEODRIVER'] = 'dummy'

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Define directories
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Agent():
    def __init__(self, hyperparameter_set):
        # Load configurations
        with open('config/hyperparameters.yml', 'r') as file:
            all_configs = yaml.safe_load(file)
            self.config = all_configs[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set

        # Extract parameters
        self.env_id             = self.config['env_id']
        self.learning_rate_a    = self.config['learning_rate_a']
        self.discount_factor_g  = self.config['discount_factor_g']
        self.replay_memory_size = self.config['replay_memory_size']
        self.mini_batch_size    = self.config['mini_batch_size']
        self.epsilon_init       = self.config['epsilon_init']
        self.epsilon_decay      = self.config['epsilon_decay']
        self.epsilon_min        = self.config['epsilon_min']
        self.stop_on_reward     = self.config['stop_on_reward']
        self.fc1_nodes          = self.config['fc1_nodes']
        self.env_make_params    = self.config.get('env_make_params', {})
        self.enable_double_dqn  = self.config.get('enable_double_dqn', True)
        self.enable_dueling_dqn = self.config.get('enable_dueling_dqn', True)
        
        # Target Network Update configuration
        self.update_type        = self.config.get('update_type', 'hard')
        self.network_sync_rate  = self.config.get('network_sync_rate', 100)
        self.tau                = self.config.get('tau', 0.005)

        # Neural Network essentials
        self.loss_fn = nn.SmoothL1Loss() # Huber Loss: Robust to outliers compared to MSE
        self.optimizer = None

        # Setup Logging & File paths
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}_{timestamp}")
        self.writer = SummaryWriter(log_dir=log_dir)
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}_best.pt')

    def run(self, is_training=True, render=False):
        set_seed(42)
        
        # Initialize Environment
        env = gym.make(self.env_id, render_mode='human' if render else None, **self.env_make_params)
        num_actions = env.action_space.n
        num_states = env.observation_space.shape[0]

        # Initialize Networks
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            target_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            target_dqn.eval() # Target network is never trained directly
            
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

            epsilon = self.epsilon_init
            rewards_per_episode = []
            best_avg_reward = -float('inf')
            step_count = 0
            
            print(f"--- Training Config ---")
            print(f"Device: {device} | Env: {self.env_id}")
            print(f"Double DQN: {self.enable_double_dqn} | Dueling DQN: {self.enable_dueling_dqn}")
            print(f"Update Type: {self.update_type.upper()}")
            print("-----------------------")
        else:
            # Testing mode: Load the best model
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
            policy_dqn.eval()
            print("Model successfully loaded. Starting Evaluation.")

        # Main Interaction Loop
        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            terminated, truncated = False, False
            episode_reward = 0.0

            while not (terminated or truncated) and episode_reward < self.stop_on_reward:
                
                # Epsilon-Greedy Action Selection
                if is_training and random.random() < epsilon:
                    action = torch.tensor(env.action_space.sample(), dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(0)).squeeze().argmax()

                # Step the environment
                new_state, reward, terminated, truncated, _ = env.step(action.item())
                episode_reward += reward
                is_done = terminated or truncated

                new_state_tensor = torch.tensor(new_state, dtype=torch.float, device=device)
                reward_tensor = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    # Store experience
                    memory.append((state, action, new_state_tensor, reward_tensor, is_done))
                    step_count += 1
                    
                    # Network Optimization step
                    if len(memory) > self.mini_batch_size:
                        mini_batch = memory.sample(self.mini_batch_size)
                        self.optimize(mini_batch, policy_dqn, target_dqn)
                        
                        # Target Network Synchronization
                        if self.update_type == 'soft':
                            # Polyak Averaging (Soft Update)
                            for target_param, policy_param in zip(target_dqn.parameters(), policy_dqn.parameters()):
                                target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
                        else:
                            # Hard Update
                            if step_count % self.network_sync_rate == 0:
                                target_dqn.load_state_dict(policy_dqn.state_dict())
                                step_count = 0

                state = new_state_tensor

            # --- Episode Wrap-up ---
            if is_training:
                rewards_per_episode.append(episode_reward)
                
                # Decay Epsilon (Performed per episode, not per step, to encourage structured exploration)
                if len(memory) > self.mini_batch_size:
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                # TensorBoard Logging
                self.writer.add_scalar('Metrics/Episode_Reward', episode_reward, episode)
                self.writer.add_scalar('Hyperparameters/Epsilon', epsilon, episode)

                # Save model based on Moving Average (More stable than saving single lucky episodes)
                if len(rewards_per_episode) >= 50:
                    avg_reward = np.mean(rewards_per_episode[-50:])
                    self.writer.add_scalar('Metrics/Moving_Avg_Reward_50', avg_reward, episode)
                    
                    if avg_reward > best_avg_reward:
                        best_avg_reward = avg_reward
                        torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                        print(f"[Episode {episode:4d}] New Best Avg Reward: {avg_reward:.2f}. Model Saved!")

                if episode % 10 == 0:
                    print(f"Episode: {episode:4d} | Reward: {episode_reward:6.1f} | Epsilon: {epsilon:.4f}")

            # Exit condition
            if episode_reward >= self.stop_on_reward:
                print(f"Target reward threshold ({self.stop_on_reward}) reached. Terminating process.")
                break

        env.close()
        if is_training:
            self.writer.close()

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        """
        Computes the Loss and updates the Policy Network.
        """
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations, dtype=torch.float, device=device)

        with torch.no_grad():
            if self.enable_double_dqn:
                # Double DQN: Policy net selects the best action, Target net evaluates it.
                best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)
                target_q = rewards + (1 - terminations) * self.discount_factor_g * \
                           target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(1)).squeeze()
            else:
                # Vanilla DQN: Target net selects and evaluates the action.
                target_q = rewards + (1 - terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]

        # Current Q-values from Policy Network
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(1)).squeeze()

        loss = self.loss_fn(current_q, target_q)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping: Prevents exploding gradients during unstable training phases
        torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), max_norm=10)
        
        self.optimizer.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DQN Training and Evaluation Framework')
    parser.add_argument('hyperparameters', help='Hyperparameter configuration block name (e.g., cartpole1)', nargs='?', default='flappybird1')
    parser.add_argument('--train', help='Initiate Training Mode', action='store_true')
    args = parser.parse_args()

    agent = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        agent.run(is_training=True)
    else:
        agent.run(is_training=False, render=True)