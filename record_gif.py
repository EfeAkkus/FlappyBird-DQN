import gymnasium as gym
import flappy_bird_gymnasium
import torch
import imageio
import os

from src.dqn import DQN

os.makedirs("assets", exist_ok=True)

print("Initializing environment and loading model...")
device = torch.device("cpu")

env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n

policy_dqn = DQN(num_states, num_actions, hidden_dim=512, enable_dueling_dqn=True).to(device)
policy_dqn.load_state_dict(torch.load("weights/flappybird1_best.pt", map_location=device))
policy_dqn.eval()

frames = []
state, _ = env.reset()
state = torch.tensor(state, dtype=torch.float, device=device)

print("Agent is playing... Capturing frames. Please wait.")

max_frames = 600

for step in range(max_frames):
    frame = env.render()
    
    # OPTIMIZATION: Save every 2nd frame to reduce file size by 50% (prevents lag)
    if step % 2 == 0:
        frames.append(frame)

    with torch.no_grad():
        action = policy_dqn(state.unsqueeze(0)).squeeze().argmax().item()

    state, reward, terminated, truncated, _ = env.step(action)
    state = torch.tensor(state, dtype=torch.float, device=device)

    if terminated or truncated:
        print(f"Bird crashed at frame {step}. Ending recording.")
        break

env.close()

print(f"Processing {len(frames)} optimized frames into a GIF...")

# OPTIMIZATION: 'loop=0' enables infinite looping. 'fps=20' keeps it smooth and lightweight.
imageio.mimsave("assets/stage3_looped.gif", frames, fps=20, loop=0)

print("Success! Optimized GIF saved to 'assets/stage3_looped.gif'.")