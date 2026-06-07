import gymnasium as gym
import flappy_bird_gymnasium
import torch
import imageio
import os

# Import our neural network architecture from the src directory
from src.dqn import DQN

# Ensure the assets directory exists to save the GIF
os.makedirs("assets", exist_ok=True)

print("Initializing environment and loading model...")
device = torch.device("cpu")

# Use 'rgb_array' to capture raw pixel data instead of rendering to a window
env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n

# Initialize the Dueling Double DQN and load the pre-trained weights
# Note: Ensure hidden_dim matches the fc1_nodes specified in hyperparameters.yml (512 for FlappyBird)
policy_dqn = DQN(num_states, num_actions, hidden_dim=512, enable_dueling_dqn=True).to(device)
policy_dqn.load_state_dict(torch.load("weights/flappybird1_best.pt", map_location=device))
policy_dqn.eval() # Set network to evaluation mode (disables dropout/batchnorm if any)

frames = []
state, _ = env.reset()
state = torch.tensor(state, dtype=torch.float, device=device)

print("Agent is playing... Capturing frames. Please wait.")

# Record a maximum of 600 frames (approx. 20 seconds of gameplay at 30 FPS)
# This prevents the GIF file size from becoming too large for GitHub
max_frames = 600

for step in range(max_frames):
    # Capture the current screen as a numpy array (RGB)
    frame = env.render()
    frames.append(frame)

    # Agent selects the best action based on the learned policy
    with torch.no_grad():
        action = policy_dqn(state.unsqueeze(0)).squeeze().argmax().item()

    # Apply the action to the environment
    state, reward, terminated, truncated, _ = env.step(action)
    state = torch.tensor(state, dtype=torch.float, device=device)

    # Stop recording if the bird crashes before reaching max_frames
    if terminated or truncated:
        print(f"Bird crashed at frame {step}. Ending recording.")
        break

env.close()

print(f"Processing {len(frames)} frames into a GIF. This may take 10-15 seconds...")

# Convert the captured frames into a GIF at 30 Frames Per Second (FPS)
imageio.mimsave("assets/stage3_best.gif", frames, fps=30)

print("Success! The GIF has been saved to 'assets/stage3_best.gif'.")