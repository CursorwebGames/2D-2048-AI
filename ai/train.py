import math
import os
import random
import time
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from datetime import datetime
from itertools import count

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from engine import Engine

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

print("using device", device)

# "struct-like"
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations: int, n_actions: int):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))  # relu nonlinear function
        x = F.relu(self.layer2(x))
        return self.layer3(x)


BATCH_SIZE = 128
""" number of transitions sampled """
GAMMA = 0.99
"""
discount rate (discount future) for equation to bias towards short-term
R_t0 = r_t0 + gamma * r_t0+1 + gamma^2 * r_t0+2 + ...
For example:
Time:     0    1    2    3
Reward:   4   16    8    2
Reward R0 = 4 + gamma*(16) + gamma^2*(8) + gamma^3*(2)

Define Q as expected return (estimator), p as function to get best action:
Q(s, a) = E[r0 + gamma*r1 + gamma^2*r2 + ...]
        = r0 + gamma * E[r1 + gamma*r2 + ...]
        = r0 + gamma * Q(s', p(a'))
"""
EPS_START = 0.9
""" Epsilon, whether to explore or to exploit """
EPS_END = 0.01
EPS_DECAY = 2500
""" Rate of exponential decay of epsilon, higher = slower decay """
TAU = 0.005
""" Update rate (changes teacher, the equation) """
LR = 3e-4
""" Learning rate (how much neural network changes weights after each step) """

n_actions = 4  # up right down left
n_observations = 16  # 4x4

env = Engine()
state = env.reset()

# Used to learn (LR)
policy_net = DQN(n_observations, n_actions).to(device)
# Used to teach (TAU), much more stable
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())


optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10_000)

steps_done = 0
episode_scores = []
start_episode = 0

CHECKPOINT_PATH = "checkpoint_2048.pth"

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    policy_net.load_state_dict(checkpoint["policy_net"])
    target_net.load_state_dict(checkpoint["target_net"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    steps_done = checkpoint["steps_done"]
    episode_scores = checkpoint["episode_scores"]
    start_episode = checkpoint["episode"]
    print(f"resumed from {CHECKPOINT_PATH} at episode {start_episode}")


def select_action(state):
    """Returns [[int]] shape [1, 1]"""
    global steps_done

    # Generate a random float between 0 and 1
    sample = random.random()

    # epsilon threshold
    epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)

    steps_done += 1

    if sample > epsilon:
        # pick exploitation
        with torch.no_grad():  # not training
            return policy_net(state).max(1).indices.view(1, 1)
            # policy_net(state) -> [[10.5, 3.2, 9.6, 7.2]]
            # max(axis=1) -> values: 10.5; indices: 0 (highest Q-value action)
            # view(1, 1) -> reshape to (1, 1) -> (batch size, action)
    else:
        # exploration
        action = random.randrange(n_actions)
        return torch.tensor([[action]], device=device, dtype=torch.long)


def plot_scores(show_result=False):
    plt.clf()

    if show_result:
        plt.title("Result")
    else:
        plt.title("Training...")

    plt.xlabel("Episode")
    plt.ylabel("Score")

    scores = torch.tensor(episode_scores, dtype=torch.float)
    plt.plot(scores.numpy())

    # Plot 100-episode moving average
    if len(scores) >= 100:
        avg = scores.unfold(0, 100, 1).mean(1)
        avg = torch.cat((torch.zeros(99), avg))
        plt.plot(avg.numpy())

    plt.pause(0.001)


def optimize_model():
    """
    Goal: randomly sample past experiences after a playthrough
    so to adjust weights and train the model
    """
    # replay memory not enough experiences, can't create a batch
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)

    # Transition(state=(...), ...)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    # ie batch.next_state is not none
    # [True, False, ...] for each index
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    # [n, 16] not batch size
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a)
    # gather selects uses action_batch as index to select reward from
    # policy_net(state_batch) predicts Q-values for each action.
    # gather selects Q(s_t, a_t), the value of the action actually taken.
    # connected to policy_net params btw
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_t+1) = max_a Q(s_t+1, a) for all next states.
    # Estimated value of next state, (for teaching)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        # Use target to find V(s_t+1) for non-terminal states
        # similar to select_action
        next_state_values[non_final_mask] = (
            target_net(non_final_next_states).max(1).values
        )
    # Compute the Bellman target, Q(s, a) = r + gamma * max Q(s', a')
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Use Huber loss (SmoothL1Loss)
    # This is the error that gradient descent should minimize
    # Minimize H(Q(s, a) - target)
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Backpropagate loss to compute gradients
    optimizer.zero_grad()
    loss.backward()

    # In-place gradient clipping
    # _ means it modifies in-place
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()  # optimizes policy_net based on the gradients computed


if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 6000
else:
    num_episodes = 1000


def save_checkpoint(episode):
    torch.save(
        {
            "policy_net": policy_net.state_dict(),
            "target_net": target_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "steps_done": steps_done,
            "episode_scores": episode_scores,
            "episode": episode,
        },
        CHECKPOINT_PATH,
    )


start_time = time.time()

interrupted = False

try:
    for i_episode in tqdm(
        range(start_episode, num_episodes),
        initial=start_episode,
        total=num_episodes,
        desc="training",
    ):
        # Initialize the environment and get its state
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():  # infinite counter
            action = select_action(state)
            board, reward, done, info = env.step(int(action.item()))
            reward = torch.tensor([reward], device=device)

            if done:
                next_state = None
            else:
                next_state = torch.tensor(
                    board, dtype=torch.float32, device=device
                ).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy=learner network)
            optimize_model()

            # Soft update of the target=teacher network's weights each step
            # closer to policy
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_scores.append(info["score"])
                epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(
                    -1.0 * steps_done / EPS_DECAY
                )
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                tqdm.write(
                    f"[{timestamp}] episode {i_episode + 1}/{num_episodes} "
                    f"score={info['score']} steps={t + 1} epsilon={epsilon:.3f}"
                )
                break

        save_checkpoint(i_episode + 1)
except KeyboardInterrupt:
    interrupted = True
    tqdm.write(f"interrupted, checkpoint saved to {CHECKPOINT_PATH}")

elapsed = time.time() - start_time
if episode_scores:
    best_score = max(episode_scores)
    result = "beat the game!" if best_score >= 18_000 else "did not beat the game"
    print(
        f"{'Stopped' if interrupted else 'Complete'} in {elapsed / 60:.1f} min "
        f"(best score={best_score}, {result})"
    )
    plot_scores(show_result=True)
    plt.ioff()
    plt.show()

if not interrupted:
    torch.save(policy_net.state_dict(), "dqn_2048.pth")
    print("saved model to dqn_2048.pth")

# ideas: more reward weights: empty cell count, monotonicity, max-tile-in-corner, penalize no-op moves, penalize terminal
