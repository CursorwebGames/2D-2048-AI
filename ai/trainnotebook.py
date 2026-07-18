# %% [markdown]
# # 2048 DQN training (notebook edition)
#
# Same training pipeline as train.py, split into `# %%` cells so it can be
# run interactively in VS Code / Jupyter (or converted to .ipynb with
# jupytext). The training-loop cell can be re-run to continue training,
# and the plot updates inline after each eval.

# %% Imports and device
import math
import os
import random
import signal
import time
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

from tqdm.auto import tqdm  # renders as a widget in notebooks, text in terminals
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from engine import Engine, WIN_EXPONENT

# No live-updating plots: the eval numbers stream as text during training
# (tqdm.write below), and the plot is rendered once at the end -- inline
# notebook backends display figures automatically when a cell finishes,
# so no backend detection or interactive redraw machinery is needed.

# if GPU is to be used
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

print("using device", device)

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 6000
else:
    num_episodes = 1000

# %% Replay memory and the Q-network
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


HIDDEN_SIZE = 128
"""Hidden layer width. 128 with scalar inputs is the config that showed a
real climbing eval curve. (A one-hot/256-wide variant was tried and was
WORSE at the ~1000-episode budget -- one-hot removes the free numeric
generalization scalar inputs give, and needs far more episodes to pay off.
Revisit only with 10k+ episode runs.)"""


class DQN(nn.Module):
    def __init__(self, n_observations: int, n_actions: int):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, HIDDEN_SIZE)
        self.layer2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.layer3 = nn.Linear(HIDDEN_SIZE, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))  # relu nonlinear function
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# %% Hyperparameters
BATCH_SIZE = 128
""" number of transitions sampled """
GAMMA = 0.99
"""
discount rate (discount future) for equation to bias towards short-term
R_t0 = r_t0 + gamma * r_t0+1 + gamma^2 * r_t0+2 + ...

Define Q as expected return (estimator), p as function to get best action:
Q(s, a) = E[r0 + gamma*r1 + gamma^2*r2 + ...]
        = r0 + gamma * E[r1 + gamma*r2 + ...]
        = r0 + gamma * Q(s', p(a'))
"""
EPS_START = 0.9
""" Epsilon, whether to explore or to exploit """
EPS_END = 0.01
EPS_DECAY = num_episodes / 5
"""Rate of exponential decay of epsilon, scaled to num_episodes instead of a
fixed constant, so exploration tapers off at roughly the same *fraction* of
training regardless of run length (a fixed constant tuned for a 1000-episode
CPU run would leave a 6000-episode GPU run at min epsilon for the last 75%
of training). exp(-5) ~ 0.007, so epsilon is within ~1% of EPS_END by the
final episode."""
TAU = 0.005
""" Update rate (changes teacher, the equation) """
LR = 3e-4
""" Learning rate (how much neural network changes weights after each step) """

REWARD_SCALE = 100.0
"""Rewards are divided by this before entering the Bellman update, keeping
Q-targets roughly in [-10, +10] (per-step |r| <= 1) so TD errors stay in
SmoothL1Loss's quadratic zone. See train.py for the full derivation."""

n_actions = 4  # up right down left
n_observations = 16  # 4x4 board of exponents, scaled to [0, 1]

# %% Environment, networks, optimizer, checkpoint resume
env = Engine()
state = env.reset()

# Used to learn (LR)
policy_net = DQN(n_observations, n_actions).to(device)
# Used to teach (TAU), much more stable
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
# 100k transitions ~ several hundred episodes of history
memory = ReplayMemory(100_000)

steps_done = 0
episode_scores = []
eval_history = []
"""(episode, mean greedy score) pairs -- the true learning curve, measured
without exploration noise (see eval_policy)."""
start_episode = 0

CHECKPOINT_PATH = "checkpoint_2048.pth"
# Resume from a checkpoint if one exists, so re-running this cell (or a
# fresh kernel) continues training an existing model instead of starting
# over. The replay buffer isn't saved, so it's rebuilt below via
# warm_fill_buffer before any optimizer steps run.
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    try:
        policy_net.load_state_dict(checkpoint["policy_net"])
    except RuntimeError as e:
        raise SystemExit(
            f"{CHECKPOINT_PATH} was trained with a different network "
            "architecture (input size or hidden width changed since it "
            "was saved). Delete or rename it to start fresh."
        ) from e
    target_net.load_state_dict(checkpoint["target_net"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    steps_done = checkpoint["steps_done"]
    episode_scores = checkpoint["episode_scores"]
    eval_history = checkpoint.get("eval_history", [])
    start_episode = checkpoint["episode"]

    # Extend the run by one more full episode budget past the checkpoint,
    # and re-derive EPS_DECAY from the new total -- otherwise epsilon would
    # still be evaluated against the ORIGINAL num_episodes, which by this
    # point has almost fully decayed to EPS_END (near-zero exploration).
    # Recomputing it re-opens a bit of exploration for the extended run
    # instead of continuing in a purely exploitative rut.
    num_episodes = start_episode + num_episodes
    EPS_DECAY = num_episodes / 5
    print(
        f"resumed from {CHECKPOINT_PATH} at episode {start_episode}, "
        f"training to {num_episodes} (EPS_DECAY={EPS_DECAY:.0f})"
    )

# %% Action selection, warm-fill, eval, plotting


def select_action(state, i_episode, legal_actions):
    """Returns [[int]] shape [1, 1], always a legal action (illegal ones
    are masked out, matching test.py play behavior)."""
    global steps_done

    sample = random.random()

    # epsilon threshold, decayed per episode (not per step) so exploration
    # pacing is predictable regardless of how long episodes run
    epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * i_episode / EPS_DECAY)

    steps_done += 1

    if sample > epsilon:
        # exploitation: argmax over Q-values, only among legal actions
        with torch.no_grad():  # not training
            q_values = policy_net(state).squeeze(0)
            masked = torch.full_like(q_values, float("-inf"))
            masked[legal_actions] = q_values[legal_actions]
            return masked.argmax().view(1, 1)
    else:
        # exploration: uniform over legal actions only
        action = random.choice(legal_actions)
        return torch.tensor([[action]], device=device, dtype=torch.long)


WARM_FILL_TRANSITIONS = 5_000
"""Replay-buffer size to reach before training resumes after a checkpoint
load. The buffer isn't saved in checkpoints, so a resumed run starts with
an empty one; without this, the first optimizer steps would train on a
handful of highly correlated recent episodes."""


def warm_fill_buffer(i_episode: int):
    fill_env = Engine()
    fill_start = time.time()
    while len(memory) < WARM_FILL_TRANSITIONS:
        fill_state = (
            torch.tensor(
                fill_env.reset(), dtype=torch.float32, device=device
            ).unsqueeze(0)
            / WIN_EXPONENT
        )
        while not fill_env.done:
            # same epsilon-greedy behavior policy as training, so the
            # collected transitions match the distribution training sees
            action = select_action(fill_state, i_episode, fill_env.legal_actions())
            board, reward, done, _ = fill_env.step(int(action.item()))
            reward_t = torch.tensor([reward / REWARD_SCALE], device=device)
            next_state = (
                None
                if done
                else torch.tensor(
                    board, dtype=torch.float32, device=device
                ).unsqueeze(0)
                / WIN_EXPONENT
            )
            memory.push(fill_state, action, next_state, reward_t)
            fill_state = next_state
    print(
        f"warm-filled replay buffer with {len(memory)} transitions "
        f"in {time.time() - fill_start:.1f}s"
    )


EVAL_EVERY = 50
EVAL_EPISODES = 5


def eval_policy() -> float:
    """Plays EVAL_EPISODES games greedily (epsilon=0) and returns the mean
    score -- the real measure of the learned weights."""
    eval_env = Engine()
    scores = []
    for _ in range(EVAL_EPISODES):
        eval_env.reset()
        while not eval_env.done:
            eval_state = (
                torch.tensor(
                    eval_env.board.flatten(), dtype=torch.float32, device=device
                ).unsqueeze(0)
                / WIN_EXPONENT
            )
            legal = eval_env.legal_actions()
            with torch.no_grad():
                q_values = policy_net(eval_state).squeeze(0)
                masked = torch.full_like(q_values, float("-inf"))
                masked[legal] = q_values[legal]
            eval_env.step(int(masked.argmax().item()))
        scores.append(eval_env.score)
    return sum(scores) / len(scores)


def plot_scores():
    """Renders the result plot. Called once, after training -- inline
    notebook backends display the figure automatically at cell end."""
    plt.figure(figsize=(9, 5))
    plt.title("Result")
    plt.xlabel("Episode")
    plt.ylabel("Score")

    scores = torch.tensor(episode_scores, dtype=torch.float)
    plt.plot(scores.numpy(), label="training (with exploration)")

    # Plot 100-episode moving average
    if len(scores) >= 100:
        avg = scores.unfold(0, 100, 1).mean(1)
        avg = torch.cat((torch.zeros(99), avg))
        plt.plot(avg.numpy(), label="100-ep moving avg")

    # The true learning curve: greedy-policy evals, no exploration noise
    if eval_history:
        eval_x, eval_y = zip(*eval_history)
        plt.plot(eval_x, eval_y, marker="o", label="eval (greedy policy)")

    plt.legend()


# %% Optimization step (Double DQN)


def optimize_model():
    """Randomly sample past experiences to adjust weights."""
    # replay memory not enough experiences, can't create a batch
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Mask of non-final states (final = episode ended after it)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Q(s_t, a_t) for the actions actually taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # V(s_t+1), Double DQN style: policy_net picks the best action,
    # target_net evaluates it -- cancels plain DQN's overestimation bias.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        best_actions = policy_net(non_final_next_states).argmax(1, keepdim=True)
        next_state_values[non_final_mask] = (
            target_net(non_final_next_states).gather(1, best_actions).squeeze(1)
        )
    # Bellman target: Q(s, a) = r + gamma * Q_target(s', argmax_policy)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


# %% Checkpointing


def save_checkpoint(episode):
    # 1. Ctrl+C / kernel interrupt is ignored during the write so it can't
    #    truncate the file. 2. Temp file + atomic rename so even a hard
    #    kill leaves the previous checkpoint intact.
    previous_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        tmp_path = CHECKPOINT_PATH + ".tmp"
        torch.save(
            {
                "policy_net": policy_net.state_dict(),
                "target_net": target_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "steps_done": steps_done,
                "episode_scores": episode_scores,
                "eval_history": eval_history,
                "episode": episode,
            },
            tmp_path,
        )
        os.replace(tmp_path, CHECKPOINT_PATH)
    finally:
        signal.signal(signal.SIGINT, previous_handler)


# No periodic auto-save in the notebook edition: in a long notebook
# session the kernel stays alive the whole time, so the only moments that
# matter are the end of the run and an explicit interrupt -- both save
# below. (train.py keeps periodic saves to survive hard kills.)

# %% Training loop
# Interrupting the kernel (notebook stop button / Ctrl+C) is safe: the
# except block saves a checkpoint for test.py. Since the kernel keeps the
# nets, replay buffer, and episode counters alive, re-running this cell
# simply continues training in place -- no resume machinery needed.

# A resumed run starts with an empty buffer; refill it with the current
# policy before any optimizer steps. Fresh runs skip this -- they build
# the buffer naturally during high-epsilon early episodes.
if start_episode > 0 and len(memory) == 0:
    warm_fill_buffer(start_episode)

start_time = time.time()

interrupted = False
completed_episode = start_episode  # last fully finished episode, for interrupt save

try:
    pbar = tqdm(
        range(start_episode, num_episodes),
        initial=start_episode,
        total=num_episodes,
        desc="training",
    )
    for i_episode in pbar:
        # Initialize the environment and get its state, exponents scaled
        # to [0, 1] -- small MLPs train more stably on unit-scale inputs
        state = (
            torch.tensor(env.reset(), dtype=torch.float32, device=device).unsqueeze(0)
            / WIN_EXPONENT
        )
        for t in count():  # infinite counter
            action = select_action(state, i_episode, env.legal_actions())
            board, reward, done, score = env.step(int(action.item()))
            # normalized so Q-targets stay in SmoothL1Loss's quadratic zone
            reward = torch.tensor([reward / REWARD_SCALE], device=device)

            if done:
                next_state = None
            else:
                next_state = (
                    torch.tensor(board, dtype=torch.float32, device=device).unsqueeze(0)
                    / WIN_EXPONENT
                )

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy=learner network)
            optimize_model()

            # Soft update of the target network, in place: θ′ ← τθ + (1−τ)θ′
            with torch.no_grad():
                for target_param, param in zip(
                    target_net.parameters(), policy_net.parameters()
                ):
                    target_param.mul_(1 - TAU).add_(param, alpha=TAU)

            if done:
                episode_scores.append(score)
                epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(
                    -1.0 * i_episode / EPS_DECAY
                )
                # Live stats go on the progress bar itself instead of one
                # printed line per episode -- per-episode prints flood
                # hosted notebook consoles (Kaggle/Colab) with thousands
                # of lines over a run.
                recent = episode_scores[-25:]
                pbar.set_postfix(
                    score=score,
                    steps=steps_done,
                    avg25=round(sum(recent) / len(recent)),
                    eps=round(epsilon, 2),
                )
                break

        completed_episode = i_episode + 1

        # Periodic greedy-policy evaluation: the real learning curve.
        # In a notebook this also refreshes the inline plot.
        if completed_episode % EVAL_EVERY == 0:
            eval_mean = eval_policy()
            eval_history.append((completed_episode, eval_mean))
            tqdm.write(
                f"  eval @ episode {completed_episode}: "
                f"mean score={eval_mean:.0f} over {EVAL_EPISODES} games"
            )

    # normal completion: checkpoint the final episodes so a later run with
    # a higher num_episodes resumes from here
    save_checkpoint(completed_episode)
except KeyboardInterrupt:
    # interrupting the kernel saves progress; a partial in-flight episode
    # is discarded, resume picks up from the last completed one
    interrupted = True
    save_checkpoint(completed_episode)
    tqdm.write(
        f"interrupted at episode {completed_episode}, "
        f"checkpoint saved to {CHECKPOINT_PATH}"
    )

# so re-running this cell continues episode numbering (and epsilon decay)
# instead of restarting from episode 0 with fresh exploration
start_episode = completed_episode

# %% Results
elapsed = time.time() - start_time
if episode_scores:
    best_score = max(episode_scores)
    result = "beat the game!" if best_score >= 18_000 else "did not beat the game"
    print(
        f"{'Stopped' if interrupted else 'Complete'} in {elapsed / 60:.1f} min "
        f"(best score={best_score}, {result})"
    )
    plot_scores()
    plt.show()  # no-op inline; opens the window when run as a plain script

if not interrupted:
    torch.save(policy_net.state_dict(), "dqn_2048.pth")
    print("saved model to dqn_2048.pth")
