import argparse
import os
import random

try:
    import msvcrt
except ImportError:
    msvcrt = None

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine import Action, Engine, WIN_EXPONENT
from random_agent import benchmark, print_stats

CHECKPOINT_PATH = "checkpoint_2048.pth"

n_actions = 4
n_observations = 16  # 4x4 board of exponents, scaled to [0, 1]

# ANSI background colors per tile exponent (1=2, 2=4, ... 11=2048, ...)
TILE_COLORS = {
    0: "\033[48;5;238m",  # empty
    1: "\033[48;5;223m",
    2: "\033[48;5;222m",
    3: "\033[48;5;215m",
    4: "\033[48;5;209m",
    5: "\033[48;5;208m",
    6: "\033[48;5;202m",
    7: "\033[48;5;220m",
    8: "\033[48;5;226m",
    9: "\033[48;5;190m",
    10: "\033[48;5;154m",
    11: "\033[48;5;51m",
}
DEFAULT_COLOR = "\033[48;5;93m"
RESET = "\033[0m"


ARROWS: dict[Action | None, str] = {
    Action.UP: "↑",
    Action.DOWN: "↓",
    Action.LEFT: "←",
    Action.RIGHT: "→",
}


def render_board(board, action: "Action | None" = None) -> str:
    lines = []
    for row in board:
        cells = []
        for exp in row:
            value = (1 << exp) if exp else 0
            color = TILE_COLORS.get(int(exp), DEFAULT_COLOR)
            cells.append(f"{color}{value:^7}{RESET}")
        lines.append("".join(cells))

    width = 7 * len(board[0])
    arrow = ARROWS.get(action)
    if action in (Action.UP, Action.DOWN):
        mid = len(lines) // 2
        lines[mid] += f"  {arrow}"
    elif action in (Action.LEFT, Action.RIGHT):
        indent = " " * (width // 2 - 1)
        lines.append(f"{indent}{arrow}")

    return "\n".join(lines)


HIDDEN_SIZE = 128  # must match train.py or checkpoints won't load


class DQN(nn.Module):
    def __init__(self, n_observations: int, n_actions: int):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, HIDDEN_SIZE)
        self.layer2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.layer3 = nn.Linear(HIDDEN_SIZE, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


def read_key() -> str:
    """Blocks until left/right/enter/quit is pressed, returns one of those."""
    if msvcrt is not None:
        while True:
            ch = msvcrt.getch()
            if ch in (b"\x00", b"\xe0"):
                ch2 = msvcrt.getch()
                if ch2 == b"K":
                    return "left"
                if ch2 == b"M":
                    return "right"
                continue
            if ch == b"\r":
                return "right"
            if ch in (b"q", b"Q"):
                return "quit"
    else:
        raw = input("[enter/right]=next  a=back  q=quit: ").strip().lower()
        if raw == "q":
            return "quit"
        if raw == "a":
            return "left"
        return "right"


def select_action(policy_net, state, legal_actions):
    with torch.no_grad():
        q_values = policy_net(state).squeeze(0)
    # mask out illegal actions so we never waste a move
    masked = torch.full_like(q_values, float("-inf"))
    masked[legal_actions] = q_values[legal_actions]
    return int(masked.argmax().item())


def dqn_play_episode(policy_net, env: Engine, device) -> int:
    env.reset()
    while not env.done:
        # same [0, 1] input scaling as training -- the net only ever saw
        # scaled inputs
        state = (
            torch.tensor(
                env.board.flatten(), dtype=torch.float32, device=device
            ).unsqueeze(0)
            / WIN_EXPONENT
        )
        action = select_action(policy_net, state, env.legal_actions())
        env.step(action)
    return env.score


def greedy_play_episode(env: Engine) -> int:
    """Plays whichever move has the best immediate shaped reward -- the
    reward function acting as a one-step policy, no learning and no
    lookahead. This is the bar the DQN has to clear: its only advantage
    over this baseline is learned multi-step value (lookahead via
    bootstrapped Q-values)."""
    env.reset()
    while not env.done:
        legal = env.legal_actions()
        action = max(legal, key=lambda a: env.evaluate_action(a))
        env.step(action)
    return env.score


def run_baseline(num_episodes: int):
    """Greedy vs random comparison -- no checkpoint needed, so it can run
    before any training exists. Greedy meaningfully beating random is the
    sanity check that the shaped reward encodes real 2048 strategy."""
    env = Engine()
    greedy_scores = [greedy_play_episode(env) for _ in range(num_episodes)]
    random_scores = benchmark(num_episodes)

    print_stats(greedy_scores, label="greedy")
    print_stats(random_scores, label="random")

    greedy_mean = sum(greedy_scores) / len(greedy_scores)
    random_mean = sum(random_scores) / len(random_scores)
    ratio = greedy_mean / random_mean if random_mean else float("inf")
    print(f"\ngreedy={greedy_mean:.1f}  random={random_mean:.1f}  ({ratio:.2f}x)")
    return greedy_mean, random_mean


def run_benchmark(policy_net, device, num_episodes: int):
    env = Engine()
    dqn_scores = [
        dqn_play_episode(policy_net, env, device) for _ in range(num_episodes)
    ]
    print_stats(dqn_scores, label="dqn")

    greedy_mean, random_mean = run_baseline(num_episodes)

    dqn_mean = sum(dqn_scores) / len(dqn_scores)
    print(
        f"dqn={dqn_mean:.1f}  greedy={greedy_mean:.1f}  random={random_mean:.1f}"
    )
    if dqn_mean > greedy_mean:
        print("dqn beats greedy: the net has learned lookahead beyond the reward function")
    else:
        print("dqn does NOT beat greedy: the net hasn't learned more than the reward function already says")


def reward_sanity_check(num_episodes: int):
    """Play random-agent episodes and report the resulting reward/score
    distribution, so reward-shaping changes in engine.py can be sanity
    checked without needing a trained policy.

    Goals: reward exponentially 1024 and 1024 merge is very very good

    """
    env = Engine()
    rewards = []
    final_scores = []
    for _ in range(num_episodes):
        env.reset()
        while not env.done:
            legal = env.legal_actions()
            action = random.choice(legal) if legal else 0
            _, reward, _, _ = env.step(action)
            rewards.append(reward)
        final_scores.append(env.score)

    print(f"ran {num_episodes} random-agent episodes\n")
    print(
        f"final score: min={min(final_scores)}  max={max(final_scores)}  "
        f"avg={sum(final_scores) / len(final_scores):.1f}"
    )
    print(
        f"reward:      min={min(rewards)}  max={max(rewards)}  "
        f"avg={sum(rewards) / len(rewards):.2f}"
    )


def draw(frame, is_history: bool, is_done: bool):
    os.system("cls" if os.name == "nt" else "clear")
    move = f"  move={frame['action'].name}" if frame["action"] is not None else ""
    reward = f"  reward={frame['reward']}" if frame["reward"] is not None else ""
    print(f"step {frame['step']}  score={frame['score']}{reward}{move}")
    print(render_board(frame["board"], frame["action"]))

    if is_done:
        print("\ngame over -- left=back  q=quit")
    elif is_history:
        print("\n(viewing history) left=back  right/enter=forward  q=quit")
    else:
        print("\nleft=back  right/enter=next move  q=quit")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=int,
        default=0,
        metavar="N",
        help="run N episodes headlessly and compare dqn vs random, instead of the interactive viewer",
    )
    parser.add_argument(
        "--reward-check",
        type=int,
        default=0,
        metavar="N",
        help="run N random-agent episodes and report reward/score stats (no checkpoint needed)",
    )
    parser.add_argument(
        "--baseline",
        type=int,
        default=0,
        metavar="N",
        help="run N episodes of greedy (best immediate shaped reward) vs random (no checkpoint needed)",
    )
    args = parser.parse_args()

    if args.reward_check:
        reward_sanity_check(args.reward_check)
        return

    if args.baseline:
        run_baseline(args.baseline)
        return

    device = torch.device("cpu")
    policy_net = DQN(n_observations, n_actions).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    policy_net.load_state_dict(checkpoint["policy_net"])
    policy_net.eval()
    print(f"loaded {CHECKPOINT_PATH} (trained through episode {checkpoint['episode']})")

    if args.benchmark:
        run_benchmark(policy_net, device, args.benchmark)
        return

    env = Engine()
    env.reset()

    history = [
        {
            "board": env.board.copy(),
            "score": 0,
            "step": 0,
            "action": None,
            "reward": None,
        }
    ]
    idx = 0

    draw(history[idx], is_history=False, is_done=False)

    while True:
        key = read_key()
        if key == "quit":
            break

        if key == "left":
            idx = max(0, idx - 1)
        elif key == "right":
            if idx < len(history) - 1:
                idx += 1
            elif not env.done:
                legal_actions = env.legal_actions()
                # same [0, 1] input scaling as training
                state = (
                    torch.tensor(
                        env.board.flatten(), dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    / WIN_EXPONENT
                )
                action = select_action(policy_net, state, legal_actions)
                _, reward, _, score = env.step(action)
                history.append(
                    {
                        "board": env.board.copy(),
                        "score": score,
                        "step": len(history),
                        "action": Action(action),
                        "reward": reward,
                    }
                )
                idx += 1

        draw(
            history[idx],
            is_history=idx < len(history) - 1,
            is_done=env.done and idx == len(history) - 1,
        )

    print(f"\nfinal score={env.score}")


if __name__ == "__main__":
    main()
