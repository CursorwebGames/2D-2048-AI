import os

try:
    import msvcrt
except ImportError:
    msvcrt = None

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine import Action, Engine

CHECKPOINT_PATH = "checkpoint_2048.pth"

n_actions = 4
n_observations = 16

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


class DQN(nn.Module):
    def __init__(self, n_observations: int, n_actions: int):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

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
    device = torch.device("cpu")
    policy_net = DQN(n_observations, n_actions).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    policy_net.load_state_dict(checkpoint["policy_net"])
    policy_net.eval()
    print(f"loaded {CHECKPOINT_PATH} (trained through episode {checkpoint['episode']})")

    env = Engine()
    env.reset()

    history = [{"board": env.board.copy(), "score": 0, "step": 0, "action": None, "reward": None}]
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
                state = torch.tensor(
                    env.board.flatten(), dtype=torch.float32, device=device
                ).unsqueeze(0)
                action = select_action(policy_net, state, legal_actions)
                _, reward, _, info = env.step(action)
                history.append(
                    {
                        "board": env.board.copy(),
                        "score": info["score"],
                        "step": len(history),
                        "action": Action(action),
                        "reward": reward,
                    }
                )
                idx += 1

        draw(history[idx], is_history=idx < len(history) - 1, is_done=env.done and idx == len(history) - 1)

    print(f"\nfinal score={env.score}")


if __name__ == "__main__":
    main()
