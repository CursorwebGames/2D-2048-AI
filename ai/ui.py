# UI Test
from engine import Action, Engine

KEY_TO_ACTION = {
    "w": Action.UP,
    "s": Action.DOWN,
    "a": Action.LEFT,
    "d": Action.RIGHT,
}

# tile value -> (fg, bg) ANSI 256-color codes, roughly matching the classic 2048 palette
TILE_COLORS = {
    0: (None, 236),
    2: (232, 230),
    4: (232, 229),
    8: (255, 208),
    16: (255, 202),
    32: (255, 196),
    64: (255, 160),
    128: (232, 220),
    256: (232, 214),
    512: (232, 208),
    1024: (255, 178),
    2048: (255, 172),
}
DEFAULT_COLOR = (255, 91)  # anything past 2048


def colorize(value: int) -> str:
    fg, bg = TILE_COLORS.get(value, DEFAULT_COLOR)
    text = f"{value:^7}" if value else " " * 7
    if fg is None:
        return f"\x1b[48;5;{bg}m{text}\x1b[0m"
    return f"\x1b[38;5;{fg};48;5;{bg}m{text}\x1b[0m"


def render(env: Engine) -> str:
    return "\n".join(
        "".join(colorize((1 << v) if v else 0) for v in row) for row in env.board
    )


def play():
    env = Engine()
    print(render(env))
    print(f"score: {env.score}\n")

    while not env.done:
        raw = input("move (w/a/s/d, q to quit): ").strip().lower()
        if raw == "q":
            break
        if raw not in KEY_TO_ACTION:
            print("invalid input, use w/a/s/d or q")
            continue

        _, reward, done, info = env.step(KEY_TO_ACTION[raw])
        print(render(env))
        print(f"score: {info['score']} (+{reward})\n")

        if done:
            print("game over!")


if __name__ == "__main__":
    play()
