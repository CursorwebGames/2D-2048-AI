import random
import statistics

from engine import Engine


def select_action(env: Engine) -> int:
    return random.choice(env.legal_actions())


def play_episode(env: Engine) -> int:
    env.reset()
    while not env.done:
        env.step(select_action(env))
    return env.score


def benchmark(num_episodes: int = 100) -> list[int]:
    env = Engine()
    return [play_episode(env) for _ in range(num_episodes)]


def print_stats(scores: list[int], label: str = "random"):
    print(f"[{label}] episodes={len(scores)}")
    print(f"[{label}] best={max(scores)}")
    print(f"[{label}] worst={min(scores)}")
    print(f"[{label}] mean={statistics.mean(scores):.1f}")
    print(f"[{label}] median={statistics.median(scores)}")


def main(num_episodes: int = 100):
    print_stats(benchmark(num_episodes))


if __name__ == "__main__":
    main()
