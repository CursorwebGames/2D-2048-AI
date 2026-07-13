import random
from enum import IntEnum

import numpy as np


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Engine:
    """2048 game engine with a gym-like step/reset API for training agents."""

    def __init__(self, size: int = 4, seed: int | None = None):
        self.size = size
        self.rng = random.Random(seed)
        self.reset()

    def reset(self) -> np.ndarray:
        self.board = np.zeros((self.size, self.size), dtype=np.int64)
        self.score = 0
        self.done = False
        self.spawn_tile()
        self.spawn_tile()
        return self.board.copy()

    def step(self, action: int):
        """Apply an action. Returns (board, reward, done, info)."""
        if self.done:
            raise RuntimeError("game is over, call reset()")

        moved, reward = self.move(Action(action))
        if moved:
            self.spawn_tile()
            self.score += reward

        self.done = not self.has_moves()
        info = {"score": self.score, "moved": moved}
        return self.board.copy(), reward, self.done, info

    def legal_actions(self) -> list[int]:
        return [a for a in Action if self.move(a, dry_run=True)[0]]

    def spawn_tile(self):
        empties = list(zip(*np.where(self.board == 0)))
        if not empties:
            return
        r, c = self.rng.choice(empties)
        self.board[r, c] = 2 if self.rng.random() < 0.1 else 1

    def move(self, action: Action, dry_run: bool = False):
        rotated = self.to_basis(action, self.board)
        new_rows = []
        reward = 0
        for row in rotated:
            merged_row, row_reward = self.merge_row(row)
            new_rows.append(merged_row)
            reward += row_reward
        new_board = np.array(new_rows, dtype=np.int64)
        moved = not np.array_equal(new_board, rotated)

        if not dry_run and moved:
            self.board = self.from_basis(action, new_board)

        return moved, reward

    def to_basis(self, action: Action, board: np.ndarray) -> np.ndarray:
        # Change to basis so the move direction is always "left".
        if action == Action.LEFT:
            return board.copy()
        if action == Action.RIGHT:
            return np.fliplr(board)
        if action == Action.UP:
            return board.T.copy()
        if action == Action.DOWN:
            return np.fliplr(board.T)

    def from_basis(self, action: Action, board: np.ndarray) -> np.ndarray:
        # Inverse of to_basis.
        if action == Action.LEFT:
            return board
        if action == Action.RIGHT:
            return np.fliplr(board)
        if action == Action.UP:
            return board.T
        if action == Action.DOWN:
            return np.fliplr(board).T

    @staticmethod
    def merge_row(row: np.ndarray):
        values = [v for v in row if v != 0]
        merged = []
        reward = 0
        i = 0
        while i < len(values):
            if i + 1 < len(values) and values[i] == values[i + 1]:
                new_exp = values[i] + 1
                merged.append(new_exp)
                reward += 1 << new_exp
                i += 2
            else:
                merged.append(values[i])
                i += 1
        merged += [0] * (len(row) - len(merged))
        return np.array(merged, dtype=np.int64), reward

    def has_moves(self) -> bool:
        if np.any(self.board == 0):
            return True
        for action in Action:
            if self.move(action, dry_run=True)[0]:
                return True
        return False

    def __str__(self) -> str:
        return "\n".join(
            " ".join(f"{(1 << v) if v else 0:5d}" for v in row) for row in self.board
        )
