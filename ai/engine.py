import random
from enum import IntEnum

import numpy as np


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


MERGE_BONUS = 5
""" flat reward per merge, on top of the log-scale merge value """

WIN_EXPONENT = 11
""" tile exponent for 2048 (1 << 11 == 2048) """
WIN_BONUS = 200
""" one-time reward for first reaching the 2048 tile, mirrors the loss penalty """


def _line_monotonicity_penalty(line: np.ndarray) -> int:
    diffs = np.diff(line.astype(np.int64))
    increasing = int(diffs[diffs > 0].sum())
    decreasing = int(-diffs[diffs < 0].sum())
    # whichever direction the line "fights against" is the penalty
    return min(increasing, decreasing)


MONOTONICITY_PERFECT_BONUS = 3
""" flat reward when every row/column is perfectly monotonic """


def monotonicity_bonus(board: np.ndarray) -> int:
    """Zero-centered reward for how monotonic rows/columns are: a flat
    positive bonus when every row and column is perfectly monotonic,
    otherwise a shrunk negative penalty proportional to choppiness."""
    penalty = sum(_line_monotonicity_penalty(line) for line in board)
    penalty += sum(_line_monotonicity_penalty(line) for line in board.T)
    if penalty == 0:
        return MONOTONICITY_PERFECT_BONUS
    return -(penalty // 2)


CORNER_BONUS = 5
""" flat reward when the highest tile on the board sits in a corner """


def corner_bonus(board: np.ndarray) -> int:
    max_val = board.max()
    if max_val == 0:
        return 0
    corners = (board[0, 0], board[0, -1], board[-1, 0], board[-1, -1])
    return CORNER_BONUS if max_val in corners else 0


class Engine:
    """2048 game engine with a gym-like step/reset API for training agents."""

    def __init__(self, size: int = 4, seed: int | None = None):
        self.size = size
        self.rng = random.Random(seed)
        self.reset()

    def reset(self) -> np.ndarray:
        """Shape (16,)"""
        self.board = np.zeros((self.size, self.size), dtype=np.int64)
        self.score = 0
        self.done = False
        self.won = False
        self.spawn_tile()
        self.spawn_tile()
        return self.board.flatten()

    def step(self, action: int) -> tuple[np.ndarray, int, bool, int]:
        """Apply an action. Returns (board = (16,), reward, done, score)."""
        if self.done:
            raise RuntimeError("game is over, call reset()")

        moved, reward, score_delta, merge_count = self.move(Action(action))

        # only here do we tweak the raw reward
        if moved:
            self.spawn_tile()
            self.score += score_delta
            reward += merge_count * MERGE_BONUS
            reward += monotonicity_bonus(self.board)
            reward += corner_bonus(self.board)
            if not self.won and np.any(self.board >= WIN_EXPONENT):
                reward += WIN_BONUS
                self.won = True
        else:
            # penalize did not move
            reward -= 50

        self.done = not self.has_moves()

        if self.done:
            # penalize losing the game
            reward -= 200

        return self.board.flatten(), reward, self.done, self.score

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
        score_delta = 0
        merge_count = 0
        for row in rotated:
            merged_row, row_reward, row_score, row_merges = self.merge_row(row)
            new_rows.append(merged_row)
            reward += row_reward
            score_delta += row_score
            merge_count += row_merges
        new_board = np.array(new_rows, dtype=np.int64)
        moved = not np.array_equal(new_board, rotated)

        if not dry_run and moved:
            self.board = self.from_basis(action, new_board)

        return moved, reward, score_delta, merge_count

    def to_basis(self, action: Action, board: np.ndarray) -> np.ndarray:
        # Change to basis so the move direction is always "left".
        # (read-only view: move() never mutates `rotated`, so no copy needed here)
        if action == Action.LEFT:
            return board
        if action == Action.RIGHT:
            return np.fliplr(board)
        if action == Action.UP:
            return board.T
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
        """Returns (merged_row, log reward, real score delta, merge count)."""
        values = [v for v in row if v != 0]
        merged = []
        reward = 0
        score_delta = 0
        merge_count = 0
        i = 0
        while i < len(values):
            if i + 1 < len(values) and values[i] == values[i + 1]:
                new_exp = values[i] + 1
                merged.append(new_exp)
                reward += new_exp
                score_delta += 1 << new_exp
                merge_count += 1
                i += 2
            else:
                merged.append(values[i])
                i += 1
        merged += [0] * (len(row) - len(merged))
        return np.array(merged, dtype=np.int64), reward, score_delta, merge_count

    def has_moves(self) -> bool:
        # Cheap vectorized check: any empty cell, or any adjacent equal pair
        # in a row or column (which a merge could exploit). Avoids running
        # 4 full dry-run moves (each doing a Python merge_row per row).
        if np.any(self.board == 0):
            return True
        if np.any(self.board[:, :-1] == self.board[:, 1:]):
            return True
        if np.any(self.board[:-1, :] == self.board[1:, :]):
            return True
        return False

    def __str__(self) -> str:
        return "\n".join(
            " ".join(f"{(1 << v) if v else 0:5d}" for v in row) for row in self.board
        )
