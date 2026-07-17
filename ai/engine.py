import random
from enum import IntEnum
from functools import lru_cache

import numpy as np


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


MERGE_BONUS_SCALE = 1
""" extra reward per merge, proportional to the merged tile's exponent
(so bigger merges are rewarded more, not just log-scale-compressed) """

WIN_EXPONENT = 11
""" tile exponent for 2048 (1 << 11 == 2048) """
WIN_BONUS = 200
""" one-time reward for first reaching the 2048 tile, mirrors the loss penalty """


MONOTONICITY_PERFECT_BONUS = 3
""" flat reward when every row/column is perfectly monotonic """


def monotonicity_bonus(board: np.ndarray) -> int:
    """Zero-centered reward for how monotonic rows/columns are: a flat
    positive bonus when every row and column is perfectly monotonic,
    otherwise a shrunk negative penalty proportional to choppiness.

    Per line the penalty is min(sum of upward jumps, sum of downward
    jumps) -- whichever direction the line "fights against". Vectorized:
    one diff over rows and one over columns instead of 8 per-line calls.
    """
    row_diffs = np.diff(board, axis=1)
    col_diffs = np.diff(board, axis=0)
    # per-row min(increasing, decreasing), summed over rows
    penalty = int(
        np.minimum(
            np.clip(row_diffs, 0, None).sum(axis=1),
            np.clip(-row_diffs, 0, None).sum(axis=1),
        ).sum()
    )
    # same for columns (lines run down axis 0)
    penalty += int(
        np.minimum(
            np.clip(col_diffs, 0, None).sum(axis=0),
            np.clip(-col_diffs, 0, None).sum(axis=0),
        ).sum()
    )
    if penalty == 0:
        return MONOTONICITY_PERFECT_BONUS
    return -(penalty // 2)


CORNER_WEIGHT = 2
""" corner potential is +/- CORNER_WEIGHT * max tile exponent: uncornering
a 256 (exp 8) swings Phi by 2*2*8=32, while early-game corner placement
stays cheap. A flat bonus let a batch of small merges outweigh
uncornering a big tile, which is strategically catastrophic. """


def corner_bonus(board: np.ndarray) -> int:
    max_val = board.max()
    if max_val == 0:
        return 0
    corners = (board[0, 0], board[0, -1], board[-1, 0], board[-1, -1])
    weight = CORNER_WEIGHT * int(max_val)
    return weight if max_val in corners else -weight


TRAPPED_TILE_PENALTY = 2
""" penalty per tile boxed in by >=2 strictly-higher neighbors (e.g. a
small tile wedged between two bigger equal tiles, blocking their merge) """


def trapped_tile_penalty(board: np.ndarray) -> int:
    """Catches tiles the line-based monotonicity check misses: a tile
    surrounded by bigger neighbors on 2+ sides is boxed in and can't merge
    without first being cleared, regardless of whether its row/column
    reads as "monotonic" overall.

    Vectorized: for each cell, count strictly-higher neighbors via four
    shifted comparisons instead of a Python loop over cells."""
    higher = np.zeros_like(board)
    higher[1:, :] += board[:-1, :] > board[1:, :]  # neighbor above is higher
    higher[:-1, :] += board[1:, :] > board[:-1, :]  # neighbor below is higher
    higher[:, 1:] += board[:, :-1] > board[:, 1:]  # neighbor left is higher
    higher[:, :-1] += board[:, 1:] > board[:, :-1]  # neighbor right is higher
    trapped = (higher >= 2) & (board > 0)
    return -int(trapped.sum()) * TRAPPED_TILE_PENALTY


def board_potential(board: np.ndarray) -> int:
    """Phi(s): a standalone measure of board quality, used for
    potential-based reward shaping (see step()). Higher = better-organized
    board. Composed of the three shape heuristics."""
    return (
        monotonicity_bonus(board)
        + corner_bonus(board)
        + trapped_tile_penalty(board)
    )


@lru_cache(maxsize=None)
def _merge_row_cached(values: tuple) -> tuple:
    """Merge one row leftward; returns (merged tuple, log reward, real
    score delta, merge count).

    Cached: 2048 rows are tiny (4 exponents, each realistically 0-12), so
    the same rows recur constantly across a training run. After warm-up
    this replaces the Python merge loop with a dict lookup, which is the
    single hottest path in the engine (called per row per move, including
    the 4 dry-run moves in legal_actions)."""
    tiles = [v for v in values if v]
    merged = []
    reward = 0
    score_delta = 0
    merge_count = 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            new_exp = tiles[i] + 1
            merged.append(new_exp)
            reward += new_exp
            score_delta += 1 << new_exp
            merge_count += 1
            i += 2
        else:
            merged.append(tiles[i])
            i += 1
    merged += [0] * (len(values) - len(merged))
    return tuple(merged), reward, score_delta, merge_count


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

        # Potential-based reward shaping (Ng et al. 1999): instead of
        # adding the absolute board quality Phi(s') to every move, reward
        # the CHANGE the move caused: Phi(after) - Phi(before). Pre-existing
        # mess the move didn't touch appears in both and cancels out, so a
        # good merge on an otherwise-ugly board isn't dragged down, and
        # breaking the corner / creating a trapped tile is an explicit
        # negative at the moment it happens. Shaping of this form provably
        # doesn't change which policy is optimal. (Textbook form is
        # gamma*Phi(after) - Phi(before); with gamma=0.99 the plain
        # difference is a close approximation.)
        # Phi(before) is the post-spawn board the agent actually acted
        # from; Phi(after) is pre-spawn, so the random spawn's effect on
        # board shape is never credited/blamed on this move -- it lands in
        # the next step's baseline instead.
        phi_before = board_potential(self.board)

        moved, reward, score_delta, merge_count = self.move(Action(action))

        # only here do we tweak the raw reward
        if moved:
            self.score += score_delta
            # Proportional merge bonus: scales with how big the merge(s)
            # were (reward is currently the summed log-reward of the
            # merges), instead of a flat amount regardless of tile size.
            reward += reward * MERGE_BONUS_SCALE
            reward += board_potential(self.board) - phi_before
            if not self.won and np.any(self.board >= WIN_EXPONENT):
                reward += WIN_BONUS
                self.won = True
            self.spawn_tile()
        # No no-move penalty: training and test both mask illegal actions
        # (train.py/test.py select_action), so a no-op step never happens
        # during learning or play; stepping one manually just rewards 0.

        self.done = not self.has_moves()

        if self.done:
            # penalize losing the game
            reward -= 200

        return self.board.flatten(), reward, self.done, self.score

    def legal_actions(self) -> list[int]:
        return [a for a in Action if self.move(a, dry_run=True)[0]]

    def evaluate_action(self, action: int) -> int | None:
        """The shaped reward this action would earn right now, without
        mutating the engine and without the random spawn or terminal check
        (both depend on chance, not on the action). Returns None if the
        action is illegal. Used by greedy/lookahead baselines."""
        sim = Engine(self.size)
        sim.board = self.board.copy()
        sim.won = self.won
        phi_before = board_potential(sim.board)
        moved, reward, _, _ = sim.move(Action(action))
        if not moved:
            return None
        reward += reward * MERGE_BONUS_SCALE
        reward += board_potential(sim.board) - phi_before
        if not sim.won and np.any(sim.board >= WIN_EXPONENT):
            reward += WIN_BONUS
        return reward

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
        # rows go through the memoized merge as plain tuples; np.array at
        # the end stacks them back into a board in one call
        for row in rotated.tolist():
            merged_row, row_reward, row_score, row_merges = _merge_row_cached(
                tuple(row)
            )
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
        merged, reward, score_delta, merge_count = _merge_row_cached(
            tuple(row.tolist())
        )
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
