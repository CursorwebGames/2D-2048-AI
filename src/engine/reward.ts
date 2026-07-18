import type { Grid } from "./grid.ts";

// Mirrors the reward shaping in ai/engine.py, so the in-browser game can
// show what an RL agent would have been rewarded for the same move.

// Extra reward per merge, proportional to the merged tile's exponent
// (so bigger merges are rewarded more, not just log-scale-compressed)
export const MERGE_BONUS_SCALE = 1;
export const MONOTONICITY_PERFECT_BONUS = 3;
// Corner potential is +/- CORNER_WEIGHT * max tile exponent: uncornering a
// 256 (exp 8) swings the potential by 2*2*8=32, while early-game corner
// placement stays cheap. A flat bonus let a batch of small merges outweigh
// uncornering a big tile, which is strategically catastrophic.
export const CORNER_WEIGHT = 2;
export const WIN_BONUS = 200;
export const GAME_OVER_PENALTY = -200;
export const WIN_EXPONENT = 11; // 1 << 11 == 2048

// One labeled term of a reward breakdown, e.g. { label: "merge", value: 7 }
export interface RewardTerm {
    label: string;
    value: number;
}

export interface RewardBreakdown {
    total: number;
    terms: RewardTerm[];
}

// Builds a breakdown from only the terms that actually contributed
// (skips zero-valued terms so the readout stays readable)
export function buildBreakdown(terms: RewardTerm[]): RewardBreakdown {
    const nonZero = terms.filter((term) => term.value !== 0);
    const total = terms.reduce((sum, term) => sum + term.value, 0);
    return { total, terms: nonZero };
}

// Extracts a size x size grid of tile exponents (log2 of value, 0 for empty)
export function exponentGrid(grid: Grid): number[][] {
    const exponents: number[][] = [];

    for (let x = 0; x < grid.size; x++) {
        const row: number[] = [];
        for (let y = 0; y < grid.size; y++) {
            const tile = grid.cells[x][y];
            row.push(tile ? Math.log2(tile.value) : 0);
        }
        exponents.push(row);
    }

    return exponents;
}

function lineMonotonicityPenalty(line: number[]): number {
    let increasing = 0;
    let decreasing = 0;

    for (let i = 0; i < line.length - 1; i++) {
        const diff = line[i + 1] - line[i];
        if (diff > 0) increasing += diff;
        else if (diff < 0) decreasing += -diff;
    }

    // whichever direction the line "fights against" is the penalty
    return Math.min(increasing, decreasing);
}

// Zero-centered reward for how monotonic rows/columns are: a flat positive
// bonus when every row and column is perfectly monotonic, otherwise a
// shrunk negative penalty proportional to choppiness.
export function monotonicityBonus(exponents: number[][]): number {
    const size = exponents.length;
    let penalty = 0;

    for (let x = 0; x < size; x++) {
        penalty += lineMonotonicityPenalty(exponents[x]);
    }
    for (let y = 0; y < size; y++) {
        penalty += lineMonotonicityPenalty(exponents.map((row) => row[y]));
    }

    // / 4 (was / 2): keeps monotonicity a tiebreaker, not a veto -- at / 2
    // the agent would pass up real merges to avoid small shape hits, and
    // merges are what actually score points and clear board space
    return penalty === 0 ? MONOTONICITY_PERFECT_BONUS : -Math.floor(penalty / 4);
}

// Reward when the highest tile sits in a corner, penalty when it doesn't,
// scaled by how big that tile is (see CORNER_WEIGHT)
export function cornerBonus(exponents: number[][]): number {
    const size = exponents.length;
    const maxVal = Math.max(...exponents.map((row) => Math.max(...row)));
    if (maxVal === 0) return 0;

    const corners = [
        exponents[0][0],
        exponents[0][size - 1],
        exponents[size - 1][0],
        exponents[size - 1][size - 1],
    ];

    const weight = CORNER_WEIGHT * maxVal;
    return corners.includes(maxVal) ? weight : -weight;
}

// Penalty per tile boxed in by >=2 strictly-higher neighbors (e.g. a small
// tile wedged between two bigger equal tiles, blocking their merge)
export const TRAPPED_TILE_PENALTY = 2;

// Catches tiles the line-based monotonicity check misses: a tile
// surrounded by bigger neighbors on 2+ sides is boxed in and can't merge
// without first being cleared, regardless of whether its row/column
// reads as "monotonic" overall.
export function trappedTilePenalty(exponents: number[][]): number {
    const size = exponents.length;
    let penalty = 0;

    for (let x = 0; x < size; x++) {
        for (let y = 0; y < size; y++) {
            const value = exponents[x][y];
            if (value === 0) continue;

            let higher = 0;
            for (const [dx, dy] of [
                [-1, 0],
                [1, 0],
                [0, -1],
                [0, 1],
            ]) {
                const nx = x + dx;
                const ny = y + dy;
                if (
                    nx >= 0 &&
                    nx < size &&
                    ny >= 0 &&
                    ny < size &&
                    exponents[nx][ny] > value
                ) {
                    higher++;
                }
            }

            if (higher >= 2) penalty += TRAPPED_TILE_PENALTY;
        }
    }

    return -penalty;
}

// Phi(s): a standalone measure of board quality, used for potential-based
// reward shaping -- moves are rewarded for the CHANGE in potential they
// cause (Phi(after) - Phi(before)), not the absolute board quality, so
// pre-existing mess a move didn't touch cancels out of its reward.
// Mirrors board_potential() in ai/engine.py.
export function boardPotential(exponents: number[][]): number {
    return (
        monotonicityBonus(exponents) +
        cornerBonus(exponents) +
        trappedTilePenalty(exponents)
    );
}
