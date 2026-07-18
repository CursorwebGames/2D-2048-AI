# 2048
2048 AI! Race against the AI to see who can solve it faster!
Research to find the best 2048 AI.

## Ideas
* Race different AI models
* Race against the AI
* Big 2048 boards
* Seed boards
* 2048 Hard: always put the worst number at the worst place
* Big 2048

## Models
* CSS Model (top right bottom left)
* Random
* Reinforcement Learning
    * DQN
    * TD
    * CNN DQN?
* Logical (https://stackoverflow.com/questions/22342854/what-is-the-optimal-algorithm-for-the-game-2048/23853848#23853848)

## RL reward design (ai/engine.py, mirrored in src/engine/reward.ts)
The step reward is:

```
reward = merge_reward + Φ(after) − Φ(before)   [+ win/game-over bonuses]
```

* `merge_reward` — log2 of each merged tile, summed, then scaled by
  `MERGE_BONUS_SCALE` so bigger merges are worth proportionally more.
* `Φ` (board potential) — one board-quality score: monotonicity of
  rows/columns + max-tile-in-corner + trapped-tile penalty. Rewarding the
  **difference** across a move (potential-based shaping, Ng et al. 1999)
  means pre-existing mess cancels out, breaking the corner is an explicit
  negative, and the shaping provably doesn't change the optimal policy.
  Φ(after) is measured before the random spawn, so luck is never
  credited/blamed on the move that preceded it.
* Illegal moves are never punished — training and play both mask them out
  (`select_action` in ai/train.py and ai/test.py).
* Network inputs are the 16 tile exponents scaled to [0, 1] (divided by
  `WIN_EXPONENT` = 11). A one-hot encoding (16 cells x 16 classes) was
  tried and performed worse at the ~1000-episode training budget: one-hot
  removes the free numeric generalization scalar inputs give (lessons
  about 32-tiles partially transfer to 64-tiles), and learning every
  (cell, value) pair independently needs far more episodes. Revisit only
  with 10k+ episode runs.

## Credit
Forked from the [🐐 GOAT](https://github.com/gabrielecirulli/2048/tree/master)

## Sources
https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
https://www.ibm.com/think/topics/reinforcement-learning