# Myok bot - Strategy and algorithm 

## Overview

This bot plays Chain Reaction on a 12×8 grid using adversarial search. It was built from scratch in Python over one day, evolving from a naive minimax implementation to a heavily optimized engine with incremental evaluation, Zobrist hashing, a transposition table, killer move heuristics, and late move reduction.

The core philosophy: **search deeper than the opponent, evaluate positions honestly, and don't waste time on moves that don't matter.**

---

## 1. Game Representation

### 1D Flat Arrays

The board is stored as two flat Python lists of length 96 (12×8):

```python
owners[idx] = -1 | 0 | 1   # -1 = empty
orbs[idx]   = int           # orb count at this cell
```

Cell index is computed as `idx = r * COLS + c`. This is faster than a 2D list for sequential access and avoids repeated row/column indexing overhead.

### Precomputed Lookup Tables

At module load time, three tables are computed once and reused forever:

```python
CAPACITY[idx]     # critical mass: 2 (corner), 3 (edge), 4 (middle)
INV_CAPACITY[idx] # 1.0 / CAPACITY[idx] — avoids division in evaluate
NEIGHBOURS[idx]   # tuple of neighbour indices
```

These replace all calls to `game.capacity()` and `game.neighbors()` during search, eliminating Python method dispatch overhead.

---

## 2. Move Simulation — `make_move` / `undo_move`

Instead of copying the board at every node (which was the original bottleneck), the bot makes moves **in place** and records a changelog for undoing.

### `make_move`

```python
changes = []  # list of (idx, old_owner, old_orb)
```

Every cell modification is logged. After the move and all chain reactions are resolved, `undo_move` replays the log in reverse to restore the board.

### Chain Reaction BFS

Explosions are resolved with a queue (FIFO), matching the reference engine's behaviour exactly:

```
place orb → if >= capacity, enqueue
while queue:
    dequeue cell
    if still < capacity, skip (stale entry)
    check for win — stop if opponent wiped
    explode: distribute 1 orb to each neighbour
    for each neighbour: if == capacity, enqueue
```

The `==` check on neighbour enqueue (not `>=`) matches the engine and avoids duplicate queue entries.

### Incremental Scoring

`make_move` also computes `score_delta` — the change in board evaluation caused by this move. This means at depth 0, evaluation is O(1) (just use `current_score + accumulated_delta`) instead of O(96). This is a significant speedup since depth-0 nodes are the most common.

### O(1) Win Detection

Player cell counts `(p0, p1)` are maintained as local variables inside `make_move` and updated incrementally whenever ownership changes. Win detection mid-explosion is a single integer comparison rather than a board scan.

---

## 3. Zobrist Hashing

Zobrist hashing produces a unique 64-bit integer for each board position, used as a key into the transposition table.

### Construction

```python
ZOBRIST[cell_idx, owner_idx, orb_count]  # shape: (96, 3, 5)
```

Three owner states: 0, 1, and -1 (mapped to index 2). Orb count is clamped to 4 (max capacity); cells above capacity are transient and strategically equivalent.

### Incremental Update

When a cell changes from `(old_owner, old_orb)` to `(new_owner, new_orb)`:

```python
h ^= ZOBRIST[idx, old_owner_idx, old_orb]  # remove old
h ^= ZOBRIST[idx, new_owner_idx, new_orb]  # add new
```

XOR is its own inverse, so `undo_move` automatically restores the hash by replaying the same operations. No separate hash undo is needed; the hash is passed by value and recomputed per branch.

---

## 4. Transposition Table

```python
TT = {}  # hash → (depth, value, flag, best_move)
```

Before evaluating any node, the bot checks if this position has already been evaluated at equal or greater depth. Three entry types:

- `EXACT` : the stored value is the true minimax value
- `LOWER` : the true value is at least this (caused a beta cutoff)
- `UPPER` : the true value is at most this (caused an alpha cutoff)

These are used to tighten the alpha-beta window before searching:

```python
if flag == "LOWER": alpha = max(alpha, val)
if flag == "UPPER": beta = min(beta, val)
if alpha >= beta: return val  # immediate cutoff
```

The best move stored in the TT is used for move ordering at that node. The move that was best before is tried first next time.

Terminal scores (wins/losses) are not stored in the TT to prevent hash collisions corrupting win detection across different game states that hash to the same value.

The table is capped at 200,000 entries and cleared when full (a known limitation) 

---

## 5. Search — Minimax with Alpha-Beta Pruning

### Alpha-Beta

The core algorithm is minimax with alpha-beta pruning:

- **Alpha**: the best score the maximizer is guaranteed so far. Starts at `-inf`.
- **Beta**: the best score the minimizer is guaranteed so far. Starts at `+inf`.
- **Prune** when `beta <= alpha` ; the current branch cannot affect the final result.

Alpha-beta does not change the result of minimax. It eliminates branches that are provably irrelevant, allowing deeper search in the same time.

### Iterative Deepening

Rather than searching at a fixed depth, the bot searches depth 1, then depth 2, then depth 3, and so on until the time limit (0.9 seconds) is reached:

```python
for depth in range(1, 15):
    if time_elapsed > 0.9: break
    best_move = search_at(depth)
```

Benefits: always has a valid answer (from the previous completed depth), uses all available time, and earlier depths populate the TT and inform move ordering for deeper searches.

If a depth iteration is aborted mid-way (time ran out), that partial result is discarded. Only completed depth results update `best_move`.

### Win Score with Depth Bonus

Wins are scored as `10000 + depth` (wins sooner = better score). This ensures the bot prefers forced checkmates in fewer moves over slower wins, rather than treating all wins as equivalent.

---

## 6. Move Ordering

Good move ordering is critical for alpha-beta efficiency. The bot orders moves in four tiers:

**Tier 1: TT move** — the best move from a previous search of this position. Most likely to be the best move again.

**Tier 2: Killer moves** — moves that caused a beta cutoff at this depth in a sibling node. Stored as two slots per player per depth. Killers are not evaluated, just tried early.

**Tier 3: Scored moves** — remaining moves ranked by `score_move`:
- `+300` if placing here causes an immediate explosion
- `+200` if that explosion lands on an opponent cell also near critical mass (chain potential)
- `+100` for corners, `+50` for edges (strategic real estate)
- `-50` if near an opponent about to explode but we are not (danger)

**Tier 4: Branching cap** — only the top `MAX_BRANCHES` (25) moves are explored. This is the most important speedup early game when there are 90+ valid moves. Move ordering ensures the pruned moves are the weakest ones.

---

## 7. Late Move Reduction (LMR)

For moves past position 3 in the ordered list (that are not killer moves), at depth >= 3, the bot first does a shallow search at `depth - 2`:

```python
if depth >= 3 and i >= 3 and not is_killer:
    score = minimax(..., depth - 2, ...)
    if score > alpha:  # surprisingly good, research at full depth
        score = minimax(..., depth - 1, ...)
    # else: trust the shallow result, skip full search
```

The intuition: later moves in a well-ordered list are probably bad. Do a quick check first. Only spend full search time if the move turns out to be better than expected.

---

## 8. Evaluation Function

The evaluation function scores how good the current position is for the root player. It is computed incrementally via `score_delta` in `make_move`.

Each cell contributes:

```python
val = CELL_W                              # owning any cell
    + orb_count * ORB_W                   # more orbs = more pressure
    + orb_count * INV_CAPACITY * VOL_W    # volatility: high orbs relative to capacity
    + CORNER_W (if corner)                # strategic real estate bonus
    + EDGE_W (if edge)
```

The cell value is positive if owned by the root player, negative if owned by the opponent.

Weights are normalized so the total score is bounded roughly in `[-1, 1]` for non-terminal positions, clearly separated from terminal scores of `±10000`.

**What this captures:**
- Cell ownership advantage
- Orb count advantage (more orbs = more potential energy)
- Volatility: cells near critical mass are more dangerous and valuable
- Positional advantage: corners are easier to hold and harder to infiltrate

**What this misses (known limitations):**
- Threat/danger from near-critical neighbours (was in earlier versions, removed for speed)
- Chain reaction potential (hard to evaluate without simulation)
- Connectivity and board control patterns

---

## 9. Performance Summary

| Optimization | Impact |
|---|---|
| 1D flat arrays over 2D list of tuples | ~2x faster board access |
| Precomputed CAPACITY / NEIGHBOURS | Eliminates method dispatch in hot loop |
| make/undo over deepcopy | ~10x faster node transitions |
| Incremental score delta | O(1) evaluation at depth 0 |
| O(1) win detection | Eliminates O(96) scan mid-explosion |
| Incremental Zobrist hash | O(changes) hash update vs O(96) recompute |
| Transposition table | Avoids re-evaluating duplicate positions |
| Move ordering (TT + killer + score) | More pruning → deeper effective search |
| Branching cap (MAX_BRANCHES=25) | Prevents exponential blowup early game |
| Iterative deepening | Uses full time budget, always has a valid answer |
| LMR | Reduces nodes spent on likely-bad moves |

Typical performance: depth 5, ~40,000-50,000 nodes per second, within 1 second per move.

---

