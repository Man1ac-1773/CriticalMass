import numpy as np
from collections import deque
from chain_reaction import ChainReactionGame
import time
import heapq
# TELEMETRY
NODES_EVAL = 0
# ---
ROWS, COLS = 12, 8
CELLS = ROWS * COLS
MAX_BRANCHES = 25 # cap for branches in early game
MAX_TT_SIZE = 200000 # cap for transposition table
_GAME = ChainReactionGame() # dummy, for utility

# precompute with _GAME to reduce overhead 
CAPACITY = [0] * CELLS 
INV_CAPACITY = [0.0] * CELLS 
NEIGHBOURS = [() for _ in range (CELLS)]

for r in range(ROWS):
    for c in range(COLS):
        idx = r*COLS + c
        CAPACITY[idx] = _GAME.capacity(r,c)
        INV_CAPACITY[idx] = 1.0/CAPACITY[idx]
        n_list = []
        for nr, nc in _GAME.neighbors(r,c):
            n_list.append(nr* COLS + nc)
            NEIGHBOURS[idx] = tuple(n_list)

# ==== ZOBRIST ==== 
MAX_ORBS = 4          # some kind of safe upper limit

# zobrist is a 3 dimensional tensor 
# used to store every possible position and combination
ZOBRIST = np.random.randint(0, 2**64, size=(CELLS, 3, MAX_ORBS+1), dtype=np.uint64)

def zobrist_cell(idx : int, owner : int, orb : int):
     o_idx = 2 if owner == -1 else owner
     clamped = min(orb, 4)
     return ZOBRIST[idx, o_idx, clamped]


def compute_hash(owners, orbs):
    h = np.uint64(0)
    for cell in range(CELLS):
        h ^= zobrist_cell(cell, owners[cell], orbs[cell])
    return h

TT = {}
# hash -> (depth, value, flag, move)
# ==== ====

# === KILLER ====
MAX_SEARCH_DEPTH = 20
killer_moves = {
    0: [[None, None] for _ in range(MAX_SEARCH_DEPTH)],
    1: [[None, None] for _ in range(MAX_SEARCH_DEPTH)]
}

# ==== ====


# === HELPER FUNCTION ===
# weights depending on ownership
CELL_W = 1.0 / 96.0
ORB_W = 1.5 / 384.0
VOL_W = 1.5 / 96.0
CORNER_W = 0.015
EDGE_W = 0.005
THREAT_W = 0.03   # heavy penalty 
SUPPORT_W = 0.01  # bonus 

def get_base_value(idx, orb_count, owner, root_player):
    """ The isolated value of the cell ignoring its neighbors. """
    if owner == -1: return 0.0
    
    val = CELL_W + (orb_count * ORB_W) + (orb_count * INV_CAPACITY[idx] * VOL_W)
    
    # Real estate value
    if CAPACITY[idx] == 2: val += CORNER_W
    elif CAPACITY[idx] == 3: val += EDGE_W
    
    return val if owner == root_player else -val

def evaluate_edge(idx1, o1, orb1, idx2, o2, orb2, root_player):
    """ The tactical relationship (Threat/Danger/Support) between two adjacent cells. 
        It assumes that the cells passed are adjacent. That safety is left upto caller."""
    if o1 == -1 or o2 == -1: return 0.0

    c1_crit = (orb1 == CAPACITY[idx1] - 1)
    c2_crit = (orb2 == CAPACITY[idx2] - 1)
    val = 0.0

    if o1 != o2:
        # Enemy Edge
        # If cell 1 is critical, it threatens cell 2 (Good for owner 1)
        if c1_crit:
            val += THREAT_W if o1 == root_player else -THREAT_W
        # If cell 2 is critical, it threatens cell 1 (Good for owner 2)
        if c2_crit:
            val += THREAT_W if o2 == root_player else -THREAT_W
    else:
        # Friendly Edge (Domino Effect)
        if c1_crit and c2_crit:
            val += SUPPORT_W if o1 == root_player else -SUPPORT_W

    return val

# evaluate a potential move and return score of it
def score_move(owners, orbs, move_idx, player_id : int):
    # assume am receiving a valid move
    final_score = 0 
    if orbs[move_idx] >= CAPACITY[move_idx] - 1:
        final_score += 300
        # check if this explosion is near opponents 
        for neighbor in NEIGHBOURS[move_idx]:
            if owners[neighbor] == 1 - player_id:
                if orbs[neighbor] >= CAPACITY[neighbor] - 1:
                   final_score += 200
    else : 
        for neighbor in NEIGHBOURS[move_idx]:
            if owners[neighbor] == 1 - player_id:
                if orbs[neighbor] >= CAPACITY[neighbor] - 1:
                    final_score -= 50 
                    # penalise for opponents near explosion but I am not
    
    if CAPACITY[move_idx] == 2:
        final_score += 100 # reward corner
    elif CAPACITY[move_idx] == 3:
        final_score += 50 # somewhat reward edge
    return final_score

# return moves sorted by move_score
def get_ordered_moves(owners, orbs, moves: list[int], player_id: int, depth : int, tt_move = None):
    ordered = []
    moves_set = set(moves) 
    
    # 1. TT Move is king
    if tt_move in moves_set:
        ordered.append(tt_move)
        moves_set.remove(tt_move)
        
    # 2. Player-specific Killer Moves for this depth
    for km in killer_moves[player_id][depth]:
        if km in moves_set:
            ordered.append(km)
            moves_set.remove(km)
            
    # 3. Evaluate and sort the rest using fast heapq
    remaining = list(moves_set)
    if remaining:
        remaining.sort(key = lambda m : score_move(owners, orbs, m, player_id), reverse=True) 
        ordered.extend(remaining)
        
    return ordered

# Convert current board to two arrays, owners and orbs
def state_to_1d(state) : 
    owners = [-1]  * CELLS
    orbs = [0] * CELLS
    for r in range(ROWS):
        for c in range(COLS):
            owner, orbs_cell = state[r][c]
            if owner is not None:
                idx = r* COLS + c
                owners[idx] = owner
                orbs[idx] = orbs_cell
    return owners, orbs


# inplace editing wrt move and updating hash 
def make_move(owners : list[int], orbs : list[int], hash_key : np.uint64, current_player : int, move_idx : int, root_player : int, state_info : tuple[int, int ,int]) -> tuple[list[tuple[int, int, int]], np.uint64 , float, tuple[int, int, int]] : 
    changes = []
    h = hash_key
    score_delta = 0.0
    p0, p1, total_orbs = state_info
    def apply_cell(idx, new_owner, new_orb):
        nonlocal h, score_delta, p0, p1
        old_owner = owners[idx]
        old_orb = orbs[idx]
        score_delta -= get_base_value(idx, old_orb, old_owner, root_player)
        for n in NEIGHBOURS[idx]:
            score_delta -= evaluate_edge(idx, old_owner, old_orb, n, owners[n], orbs[n], root_player)
        changes.append((idx, old_owner, old_orb))
        h ^= zobrist_cell(idx, old_owner, old_orb)
        owners[idx] = new_owner
        orbs[idx] = new_orb
        h ^= zobrist_cell(idx, new_owner, new_orb)
        
        # O(1) Counter updates
        if old_owner != new_owner:
            if old_owner == 0: p0 -= 1
            elif old_owner == 1: p1 -= 1
            if new_owner == 0: p0 += 1
            elif new_owner == 1: p1 += 1
        score_delta += get_base_value(idx, new_orb, new_owner, root_player)
        for n in NEIGHBOURS[idx]:
            score_delta += evaluate_edge(idx, old_owner, old_orb, n, owners[n], orbs[n], root_player)
    
    apply_cell(move_idx, current_player, orbs[move_idx] + 1)
    
    queue = deque()
    if orbs[move_idx] >= CAPACITY[move_idx]:
        queue.append(move_idx)
    
    while queue:
        curr = queue.popleft()
        
        if orbs[curr] < CAPACITY[curr]:
            continue
            
        # O(1) Win check mid-explosion using our local counters
        if current_player == 0 and p1 == 0: break
        if current_player == 1 and p0 == 0: break
           
        cap = CAPACITY[curr]
        exploding_owner = owners[curr]
        remaining = orbs[curr] - cap
        
        if remaining > 0:
            apply_cell(curr, exploding_owner, remaining)
            if remaining >= cap:
                queue.append(curr)
        else:
            apply_cell(curr, -1, 0)
        
        for n_idx in NEIGHBOURS[curr]:
            apply_cell(n_idx, exploding_owner, orbs[n_idx] + 1)
            if orbs[n_idx] == CAPACITY[n_idx]:  # match engine
                queue.append(n_idx)            # shitty design, honestly
    
    return changes, h, score_delta, (p0, p1, total_orbs + 1)

# undo a move
def undo_move(owners, orbs, changes):
    for idx, old_owner, old_orb in reversed(changes):
        owners[idx] = old_owner
        orbs[idx] = old_orb


# new 1d version, faster than numpy (tested)
def get_valid_moves(owners, player_id : int):
    return [i for i in range(CELLS) if owners[i] == player_id or owners[i] == -1] 

# change to state info. O(1) eval instead of O(N)
def check_winner(state_info, root_player):
    p0_count, p1_count, total_orbs = state_info
    
    if total_orbs < 2: 
        return None
        
    if p0_count > 0 and p1_count == 0: 
        return 10000 if root_player == 0 else -10000
    elif p1_count > 0 and p0_count == 0: 
        return 10000 if root_player == 1 else -10000
    return None

# === ===



# Do minimax yayy
def minimax(owners, orbs, hash_key ,player_id, depth, alpha, beta, maximizing, start_time, current_score, state_info):
    global NODES_EVAL
    NODES_EVAL += 1
    # time constraint
    if (time.time() - start_time >= 0.95):
        return current_score 
    # TT lookup
    entry = TT.get(hash_key)
    if entry:
        stored_depth, val, flag, best_move = entry
        if stored_depth >= depth:
            if flag == "EXACT":
                return val
            elif flag == "LOWER": # more aggro pruning
                alpha = max(alpha, val)
            elif flag == "UPPER":
                beta = min(beta, val)
            if alpha >= beta:
                return val
    # terminal
    win_score = check_winner(state_info, player_id)
    if win_score is not None:
        return win_score    
    # base condition
    if depth == 0:
        TT[hash_key] = (depth, current_score, "EXACT", None)
        return current_score 
    
    opponent = 1 - player_id
    current_player = player_id if maximizing else opponent
    alpha_orig = alpha
    beta_orig = beta
    # IMP : EVALUATING MOVES WRT CURRENT PLAYER
    moves = get_valid_moves(owners, current_player)
    tt_move = None
    if entry and entry[3] in moves:
        tt_move = entry[3]
    # handles insertion and order of tt_move
    moves = get_ordered_moves(owners, orbs, moves, current_player, depth, tt_move)
    best_move = None
    if maximizing:
        best = float('-inf')
        for i, move in enumerate(moves):
            changes, inc_hash, d_score, next_state = make_move(owners, orbs, hash_key, current_player, move, player_id, state_info)
            new_score = current_score + d_score
            
            needs_full_search = True
            is_killer = move in killer_moves[current_player][depth]
            
            # LMR Condition for Maximizer
            # If we are deep enough, past the first 3 promising moves, and it's not a killer move
            if depth >= 3 and i >= 3 and not is_killer:
                # 1. Do a shallow search (depth - 2)
                score = minimax(owners, orbs, inc_hash, player_id, depth - 2, alpha, beta, False, start_time, new_score, next_state)
                
                # 2. If the shallow search surprisingly beats alpha, our ordering was wrong! 
                # We must research it at full depth to get the exact value.
                if score <= alpha:
                    needs_full_search = False
                    
            if needs_full_search:
                score = minimax(owners, orbs, inc_hash, player_id, depth - 1, alpha, beta, False, start_time, new_score, next_state)
                
            undo_move(owners, orbs, changes)
            
            if score > best:
                best = score
                best_move = move
            alpha = max(alpha, best)
            if beta <= alpha:
                break
                
    else: 
        best = float('inf') # worst for maximizer
        for i, move in enumerate(moves):
            changes, inc_hash, d_score, next_state = make_move(owners, orbs, hash_key, current_player, move, player_id, state_info)
            new_score = current_score + d_score 
            needs_full_search = True
            is_killer = move in killer_moves[current_player][depth]
            
            # LMR Condition for Minimizer
            if depth >= 3 and i >= 3 and not is_killer:
                # 1. Shallow search (depth - 2)
                score = minimax(owners, orbs, inc_hash, player_id, depth - 2, alpha, beta, True, start_time, new_score, next_state)
                
                # 2. Minimizer wants to push the score DOWN. 
                # If the shallow score drops below beta, it's a dangerous move and needs a full search.
                if score >= beta: 
                    needs_full_search = False
                    
            if needs_full_search:
                score = minimax(owners, orbs, inc_hash, player_id, depth - 1, alpha, beta, True, start_time, new_score, next_state)
                
            undo_move(owners, orbs, changes)
            
            if score < best:
                best = score
                best_move = move
            beta = min(beta, best)
            if beta <= alpha:
                # caused a cutoff, therefore killer
                if killer_moves[current_player][depth][0] != move:
                    killer_moves[current_player][depth][1] = killer_moves[current_player][depth][0]
                    killer_moves[current_player][depth][0] = move
                break 
    
    # store TT
    if best <= alpha_orig:
        flag = "UPPER"
    elif best >= beta_orig:
        flag = "LOWER"
    else : 
        flag = "EXACT"
    TT[hash_key] = (depth, best, flag, best_move)
    if len(TT) > MAX_TT_SIZE:
        TT.clear() # prevent blowup of memory
    return best


# actual function called
def get_move(state, player_id : int):
    owners, orbs = state_to_1d(state) 
    TT.clear()
    global MAX_BRANCHES, NODES_EVAL
    NODES_EVAL = 0
    
    best_move = None
    
    start_time = time.time()
    

    root_hash = compute_hash(owners, orbs)
    root_score = 0.0
    

    for curr in range(CELLS):
        root_score += get_base_value(curr, orbs[curr], owners[curr], player_id)
        for n in NEIGHBOURS[curr]:
            if n > curr:
                root_score += evaluate_edge(curr, owners[curr], orbs[curr], n, owners[n], orbs[n], player_id)
    
    root_state = (owners.count(0), owners.count(1), sum(orbs))

    for depth in range(1, 15):
        if time.time() - start_time > 0.9:
            break
        
        moves = get_valid_moves(owners, player_id)
        if best_move is None and moves : best_move = moves[0] # fallback
        entry = TT.get(root_hash)
        tt_move = None
        if entry and entry[3] in moves:
            tt_move = entry[3]
        
        moves = get_ordered_moves(owners, orbs,moves, player_id, depth, tt_move)
        best_score = float('-inf')
        current_best = None
        
        for move in moves:
            changes, inc_hash, d_score, next_state = make_move(owners, orbs, root_hash, player_id, move, player_id, root_state)
            score = minimax(owners, orbs, inc_hash, player_id, depth, float('-inf'), float('inf'), False, start_time, root_score + d_score, next_state)
            undo_move(owners, orbs, changes)
            if score > best_score : 
                best_score = score
                current_best = move
        best_move = current_best
    

    elapsed = time.time() - start_time 
    nps = int(NODES_EVAL/elapsed) if elapsed > 0 else 0 
    print(f"Max Depth: {depth} | Nodes: {NODES_EVAL:<8} | NPS: {nps}")
    return (best_move // COLS, best_move % COLS)

