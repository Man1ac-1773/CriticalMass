import numpy as np
from collections import deque
from chain_reaction import ChainReactionGame
import time
import heapq
ROWS, COLS = 12, 8
MAX_BRANCHES = 20 # cap for branches in early game
MAX_TT_SIZE = 200000 # cap for transposition table
_GAME = ChainReactionGame() # dummy, for utility

# precompute with _GAME to reduce overhead 
CAPACITY = np.zeros((ROWS, COLS), dtype=np.int8)
for r in range(ROWS):
    for c in range(COLS):
        CAPACITY[r][c] = _GAME.capacity(r, c)

INV_CAPACITY = 1/CAPACITY

# neighbours as a dict 
NEIGHBOURS = {}
for r in range(ROWS):
    for c in range(COLS):
        NEIGHBOURS[(r,c)] = _GAME.neighbors(r, c)


# ==== ZOBRIST ==== 
MAX_ORBS = 4          # some kind of safe upper limit

# zobrist is a 4 dimensional tensor 
# used to store every possible position and combination
ZOBRIST = np.random.randint(0, 2**64, size=(ROWS, COLS, 2, MAX_ORBS+1), dtype=np.uint64)
EMPTY_HASH = np.random.randint(0, 2**64, size=(ROWS, COLS), dtype=np.uint64)

def zobrist_cell(r, c, owner, orb):
    if owner == -1:
        return EMPTY_HASH[r, c]
    clamped = min(int(orb), CAPACITY[r,c])
    return ZOBRIST[r, c, owner, clamped]

def compute_hash(owners, orbs):
    h = np.uint64(0)
    for r in range(ROWS):
        for c in range(COLS):
            h ^= zobrist_cell(r, c, owners[r,c], orbs[r,c])
    return h

TT = {}
# hash -> (depth, value, flag, move)
# ==== ====

# === HELPER FUNCTION ===
# evaluate a potential move and return score of it
def score_move(owners, orbs, move, player_id : int):
    # assume am receiving a valid move
    final_score = 0 
    r, c = move
    if orbs[r][c] >= CAPACITY[r][c] - 1:
        final_score += 300
        # check if this explosion is near opponents 
        for nr, nc in NEIGHBOURS[(r,c)]:
            if owners[nr][nc] == 1 - player_id:
               if orbs[nr][nc] >= CAPACITY[nr][nc] - 1:
                   final_score += 200
    else : 
        for nr, nc in NEIGHBOURS[(r,c)]:
            if owners[nr][nc] == 1 - player_id:
                if orbs[nr][nc] >= CAPACITY[nr][nc] - 1:
                    final_score -= 50 
                    # penalise for opponents near explosion but I am not
    
    if CAPACITY[r][c] == 2:
        final_score += 100 # reward corner
    elif CAPACITY[r][c] == 3:
        final_score += 50 # somewhat reward edge
    return final_score

# return moves sorted by move_score
def get_ordered_moves(owners, orbs, moves, player_id: int):
    return heapq.nlargest(
                MAX_BRANCHES, 
                moves, 
                lambda m : score_move(owners, orbs, m, player_id) 
                )

# Convert current board to two numpy arrays, owners and orbs
def state_to_numpy(state) : 
    owners = np.full((ROWS, COLS), -1, dtype=np.int8)
    orbs = np.zeros((ROWS, COLS), dtype=np.int16)

    for r in range(ROWS):
        for c in range(COLS):
            owner, orbs_cell = state[r][c]
            if owner is not None:
                owners[r,c] = owner
                orbs[r, c] = orbs_cell
    return owners, orbs


# inplace editing wrt move and updating hash 
def make_move(owners, orbs, hash_key, player, move):
    changes = []
    h = hash_key
    
    def apply_cell(r, c, new_owner, new_orb):
        nonlocal h
        old_owner = owners[r, c]
        old_orb = orbs[r, c]
        changes.append((r, c, old_owner, old_orb))
        h ^= zobrist_cell(r, c, old_owner, old_orb)
        owners[r, c] = new_owner
        orbs[r, c] = new_orb
        h ^= zobrist_cell(r, c, new_owner, new_orb)

    r, c = move
    apply_cell(r, c, player, orbs[r, c] + 1)
    
    queue = deque()
    if orbs[r, c] >= CAPACITY[r, c]:
        queue.append((r, c))
    
    while queue:
        cr, cc = queue.popleft()
        
        if orbs[cr, cc] < CAPACITY[cr, cc]:
            continue
            
        # MAJOR CHANGE
        # win check — stop if opponent wiped out
        if np.sum(owners == 1 - player) == 0:
            break
            
        cap = CAPACITY[cr, cc]
        exploding_owner = owners[cr, cc]
        remaining = orbs[cr, cc] - cap
        
        if remaining > 0:
            apply_cell(cr, cc, exploding_owner, remaining)
            if remaining >= cap:
                queue.append((cr, cc))
        else:
            apply_cell(cr, cc, -1, 0)
        
        for nr, nc in NEIGHBOURS[(cr, cc)]:
            apply_cell(nr, nc, exploding_owner, orbs[nr, nc] + 1)
            if orbs[nr, nc] == CAPACITY[nr, nc]:  # match engine
                queue.append((nr, nc))            # shitty design, honestly
    
    return changes, h

# undo a move
def undo_move(owners, orbs, changes):
    for r, c, old_owner, old_orb in reversed(changes):
        owners[r, c] = old_owner
        orbs[r, c] = old_orb


# quick moves, vectorized. returns (np.int64(r), np.uint64(c))
def get_valid_moves(owners, player_id : int):
    mask = (owners == player_id) | (owners == -1)
    return list(zip(*np.where(mask))) 

def check_winner(owners, player_id : int):
    a = np.sum(owners == player_id)
    b = np.sum(owners == 1 - player_id)
    if a > 0 and b == 0 : return True
    elif a == 0 and b > 0 : return False
    else : return None

# === ===


# Evaluate how good the position is 
def evaluate(owners, orbs, player_id):
    my_cells = (owners == player_id)
    opp_cells = (owners == 1 - player_id)
    cell_diff = np.sum(my_cells) - np.sum(opp_cells)
    orb_diff = np.sum(orbs[my_cells]) - np.sum(orbs[opp_cells])

    volatility = orbs * INV_CAPACITY # minor improvement 
    volatility_diff = np.sum(volatility[my_cells]) - np.sum(volatility[opp_cells])

    threat_score = 0 # how much offensive pressure I am exerting
    danger_score = 0 # how much offensive pressure is being exerted on me
    for r in range(ROWS):
        for c in range(COLS):
            owner, _orbs = owners[r][c], orbs[r][c]
            if owner != -1:
                if (owner == player_id) :
                    if CAPACITY[r][c] - _orbs <= 1:
                        # I own this, add to threat
                        for r1, c1 in NEIGHBOURS[(r,c)]:
                            if owners[r1][c1] != -1 :
                                if owners[r1][c1] == 1 - player_id:
                                    # belongs to opponent
                                    threat_score+=1

                elif owner == 1 - player_id:
                   # I don't own, add to dange
                    if CAPACITY[r][c] - _orbs <= 1:
                        for r1, c1 in NEIGHBOURS[(r,c)]:
                            if owners[r1][c1] != -1 :
                                if owners[r1][c1] == player_id:
                                    # belongs to me 
                                    danger_score+=1
    
    # everything counted, now score
    cell_weight = 1.0 
    _cell_score = cell_weight * (cell_diff/96)
    # 96 is magic number representing hypothetical max difference
    orbs_owned_weight = 1.5
    _orb_score = orbs_owned_weight * (orb_diff/384) 
    # magic number 384 represents all cells full with one particular type, max limit
    vol_weight = 1.5
    _vol_score = vol_weight * (volatility_diff/96)
    # magic number 96 comes from all cells at max volatility
    # for each cell which is my neighbour, my threat can be max one per cell
    # what should my max be for scaling?
    threat_score /= 96; danger_score /= 96 
    _threat_weight = 2.5; _danger_weight = 2.5
    threat_score *= _threat_weight; danger_score *= _danger_weight
    final_score = _cell_score + _orb_score + _vol_score - danger_score + threat_score
    return final_score


# Do minimax yayy
def minimax(owners, orbs, hash_key ,player_id, depth, alpha, beta, maximizing, start_time):
    # time constraint
    if (time.time() - start_time >= 0.95):
        return evaluate(owners, orbs, player_id)
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
    win = check_winner(owners, player_id)
    if win is not None:
        if win: 
            return 10000
        else: return -10000
    # base condition
    if depth == 0:
        val = evaluate(owners, orbs, player_id)
        TT[hash_key] = (depth, val, "EXACT", None)
        return val
    
    opponent = 1 - player_id
    current_player = player_id if maximizing else opponent
    alpha_orig = alpha
    beta_orig = beta
    # IMP : EVALUATING MOVES WRT CURRENT PLAYER
    moves = get_valid_moves(owners, current_player)
    tt_move = None
    if entry and entry[3] in moves:
        tt_move = entry[3]
        moves.remove(entry[3])
    moves = get_ordered_moves(owners, orbs, moves, current_player)
    if tt_move : moves.insert(0, tt_move)
    best_move = None
    if maximizing:
        
        best = float('-inf')
        for move in moves:
            changes, inc_hash = make_move(owners, orbs, hash_key, current_player, move)
            score = minimax(owners, orbs, inc_hash, player_id, depth-1, alpha, beta, False, start_time)
            undo_move(owners, orbs, changes)
            if (score > best):
                best = score
                best_move = move
            alpha = max(alpha, best)
            if (beta <= alpha):
                break
    else : 
        best = float('inf') # worst for maximizer. 
        # enemy always playing best possible moves for himself,
        for move in moves:
            changes, inc_hash = make_move(owners, orbs, hash_key, current_player, move)
            score = minimax(owners, orbs, inc_hash, player_id, depth-1, alpha, beta, True, start_time)
            undo_move(owners, orbs, changes)
            if score < best:
                best = score
                best_move = move
            beta = min(beta, best)
            if beta <= alpha : break
    
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
    owners, orbs = state_to_numpy(state) 
    TT.clear()
    global MAX_BRANCHES
    best_move = None
    start_time = time.time()
    root_hash = compute_hash(owners, orbs)
    
    for depth in range(1, 10):
        if time.time() - start_time > 0.9:
            print(f"Depth reached : {depth}")
            break
        if (depth >= 4):
            MAX_BRANCHES = 8
        elif depth == 3:
            MAX_BRANCHES = 12 
        else : 
            MAX_BRANCHES = 20
        moves = get_valid_moves(owners, player_id)
        moves = get_ordered_moves(owners, orbs,moves,  player_id)
        best_score = float('-inf')
        current_best = None
        for move in moves:
            changes, inc_hash = make_move(owners, orbs, root_hash, player_id, move)
            score = minimax(owners, orbs, inc_hash, player_id, depth, float('-inf'), float('inf'), False, start_time)
            undo_move(owners, orbs, changes)
            if score > best_score : 
                best_score = score
                current_best = move
        best_move = current_best
        

    elapsed = time.time() - start_time 
    print(f"Moves available : {len(moves)}, Time elapsed : {elapsed}")
    return best_move


