import numpy as np
from chain_reaction import ChainReactionGame
import time
import heapq
ROWS, COLS = 12, 8
MAX_BRANCHES = 20 # cap for branches in early game
_GAME = ChainReactionGame() # dummy, for utility
# precompute with _GAME to reduce overhead 
CAPACITY = np.zeros((ROWS, COLS), dtype=np.int8)
for r in range(ROWS):
    for c in range(COLS):
        CAPACITY[r][c] = _GAME.capacity(r, c)

# neighbours as a dict 
NEIGHBOURS = {}
for r in range(ROWS):
    for c in range(COLS):
        NEIGHBOURS[(r,c)] = _GAME.neighbors(r, c)

# === HELPER FUNCTION ===
# evaluate a potential move and return score of it
def score_move(owners, orbs, move, player_id):
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
def get_ordered_moves(owners, orbs, player_id):
    moves = get_valid_moves(owners, player_id)
    best_moves = heapq.nlargest(MAX_BRANCHES, moves)
    return best_moves



# Convert current board to two numpy arrays, owners and orbs
def state_to_numpy(state) : 
    owners = np.full((ROWS, COLS), -1, dtype=np.int8)
    orbs = np.zeros((ROWS, COLS), dtype=np.int8)

    for r in range(ROWS):
        for c in range(COLS):
            owner, orbs_cell = state[r][c]
            if owner is not None:
                owners[r,c] = owner
                orbs[r, c] = orbs_cell
    return owners, orbs

# Convert numpy representation of board to state 
def numpy_to_state(owners, orbs):
    state = []
    for r in range(ROWS):
        layer = []
        for c in range(COLS):
            if owners[r, c] == -1 : 
                layer.append((None, orbs[r,c]))
            else :
                layer.append((owners[r,c], orbs[r,c]))
            
        state.append(layer)
    return state


# simulating moves and board without calling ChainReactionGame
def apply_move_fast(owners, orbs, player, move):
    owners = owners.copy()
    orbs = orbs.copy()
    stack = [move]
    r, c = move
    owners[r, c] = player
    orbs[r, c] += 1

    if orbs[r, c] < CAPACITY[r, c]:
        return owners, orbs

    stack.append((r, c))

    while stack:
        cr, cc = stack.pop()

        cur_count = orbs[cr, cc]
        cap = CAPACITY[cr, cc]

        if cur_count < cap:
            continue

        exploding_owner = owners[cr, cc]

        remaining = cur_count - cap

        if remaining > 0:
            orbs[cr, cc] = remaining
            if remaining >= cap:
                stack.append((cr, cc))
        else:
            orbs[cr, cc] = 0
            owners[cr, cc] = -1  # empty

        for nr, nc in NEIGHBOURS[(cr, cc)]:
            owners[nr, nc] = exploding_owner
            orbs[nr, nc] += 1

            # ONLY push when it reaches threshold (important optimization)
            if orbs[nr, nc] >= CAPACITY[nr, nc]:
                stack.append((nr, nc))

    return owners, orbs


def get_valid_moves(owners, player_id):
    moves = []
    for r in range(ROWS):
        for c in range(COLS):
            if owners[r][c] == player_id or owners[r][c] == -1:
                moves.append((r,c))
    return moves

def check_winner(owners, player_id):
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

    volatility = orbs / CAPACITY
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
def minimax(owners, orbs, player_id, depth, alpha, beta, maximizing):
    win = check_winner(owners, player_id)
    if win is not None:
        if win: 
            return 10000
        else: return -10000
    if depth == 0:
        return evaluate(owners, orbs, player_id)
    
    opponent = 1 - player_id
    current_player = player_id if maximizing else opponent
    # IMP : EVALUATING MOVES WRT CURRENT PLAYER
    moves = get_ordered_moves(owners, orbs, current_player) 
    if maximizing:
        best = float('-inf')
        for move in moves:
            owners_copy, orbs_copy = apply_move_fast(owners, orbs, current_player, move)
            score = minimax(owners_copy, orbs_copy, player_id, depth-1, alpha, beta, False)
            best = max(best, score)
            alpha = max(alpha, best)
            if beta <= alpha : break
        return best
    else : 
        worst = float('inf') # worst for maximizer. 
        # enemy always playing best possible moves
        for move in moves:
            owners_copy, orbs_copy = apply_move_fast(owners, orbs, current_player, move)
            score = minimax(owners_copy, orbs_copy, player_id, depth-1, alpha, beta, True)
            worst = min(worst, score)
            beta = min(beta, worst)
            if beta <= alpha : break
        return worst



# actual function called
def get_move(state, player_id : int):
    owners, orbs = state_to_numpy(state) 
    best_move = None
    best_score = float('-inf')
    t0 = time.time()
    depth = 3 
    moves = get_ordered_moves(owners, orbs, player_id)
    for move in moves:
        owners_copy, orbs_copy = apply_move_fast(owners, orbs, player_id, move)
        score = minimax(owners_copy, orbs_copy, player_id, depth, float('-inf'), float('inf'), False)
        if score > best_score:
            best_score = score
            best_move = move
    elapsed = time.time() - t0
    print(f"Depth = {depth}, {len(moves)} moves, Time elapsed : {elapsed:.3f}s")
    return best_move
