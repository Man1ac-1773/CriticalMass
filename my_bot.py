import numpy as np
from chain_reaction import ChainReactionGame 
ROWS, COLS = 12, 8
_GAME = ChainReactionGame() # dummy, for utility
# === HELPER FUNCTION FOR NUMPY ===
# Convert current board to two numpy arrays, owners and orbs
def state_to_numpy(state) : 
    owners = np.full((ROWS, COLS), -1, dtype=np.int8)
    orbs = np.zeros((ROWS, COLS), dtype=np.int8)

    for r in range(len(state)):
        for c in range(len(state[r])):
            owner, orbs_cell = state[r][c]
            if owner is not None:
                owners[r][c] = owner
                orbs[r][c] = orbs_cell
    return owners, orbs

# Convert numpy representation of board to state 
def numpy_to_state(owners, orbs):
    state = []
    for r in range(ROWS):
        layer = []
        for c in range(COLS):
            if owners[r][c] == -1 : 
                layer.append((None, orbs[r][c]))
            else :
                layer.append((owners[r][c], orbs[r][c]))
            
        state.append(layer)
    return state

def apply_move_numpy(owners, orbs, player_id, move):
    state = numpy_to_state(owners, orbs)
    game = ChainReactionGame()
    game.board = state
    game.moves_played = {0 : 1, 1 : 1}
    game.apply_move(player_id, move)
    new_owners, new_orbs = state_to_numpy(game.board)
    return new_owners, new_orbs

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
    cell_diff = np.sum(owners == player_id)-np.sum(owners == 1 - player_id)
    orb_diff = 0
    volatility_diff = 0.0
    threat_score = 0 # how much offensive pressure I am exerting
    danger_score = 0 # how much offensive pressure is being exerted on me
    for r in range(ROWS):
        for c in range(COLS):
            owner, _orbs = owners[r][c], orbs[r][c]
            if owner != -1:
                if owner == player_id:
                    orb_diff += _orbs
                    volatility_diff += _orbs/_GAME.capacity(r,c)
                else :
                    orb_diff -= _orbs
                    volatility_diff -= _orbs/_GAME.capacity(r,c)

                
                if (owner == player_id) :
                    if _orbs/_GAME.capacity(r,c) >= 0.75:
                        # I own this, add to threat
                        for r1, c1 in _GAME.neighbors(r,c):
                            if owners[r1][c1] != -1 :
                                if owners[r1][c1] == 1 - player_id:
                                    # belongs to opponent
                                    threat_score+=1

                elif owner == 1- player_id:
                   # I don't own, add to danger
                    if _orbs/_GAME.capacity(r,c) >= 0.75:
                        for r1, c1 in _GAME.neighbors(r,c):
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
    threat_score /= 384; danger_score /= 384 # magic numbers from hypothetical max
    _threat_weight = 1.5; _danger_weight = 1.5
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
    moves = get_valid_moves(owners, current_player) 
    if maximizing:
        best = float('-inf')
        for move in moves:
            owners_copy, orbs_copy = owners.copy(), orbs.copy()
            apply_move_numpy(owners_copy, orbs_copy, player_id, move)
            score = minimax(owners_copy, orbs_copy, player_id, depth-1, alpha, beta, False)
            best = max(best, score)
            alpha = max(alpha, best)
            if beta <= alpha : break
        return best
    else : 
        worst = float('inf')
        for move in moves:
            owners_copy, orbs_copy = owners.copy(), orbs.copy()
            apply_move_numpy(owners_copy, orbs_copy, player_id, move)
            score = minimax(owners_copy, orbs_copy, player_id, depth-1, alpha, beta, True)
            worst = min(worst, score)
            beta = min(beta, worst)
            if beta <= alpha : break
        return worst



# actual function called
def get_move(state , player_id : int):
    owners, orbs = state_to_numpy(state) 
    best_move = None
    best_score = float('-inf')
    for move in get_valid_moves(owners, player_id):
        owners_copy, orbs_copy = owners.copy(), orbs.copy()
        apply_move_numpy(owners_copy, orbs_copy, player_id, move)
        score = minimax(owners_copy, orbs_copy, player_id, 3, float('-inf'), float('inf'), False)
        if score > best_score:
            best_score = score
            best_move = move
    return best_move
