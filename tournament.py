import os
import glob
import time
import importlib.util
from chain_reaction import ChainReactionGame

def load_bot(filepath):
    """Dynamically loads a bot module from a filepath."""
    module_name = os.path.basename(filepath)[:-3]
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_move

def play_match(bot0_name, bot0_func, bot1_name, bot1_func, rows=12, cols=8, max_turns=1000):
    """Runs a single game and returns the winner (0, 1, or -1 for tie/crash)."""
    game = ChainReactionGame(rows=rows, cols=cols)
    bots = {0: bot0_func, 1: bot1_func}

    for turn in range(max_turns):
        player = turn % 2
        
        try:
            start_time = time.time()
            move = bots[player](game.get_state(), player)
            elapsed = time.time() - start_time
            
            # Strict tournament time enforcement
            if elapsed > 1.05: # Slight buffer for system noise
                print(f"[{bot0_name} vs {bot1_name}] Player {player} timed out ({elapsed:.3f}s)!")
                return 1 - player
                
            game.apply_move(player, move)
            
        except Exception as e:
            print(f"[{bot0_name} vs {bot1_name}] Player {player} crashed: {e}")
            return 1 - player # Opponent wins by default

        winner = game.check_winner()
        if winner is not None:
            return winner

    # Tiebreaker logic from your original script
    state = game.get_state()
    counts = {0: 0, 1: 0}
    for row in state:
        for owner, orb_count in row:
            if owner in (0, 1) and orb_count > 0:
                counts[owner] += 1
                
    if counts[0] > counts[1]: return 0
    elif counts[1] > counts[0]: return 1
    return -1 # Absolute tie

def main():
    bot_files = sorted(glob.glob("competitors/*.py"))
    bot_files = [f for f in bot_files if not f.endswith("__init__.py")]
    
    if len(bot_files) < 2:
        print("Need at least 2 bots in the 'competitors' folder!")
        return

    print("Loading competitors...")
    bots = {os.path.basename(f)[:-3]: load_bot(f) for f in bot_files}
    bot_names = list(bots.keys())
    
    print("\n--- Starting Tournament ---")
    results = {name: {"wins": 0, "losses": 0, "ties": 0} for name in bot_names}

    # Round Robin: Everyone plays everyone else TWICE (once as P0, once as P1)
    for i in range(len(bot_names)):
        for j in range(len(bot_names)):
            if i == j: continue
            
            b0, b1 = bot_names[i], bot_names[j]
            print(f"Match: {b0} (P0) vs {b1} (P1) ... ", end="", flush=True)
            
            winner_idx = play_match(b0, bots[b0], b1, bots[b1])
            
            if winner_idx == 0:
                print(f"{b0} wins")
                results[b0]["wins"] += 1
                results[b1]["losses"] += 1
            elif winner_idx == 1:
                print(f"{b1} wins")
                results[b1]["wins"] += 1
                results[b0]["losses"] += 1
            else:
                print("Tie")
                results[b0]["ties"] += 1
                results[b1]["ties"] += 1

    print("\n--- Final Standings ---")
    # Sort by wins
    standings = sorted(results.items(), key=lambda x: x[1]["wins"], reverse=True)
    for name, stats in standings:
        print(f"{name.ljust(15)} | Wins: {stats['wins']} | Losses: {stats['losses']} | Ties: {stats['ties']}")

if __name__ == "__main__":
    main()
