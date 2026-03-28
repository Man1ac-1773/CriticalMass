import os
import glob
import time
import importlib.util
import io
import contextlib
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
    names = {0: bot0_name, 1: bot1_name}

    # Open the log file in append mode once per match
    with open("tournament_stats.log", "a") as logfile:
        logfile.write(f"\n=== MATCH START: {bot0_name} (P0) vs {bot1_name} (P1) ===\n")

        for turn in range(max_turns):
            player = turn % 2
            
            # Create an invisible buffer to catch the bot's print statements
            stdout_trap = io.StringIO()
            
            try:
                start_time = time.time()
                
                # Redirect standard output into the trap while the bot thinks
                with contextlib.redirect_stdout(stdout_trap):
                    move = bots[player](game.get_state(), player)
                    
                elapsed = time.time() - start_time
                bot_output = stdout_trap.getvalue().strip()
                
                # Log it cleanly
                if bot_output:
                    logfile.write(f"Turn {turn+1:3} | {names[player]:<15} | {bot_output} | Time: {elapsed:.3f}s\n")
                else:
                    logfile.write(f"Turn {turn+1:3} | {names[player]:<15} | (No telemetry) | Time: {elapsed:.3f}s\n")

                # Strict tournament time enforcement
                if elapsed > 1.05: # Slight buffer for system noise
                    error_msg = f"[{names[player]}] timed out ({elapsed:.3f}s)!"
                    print(error_msg)
                    logfile.write(f"FATAL: {error_msg}\n")
                    return 1 - player
                    
                game.apply_move(player, move)
                
            except Exception as e:
                error_msg = f"[{names[player]}] crashed: {e}"
                print(error_msg)
                logfile.write(f"FATAL: {error_msg}\n")
                return 1 - player # Opponent wins by default

            winner = game.check_winner()
            if winner is not None:
                logfile.write(f"=== MATCH END: Winner {names[winner]} ===\n")
                return winner

    # Tiebreaker logic if max_turns reached without a wipe
    state = game.get_state()
    counts = {0: 0, 1: 0}
    for row in state:
        for owner, orb_count in row:
            if owner in (0, 1) and orb_count > 0:
                counts[owner] += 1
                
    with open("tournament_stats.log", "a") as logfile:
        logfile.write(f"=== MATCH END: Tiebreaker Reached. P0: {counts[0]}, P1: {counts[1]} ===\n")

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
    
    # Wipe the log file clean for the new tournament run
    with open("tournament_stats.log", "w") as logfile:
        logfile.write("=== NEW TOURNAMENT RUN ===\n")
    
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
