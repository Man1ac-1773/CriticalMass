# Critical Mass 

A python bot that plays the chain reaction game. 

Designed for the competition "Critical Mass" hosted by NJACK ML, IIT Patna. 
It features a simple minimax with alpha beta pruning, highly optimized by using transposition tables (with zobrist hashing), iterative deepening, killer move heuristics, and late move reduction. 

## Repo structure
### Resources `resources/`
Contains the official game resources provided. 

Read `resources/README.md` for more details.

### Solution `my_bot.py`
My submitted solution for the competition. 

For details on the bot read `report.md`

### Running tournaments
Functionality for running a tournament (written by self) is given in `run_tournament.sh` and `tournament.py`


## Instructions for running tournament
There are two ways provided in which a tournament can be run. 

### Manual
Put all the competition bots in a `competitors/` folder inside the root directory and run the script `tournament.py`. 

All print statements by bots will be caught and thrown inside a dedicated log. Can be used for telemetry and debugging.

### Using git
A tournament can also be run across branches without having to manually extract the files. 
Just make sure to make a branch by `git branch` and mentioning branch names or hashes inside `run_tournament.sh` inside the `BRANCHES = ` variable. 

It will automatically extract corresponding `my_bot.py` across those branches and run a tournament amongst them. 

Read `run_tournament.sh` for more details. 
