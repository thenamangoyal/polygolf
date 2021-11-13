# polygolf

Project 4 Polygolf - COMS 4444 Fall 2021 Programming and Problem Solving

<http://www.cs.columbia.edu/~kar/4444f21/node21.html>

## Installation

Requires **python3.6** or higher

Install simulator packages only
```bash
pip install -r requirements.txt
```

Install map generation packages. Note the `--user` option to avoid conflicts with system packages
```bash
pip install -r requirements.txt --user
```

## Usage

```bash
python golf_game.py
```

You can also specify the optional parameters below to disable GUI, disable browser launching, change port and address of server.

```bash
usage: golf_game.py [-h] [--map MAP] [--automatic] [--seed SEED] [--port PORT]
                    [--address ADDRESS] [--no_browser] [--no_gui]
                    [--log_path LOG_PATH] [--disable_timeout]
                    [--disable_logging] [--players PLAYERS [PLAYERS ...]]

optional arguments:
  -h, --help            show this help message and exit
  --map MAP, -m MAP     Path to map json file
  --automatic           Start playing automatically in GUI mode
  --seed SEED, -s SEED  Seed used by random number generator, specify 0 to use
                        no seed and have different random behavior on each
                        launch
  --port PORT           Port to start
  --address ADDRESS, -a ADDRESS
                        Address
  --no_browser, -nb     Disable browser launching in GUI mode
  --no_gui, -ng         Disable GUI
  --log_path LOG_PATH   Directory path to dump log files, filepath if
                        disable_logging is false
  --disable_timeout, -time
                        Disable Timeout in non GUI mode
  --disable_logging     Disable Logging, log_path becomes path to file
  --players PLAYERS [PLAYERS ...], -p PLAYERS [PLAYERS ...]
                        List of players space separated
```


## Debugging

The code generates a `log/debug.log` (detailed), `log/results.log` (minimal) and `log\<player_name>.log` (logs from player) on every execution, detailing all the turns and steps in the game.
