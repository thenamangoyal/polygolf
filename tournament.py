import os
import json
import argparse
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
from golf_game import GolfGame, return_vals
from multiprocessing import Pool


def generate_args(map, skill, log_path, seed):
    args = argparse.Namespace(address='127.0.0.1', automatic=False, disable_logging=True, disable_timeout=False, log_path=log_path, map=map, no_browser=False, no_gui=True, port=8080, seed=seed, skill=skill)
    return args


def worker(config):
    global RESULT_DIR, extra_df_cols
    log_path = None
    args = generate_args(map=config["map"], skill= config["skill"], log_path=log_path, seed=config["seed"])
    golf_game = GolfGame(player_list=config["player_list"], args=args)
    golf_game.play_all()
    result = golf_game.get_state()
    for df_col in extra_df_cols:
        result[df_col] = config[df_col]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", default="results", help="Directory path to dump results")
    parser.add_argument("--seed_entropy", "-s", type=int, help="Seed used to generate seed for each game")
    parser.add_argument("--trials", "-t", default=5, type=int, help="Number of trials for each config")
    parser.add_argument("--maps", "-m", default="tournament_maps.json", help="Json for tournament maps")
    args = parser.parse_args()
    RESULT_DIR = args.result_dir
    os.makedirs(RESULT_DIR, exist_ok=True)

    PLAYERS_LIST = list(map(str, range(1, 10)))
    SKILLS = [10, 40, 70, 100]
    TRIALS = args.trials
    with open(args.maps, "r") as f:
        map_dict = json.load(f)
    MAPS = list(itertools.chain.from_iterable(map_dict.values()))
    extra_df_cols = ["trial", "seed"]
    all_df_cols = extra_df_cols+return_vals

    seed_sequence = np.random.SeedSequence(args.seed_entropy)
    print("Using seed sequence with entropy {}".format(seed_sequence.entropy))
    with open(os.path.join(RESULT_DIR, "config.txt"), "w") as f:
        f.write("Seed entropy {}\n".format(seed_sequence.entropy))
        f.write("Players {}\n".format(PLAYERS_LIST))
        f.write("Skills {}\n".format(SKILLS))
        f.write("Trials {}\n".format(TRIALS))
        f.write("Maps\n{}\n".format(MAPS))

    base_tournament_configs = []
    for map in MAPS:
        for skill in SKILLS:
            for player in PLAYERS_LIST:
                player_list = [player]
                for trial in range(1, TRIALS+1):
                    base_config = dict()
                    base_config["map"] = map
                    base_config["skill"] = skill
                    base_config["player_list"] = player_list
                    base_config["trial"] = trial
                    base_tournament_configs.append(base_config)

    print("Total configs {}".format(len(base_tournament_configs)))
    seeds = seed_sequence.generate_state(len(base_tournament_configs), dtype=np.uint64)

    tournament_configs = []
    for i, base_config in enumerate(base_tournament_configs):
        config = base_config.copy()
        config["seed"] = seeds[i]
        tournament_configs.append(config)
    
    out_fn = os.path.join(RESULT_DIR, "aggregate_results.csv")
    with open(out_fn, "w") as csvf:
        header_df = pd.DataFrame([], columns=all_df_cols)
        header_df.to_csv(csvf, index=False, header=True)
        csvf.flush()
        with Pool() as p:
            for result in tqdm(p.imap(worker, tournament_configs), total=len(tournament_configs)):
                df = pd.DataFrame([result], columns=all_df_cols)
                df.to_csv(csvf, index=False, header=False)
                csvf.flush()