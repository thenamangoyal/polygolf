import pandas as pd

df = pd.read_csv(
    "run1_aggregate_results.csv",
    converters={
        "trial": int,
        "seed": int,
        "player_names": eval,
        "map": str,
        "skills": eval,
        "player_states": eval,
        "distances_from_target": eval,
        "distance_source_to_target": float,
        "start": eval,
        "target": eval,
        "penalties": eval,
        "timeout_count": eval,
        "error_count": eval,
        "winner_list": eval,
        "total_time_sorted": eval
    }
)

# get list of dictionaries suing
results_ld = df.to_dict("records")
# get list of lists using
results_ll = df.values.tolist()

# iterate over results directly using, slower than above too
for idx, row in df.iterrows():
    print(row)