import os

timeout = 1

default_map = os.path.join("maps", "default", "simple.json")
possible_players = ["d"] + list(map(str, range(1, 11)))
end_player_states = ["S", "F"]

vis_width = 1000
vis_height = 600

max_tries = 10
target_radius = 5.4
min_skill = 10
max_skill = 100
extra_roll = 0.1
min_putter_dist = 20
max_dist = 200
