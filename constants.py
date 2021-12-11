import os

timeout = 10*60 # 10 min aggregate time limit

default_map = os.path.join("maps", "default", "simple.json")
possible_players = ["d"] + list(map(str, range(1, 10)))
end_player_states = ["S", "F"]

vis_width = 1000
vis_height = 550
vis_width_ratio = 0.8
vis_height_ratio = 0.75
vis_padding = 0.02

max_tries = 10
target_radius = 0.054
min_skill = 10
max_skill = 100
extra_roll = 0.1
min_putter_dist = 20
max_dist = 200
