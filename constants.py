import os
max_tries = 10
target_radius = 5.4
default_map = os.path.join("maps", "default", "simple.json")
end_player_states = ["S", "F"]
min_skill = 10
max_skill = 100
extra_roll = 0.1
min_putter_dist = 20
max_dist = 200
possible_players = ["d"] + list(map(str, range(1, 11)))
