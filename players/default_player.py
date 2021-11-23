import numpy as np
import sympy
import logging


class Player:
    def __init__(self, skill, rng, logger) -> None:
        self.skill = skill
        self.rng = rng
        self.logger = logger

    def play(self, score, golf_map, target, curr_loc, prev_loc, prev_landing_point, prev_admissible):
        distance = min(200+self.skill, float(curr_loc.distance(target)/(1.1)))
        slope = (target.y - curr_loc.y)/(target.x - curr_loc.x)
        slope = float(slope)
        angle = np.arctan(np.abs(slope))
        if slope > 0 and curr_loc.x <= target.x:
            angle = angle
        elif slope > 0 and curr_loc.x > target.x:
            angle = np.pi + angle
        elif slope < 0 and curr_loc.x <= target.x:
            angle = 2*np.pi - angle
        else:
            # means slope < 0 and curr_loc.x > target.x:
            angle = np.pi - angle
        return distance, angle
