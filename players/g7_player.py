import numpy as np
import sympy
import logging
import math
from typing import Tuple

class Player:
    def __init__(self, skill: int, rng: np.random.Generator, logger: logging.Logger) -> None:
        """Initialise the player with given skill.

        Args:
            skill (int): skill of your player
            rng (np.random.Generator): numpy random number generator, use this for same player behvior across run
            logger (logging.Logger): logger use this like logger.info("message")
        """
        self.skill = skill
        self.rng = rng
        self.logger = logger
        self.logger.info(f'SKILL LEVEL: {self.skill}')

    def get_location_from_shot(self, distance, angle, curr_loc):
        # angle is in rads
        y_delta = distance * math.sin(angle) 
        x_delta = distance * math.cos(angle)
        new_x = curr_loc.x + x_delta
        new_y = curr_loc.y + y_delta
        return sympy.Point(new_x, new_y)

    def check_shot(self, distance, angle, curr_loc, roll_factor, golf_map):
        final_location = self.get_location_from_shot(distance * roll_factor, angle, curr_loc)
        drop_location = self.get_location_from_shot(distance, angle, curr_loc)
        return golf_map.encloses_point(final_location) and golf_map.encloses_point(drop_location)

    def find_shot(self, distance, angle, curr_loc, target, roll_factor, golf_map):
        # if roll_factor is less than 1.1, it's a putt
        distance_to_target = curr_loc.distance(target)
        min_distance_threshold = (200 + self.skill) * 0.5
        if (roll_factor > 1.0 and
            distance < min_distance_threshold and 
            distance_to_target > min_distance_threshold
            ) or distance < 0:
            return (None, None)
        if self.check_shot(distance, angle, curr_loc, roll_factor, golf_map):
            return (distance, angle)
        else:
            self.logger.info(f'shot {distance} {angle} not viable')
            return self.find_shot(distance - 10, angle, curr_loc, target, roll_factor, golf_map)

    def play(self, score: int, golf_map: sympy.Polygon, target: sympy.geometry.Point2D, curr_loc: sympy.geometry.Point2D, prev_loc: sympy.geometry.Point2D, prev_landing_point: sympy.geometry.Point2D, prev_admissible: bool) -> Tuple[float, float]:
        """Function which based n current game state returns the distance and angle, the shot must be played 

        Args:
            score (int): Your total score including current turn
            golf_map (sympy.Polygon): Golf Map polygon
            target (sympy.geometry.Point2D): Target location
            curr_loc (sympy.geometry.Point2D): Your current location
            prev_loc (sympy.geometry.Point2D): Your previous location. If you haven't played previously then None
            prev_landing_point (sympy.geometry.Point2D): Your previous shot landing location. If you haven't played previously then None
            prev_admissible (bool): Boolean stating if your previous shot was within the polygon limits. If you haven't played previously then None

        Returns:
            Tuple[float, float]: Return a tuple of distance and angle in radians to play the shot
        """
        required_dist = curr_loc.distance(target)
        roll_factor = 1.0 if required_dist < 20 else 1.1 
        distance = sympy.Min(200+self.skill, required_dist/roll_factor)
        angle = sympy.atan2(target.y - curr_loc.y, target.x - curr_loc.x)
        new_distance, new_angle = self.find_shot(distance, angle, curr_loc, target, roll_factor, golf_map)

        angle_adjusted = sympy.pi/18
        while new_distance == None:
            new_distance, new_angle = self.find_shot(distance, angle+angle_adjusted, curr_loc, target, roll_factor, golf_map)
            if new_distance == None:
                new_distance, new_angle = self.find_shot(distance, angle-angle_adjusted, curr_loc, target, roll_factor, golf_map)
            angle_adjusted += sympy.pi/18
            if angle_adjusted == sympy.pi:
                print("Edge Case Hitted: Cannot find angle to shoot more than 0.5 of max distance.")
                break

        return (new_distance, new_angle)
