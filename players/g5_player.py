import numpy as np
import sympy
import logging
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

    def get_targets(self, target: sympy.geometry.Point2D, curr_loc: sympy.geometry.Point2D):
        required_dist = curr_loc.distance(target)
        roll_factor = 1.1
        if required_dist < 20:
            roll_factor  = 1.0 
        max_dist_traveled = 200 + self.skill
        distance = sympy.Min(max_dist_traveled, required_dist/roll_factor)
        angle = sympy.atan2(target.y - curr_loc.y, target.x - curr_loc.x)
        farthest_point = (curr_loc.x+distance*sympy.cos(angle), curr_loc.y + distance*sympy.sin(angle))
        cone_angle = sympy.atan2(curr_loc.y - farthest_point.y, curr_loc.x - farthest_point.x)
        
        point_list = []
        for r in range(distance):
            for theta in range(sympy.pi/4):
                if theta == 0:
                    point_list.append((farthest_point.x + r*sympy.cos(cone_angle), farthest_point.y + r*sympy.sin(cone_angle)))
                else:
                    point_list.append((farthest_point.x + r*sympy.cos(cone_angle+theta), farthest_point.y + r*sympy.sin(cone_angle+theta)))
                    point_list.append((farthest_point.x + r*sympy.cos(cone_angle-theta), farthest_point.y + r*sympy.sin(cone_angle-theta)))
        return point_list

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
        roll_factor = 1.1
        if required_dist < 20:
            roll_factor  = 1.0
        distance = sympy.Min(200+self.skill, required_dist/roll_factor)
        angle = sympy.atan2(target.y - curr_loc.y, target.x - curr_loc.x)
        return (distance, angle)
