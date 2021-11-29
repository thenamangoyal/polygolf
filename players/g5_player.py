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

    def calculate_angle_and_distance(self, point_a, point_b):
        distance = point_a.distance(point_b)
        angle = sympy.atan2(point_b.y - point_a.y, point_b.x - point_a.x)
        return angle, distance

    def is_roll_in_polygon(self, curr_point, distance, angle, map):
        curr_point = sympy.Point2D(curr_point.x + distance * sympy.cos(angle),
                                   curr_point.y + distance * sympy.sin(angle))
        final_point = sympy.Point2D(
            curr_point.x + (1.1) * distance * sympy.cos(angle),
            curr_point.y + (1.1) * distance * sympy.sin(angle))

        segment_land = sympy.geometry.Segment2D(curr_point, final_point)
        return map.encloses(segment_land)

    def search_landing_points(self, landing_points, curr_loc, map):
        for landing_point in landing_points:
            angle, distance = self.calculate_angle_and_distance(curr_loc, landing_point)
            if self.is_roll_in_polygon(curr_loc, distance, angle, map):
                return angle, distance

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
