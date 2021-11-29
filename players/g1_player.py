import numpy as np
import sympy
from sympy.geometry import Point2D as Point
import logging
from typing import Tuple
import heapq
import constants
import matplotlib.pyplot as plt

class Player:
    def __init__(self, skill: int, rng: np.random.Generator, logger: logging.Logger) -> None:
        """Initialise the player with given skill.

        Args:
            skill (int): skill of your player
            rng (np.random.Generator): numpy random number generator, use this for same player behvior across run
            logger (logging.Logger): logger use this like logger.info("message")
            paths (list): a list of points
        """
        self.skill = skill
        self.rng = rng
        self.logger = logger

    def play(self, score: int, golf_map: sympy.Polygon, target: sympy.geometry.Point2D,
             curr_loc: sympy.geometry.Point2D, prev_loc: sympy.geometry.Point2D,
             prev_landing_point: sympy.geometry.Point2D, prev_admissible: bool) -> Tuple[float, float]:
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

        # Greedy first
        distance, angle = self.greedy(target, curr_loc)
        # return distance, angle
        # print(distance, (angle * 180/sympy.pi).evalf(), self.skill)
        if self.is_landing_pt_safe(5, golf_map, curr_loc, distance, angle):
            return distance, angle
        # Sweep
        delta_angle = 5 * sympy.pi / 180
        distance, angle = self.sweep(golf_map, target, curr_loc, delta_angle)
        # print(distance, (angle * 180/sympy.pi).evalf())
        return distance, angle

    def sweep(self, golf_map: sympy.Polygon, target: sympy.geometry.Point2D, curr_loc: sympy.geometry.Point2D,
              delta_angle: float) -> Tuple[float, float]:
        # counter-clock-wise or clock-wise
        trails = 5
        direction = 1 if self.rng.random() < 0.5 else -1
        num_times = 2 * sympy.pi // delta_angle
        distance, start_angle = self.greedy(target, curr_loc)  # Already take 10% extra distance traveled into acct
        segment_len = distance // trails  # this can also be dynamic
        hq = []
        # print("Start_angle: ", (start_angle * 180/sympy.pi).evalf())
        for i in range(num_times):
            cur_dist = distance
            new_angle = start_angle + i * delta_angle * direction
            #print("New angle: ", (new_angle * 180/sympy.pi).evalf())
            # For each angle, try various distances
            cnt = trails
            while cnt > 0:
                # print("current dist:", cur_dist)

                if self.is_landing_pt_safe(3, golf_map, curr_loc, cur_dist, new_angle):
                    # print("Point safe")
                    f = self.safety_measure(cur_dist, new_angle, curr_loc, target, golf_map)
                    heapq.heappush(hq, (f, (cur_dist, new_angle)))
                    # Keep the top 10 best (dist, ang)
                cnt -= 1
                cur_dist -= segment_len
            # print("len of hq:", len(hq))
        # for item in hq:
        #     print(item[1][0], item[1][-1].evalf())

        return heapq.heappop(hq)[1] if hq else (0.1, 0)

    def safety_measure(self, distance: float, angle: float, curr_loc: sympy.geometry.Point2D,
                       target: sympy.geometry.Point2D, golf_map: sympy.Polygon) -> float:

        # STD for distance; rn this is a fixed value but should be dynamic
        # dist_std = 2.2 * distance / self.skill
        land_point = Point(curr_loc.x + distance * sympy.cos(angle), curr_loc.y + distance * sympy.sin(angle))
        # land_circle = sympy.Circle(land_point, dist_std)
        # intersections = golf_map.intersection(land_circle)
        # print("Landpoint distance:", land_point.distance(target).evalf())
        return land_point.distance(target).evalf()/10


    def greedy(self, target: sympy.geometry.Point2D, curr_loc: sympy.geometry.Point2D) -> Tuple[float, float]:

        """
            Default greedy algorithm
        :param target:  Target location
        :param curr_loc:  Current location
        :return:
        """

        required_dist = curr_loc.distance(target)
        roll_factor = 1. + constants.extra_roll
        if required_dist < 20:
            roll_factor = 1.0
        distance = sympy.Min(200 + self.skill, required_dist / roll_factor)
        angle = sympy.atan2(target.y - curr_loc.y, target.x - curr_loc.x)
        return distance, angle

    def is_landing_pt_safe(self, iteration: int, golf_map: sympy.Polygon, curr_loc: sympy.geometry.Point2D,
                           distance: float,
                           angle: float) -> bool:
        """
            Check if a point is safe.

        :param iteration:  number of iterations we want to try this distance and angle
        :param golf_map:   Golf Map polygon
        :param curr_loc:   Current location
        :param distance:   Aimed distance
        :param angle:      Aimed angle
        :return:   True if it landed successfully iteration of times.
        """
        valid_cnt = 0
        for _ in range(iteration):
            # From golf_game.py
            actual_distance = self.rng.normal(distance, distance / self.skill)
            actual_angle = self.rng.normal(angle, 1 / (2 * self.skill))

            if constants.max_dist + self.skill >= distance >= constants.min_putter_dist:
                landing_point = Point(curr_loc.x + actual_distance * sympy.cos(actual_angle),
                                      curr_loc.y + actual_distance * sympy.sin(actual_angle))
                final_point = Point(
                    curr_loc.x + (1. + constants.extra_roll) * actual_distance * sympy.cos(actual_angle),
                    curr_loc.y + (1. + constants.extra_roll) * actual_distance * sympy.sin(actual_angle))
            else:
                landing_point = curr_loc
                final_point = sympy.Point(curr_loc.x + actual_distance * sympy.cos(actual_angle),
                                          curr_loc.y + actual_distance * sympy.sin(actual_angle))

            segment_land = sympy.geometry.Segment2D(landing_point, final_point)
            if golf_map.encloses(segment_land):
                valid_cnt += 1

        return valid_cnt == iteration
