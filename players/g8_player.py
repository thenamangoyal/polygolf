import numpy as np
import sympy
import logging
from typing import Tuple
import constants
from time import time
from shapely.geometry import Polygon, Point
from math import pi, atan2

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

        self.shapely_polygon = None
        self.np_curr_loc = np.empty(2)
        self.np_target = np.empty(2)

        self.n_distances = 20
        self.n_angles = 50

        self.angle_offset = pi/4 # 45 deg

        self.min_conf = 0.50

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
            Tuple[float, float]: Return a tuple of (landing, doesn't include roll) distance and angle in radians to play the shot
        """

        # init golf map polygon
        if score == 1:
            self.shapely_polygon = Polygon([(p.x,p.y) for p in golf_map.vertices])

        np.copyto(self.np_curr_loc, curr_loc.coordinates, casting='unsafe')
        np.copyto(self.np_target, target.coordinates, casting='unsafe')

        required_dist = curr_loc.distance(target)
        roll_factor = 1.1
        if required_dist < 20:
            roll_factor  = 1.0
        max_distance = min(200+self.skill, required_dist/roll_factor)
        target_angle = atan2(target.y - curr_loc.y, target.x - curr_loc.x)

        min_target_dist = float('inf')
        out = (0,0)
        distance_step = float(max_distance) / self.n_distances
        angle_step = (2 * self.angle_offset) / self.n_angles
        for distance in np.arange(distance_step, float(max_distance) + distance_step, distance_step):
            for angle in np.arange(target_angle-self.angle_offset, target_angle+self.angle_offset+angle_step, angle_step):
                conf = self.est_shot_conf(distance, angle)
                p = self.np_curr_loc + distance * np.array([np.cos(angle), np.sin(angle)])
                target_dist = np.linalg.norm(self.np_target - p)
                # print(distance, angle, conf)
                if conf >= self.min_conf:
                    if target_dist < min_target_dist:
                        min_target_dist = target_dist
                        out = (distance, angle)

        return out

    def est_shot_conf(self, distance: float, angle: float, n_tries: int = 100, n_points_on_seg: int = 10):
        start_time = time()
        n_valid = 0

        # create memory now to save time creating lots of new np arrays
        landing_point = np.empty(2)
        final_point = np.empty(2)
        rot = np.empty(2)
        temp = np.empty(2)
        direction = np.empty(2)
        p = np.empty(2)

        if constants.min_putter_dist <= distance <= constants.max_dist + self.skill: # normal shot
            case = 0
        elif distance < constants.min_putter_dist: # putter
            case = 1
        else: # invalid, doesn't move
            case = 2

        for _ in range(n_tries):
            
            actual_distance = self.rng.normal(distance, distance/self.skill)
            actual_angle = self.rng.normal(angle, 1/(2*self.skill))

            if case == 0:
                rot[:2] = np.cos(actual_angle), np.sin(actual_angle)

                np.multiply(rot, actual_distance, out=temp)
                np.add(self.np_curr_loc, temp, out=landing_point)

                np.multiply(rot, (1.0 + constants.extra_roll)*actual_distance, out=temp)
                np.add(self.np_curr_loc, temp, out=final_point)

            elif case == 1:
                np.copyto(landing_point, self.np_curr_loc)

                rot[:2] = np.cos(actual_angle), np.sin(actual_angle)
                np.multiply(rot, actual_distance, out=temp)
                np.add(self.np_curr_loc, temp, out=final_point)

            else:
                np.copyto(landing_point, self.np_curr_loc)
                np.copyto(final_point, self.np_curr_loc)

            # check if n_points_on_seg are inside polygon
            seg_inside_poly = True
            np.subtract(final_point, landing_point, out=direction)
            i_step = 1 / n_points_on_seg
            for i in np.arange(0, 1 + i_step, i_step):
                np.multiply(i, direction, out=temp)
                np.add(landing_point, temp, out=p)
                if not self.shapely_polygon.contains(Point(p[0], p[1])):
                    seg_inside_poly = False
                    break

            if seg_inside_poly:
                n_valid += 1

        t = time() - start_time
        # print('est_shot_conf time:', t)
        return n_valid / n_tries