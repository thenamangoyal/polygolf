import math
import random

import numpy as np
import sympy
import logging
from typing import Tuple
from shapely.geometry import Polygon, Point, LineString
from time import time
from shapely.ops import triangulate
import math


def is_roll_in_polygon(point_a, distance, angle, polygon):
    curr_point = Point(point_a.x + distance * np.cos(angle),
                       point_a.y + distance * np.sin(angle))
    final_point = Point(
        point_a.x + (1.1) * distance * np.cos(angle),
        point_a.y + (1.1) * distance * np.sin(angle))

    segment_land = LineString([curr_point, final_point])
    if_encloses = polygon.contains(segment_land)
    return if_encloses


def convert_sympy_shapely(point):
    return Point(point.x, point.y)


class LandingPoint(object):
    def __init__(self, point, distance_from_origin, angle_from_origin, start_point, hole_point):
        # distance_from_origin is the distance from our curr_location to the landing point

        self.point = point
        self.distance_from_origin = distance_from_origin
        self.angle_from_origin = angle_from_origin
        self.hole = hole_point
        self.score_threshold = 95
        self.trials = 20
        self.start_point = start_point


    @staticmethod
    def is_on_land(point, polygon):
        return point.within(polygon)

    def confidence(self, polygon, skill, rng):
        try:
            return self.shot_confidence
        except AttributeError:
            intended_distance = self.distance_from_origin
            intended_angle = self.angle_from_origin
            successful = 0
            for t in range(0, self.trials):
                actual_distance = rng.normal(intended_distance, intended_distance / skill)
                actual_angle = rng.normal(intended_angle, 1 / (2 * skill))
                if is_roll_in_polygon(self.start_point, actual_distance, actual_angle, polygon):
                    successful += 1
            self.shot_confidence = successful / self.trials
            return self.shot_confidence

    def heuristic(self):
        try:
            return self.distance_to_hole
        except AttributeError:
            self.distance_to_hole = -self.point.distance(self.hole)
            return self.distance_to_hole

    def score(self, polygon, skill, rng):
        # uses confidence and heuristic
        return self.heuristic() + (self.confidence(polygon, skill, rng) * 100)

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

    def direct_distance_angle(self, curr_loc, target):
        required_dist = curr_loc.distance(target)
        roll_factor = 1.1
        if required_dist < 20:
            roll_factor = 1.0
        max_dist_traveled = 200 + self.skill
        distance = min(max_dist_traveled, required_dist / roll_factor)
        angle = math.atan2(target.y - curr_loc.y, target.x - curr_loc.x)
        print("d, a:", distance, angle)

        return distance, angle

    def search_points(self, curr_loc, target, polygon, increment=25):
        distance, angle = self.direct_distance_angle(curr_loc, target)
        points = []
        r = distance
        while r > 0:
            semicircle_length = np.pi * r
            num_sector = int(semicircle_length / increment)  # divide the semicircle into equally sized sectors
            num_sector = num_sector if num_sector % 2 == 0 else num_sector + 1
            if num_sector == 0:
                r -= increment
                continue
            arc_length = semicircle_length / num_sector
            angle_increment = np.pi / (2 * num_sector)
            for i in range(0, int(num_sector) + 1):
                new_angle = float(angle + (i * angle_increment))
                point = Point(curr_loc.x + r * np.cos(new_angle),
                              curr_loc.y + r * np.sin(new_angle))

                if LandingPoint.is_on_land(point, polygon):
                    lp = LandingPoint(point, r, new_angle, curr_loc, target)
                    points.append(lp)
                if i > 0:
                    new_angle = float(angle - (i * angle_increment))
                    point = Point(curr_loc.x + r * np.cos(new_angle),
                                  curr_loc.y + r * np.sin(new_angle))

                    if LandingPoint.is_on_land(point, polygon):
                        lp = LandingPoint(point, r, new_angle, curr_loc, target)
                        points.append(lp)
            r -= increment

        return points

    def search_landing_points(self, points, polygon, skill, rng):
        largest_point = None
        largest_point_score = -1 * float('inf')
        for point in points:
            start = time()
            score = point.score(polygon, skill, rng)
            end = time()
            # print("time: ", end-start)
            if score > largest_point_score:
                largest_point = point
                largest_point_score = score

        return largest_point.distance_from_origin, largest_point.angle_from_origin

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

        if score == 1:
            self.shapely_polygon = Polygon([(p.x, p.y) for p in golf_map.vertices])

        curr_loc = convert_sympy_shapely(curr_loc)
        target = convert_sympy_shapely(target)
        print((curr_loc.x, curr_loc.y), (target.x, target.y))
        landing_points = self.search_points(curr_loc, target, self.shapely_polygon)
        print([(l.point.x, l.point.y) for l in landing_points])
        return self.search_landing_points(landing_points, self.shapely_polygon, self.skill, self.rng)