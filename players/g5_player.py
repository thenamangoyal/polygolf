import math
import random

import numpy as np
import sympy
import logging
from typing import Tuple
from shapely.geometry import Polygon, Point, LineString
from time import time
from shapely.ops import triangulate
import matplotlib.pyplot as plt


class LandingPoint(object):
    def __init__(self, x, y, distance_from_origin, angle):
        # distance_from_origin is the distance from our curr_location to the landing point

        self.point = Point(x, y)
        self.distance_from_origin = distance_from_origin
        self.angle = angle

    @staticmethod
    def is_on_land(x, y, polygon):
        pass

    def confidence(self):
        pass

    def heuristic(self):
        pass

    def score(self):
        # uses confidence and heuristic
        pass


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
        distance = sympy.Min(max_dist_traveled, required_dist / roll_factor)
        angle = sympy.atan2(target.y - curr_loc.y, target.x - curr_loc.x)
        return distance, angle

    def search_points(self, curr_loc, target, increment=25):
        distance, angle = self.direct_distance_angle(curr_loc, target)
        points = [curr_loc]
        r = distance
        while r > 0:
            semicircle_length = np.pi * r
            num_sector = int(semicircle_length / increment)  # divide the semicircle into equally sized sectors
            num_sector = num_sector if num_sector % 2 == 0 else num_sector + 1
            arc_length = semicircle_length / num_sector
            angle_increment = np.pi / (2 * num_sector)
            for i in range(0, int(num_sector) + 1):
                point = Point(curr_loc.x + r * np.cos(float(angle + (i * angle_increment))),
                              curr_loc.y + r * np.sin(float(angle + (i * angle_increment))))
                points.append(point)
                if i > 0:
                    point = Point(curr_loc.x + r * np.cos(float(angle - (i * angle_increment))),
                                  curr_loc.y + r * np.sin(float(angle - (i * angle_increment))))
                    points.append(point)
            r -= increment
        # x = [c.x for c in corners]
        # y = [c.y for c in corners]
        # plt.scatter(x, y)
        # plt.gcf().set_size_inches((20, 40))
        # plt.show()
        '''
        semicircle_length = np.pi * distance
        num_sector = int(semicircle_length / increment)  # divide the semicircle into equally sized sectors
        num_sector = num_sector if num_sector % 2 == 0 else num_sector + 1
        arc_length = semicircle_length / num_sector
        angle_increment = np.pi / (2 * num_sector)
        corners = [curr_loc]
        for i in range(0, int(num_sector) + 1):
            point = Point(curr_loc.x + distance * np.cos(float(angle + (i * angle_increment))),
                          curr_loc.y + distance * np.sin(float(angle + (i * angle_increment))))
            corners.append(point)
            if i > 0:
                point = Point(curr_loc.x + distance * np.cos(float(angle - (i * angle_increment))),
                              curr_loc.y + distance * np.sin(float(angle - (i * angle_increment))))
                corners.append(point)
        total_points = 100
        semicircle = Polygon(corners)
        semicircle_area = semicircle.area
        triangles = triangulate(semicircle, edges=False)
        random_points = []
        for triangle in triangles:
            num_points = int(triangle.area * total_points / semicircle_area)
            for i in range(0, num_points):
                minx, miny, maxx, maxy = triangle.bounds
                x = np.random.uniform(low=minx, high=maxx)
                y = np.random.uniform(low=miny, high=maxy)
                # triangle.representative_point()
                random_points.append(Point(x, y))

        x = [c.x for c in random_points]
        y = [c.y for c in random_points]
        plt.scatter(x, y)
        plt.gcf().set_size_inches((20, 40))
        plt.show()
        '''
        return points

    def calculate_angle_and_distance(self, point_a, point_b):
        distance = point_a.distance(point_b)
        angle = sympy.atan2(point_b.y - point_a.y, point_b.x - point_a.x)
        return angle, distance

    def is_roll_in_polygon(self, point_a, distance, angle, map):
        x = float(point_a.x.evalf())
        y = float(point_a.y.evalf())
        # distance = float(distance.evalf())
        angle = float(angle.evalf())
        curr_point = Point(x + distance * np.cos(angle),
                           y + distance * np.sin(angle))
        final_point = Point(
            x + (1.1) * distance * np.cos(angle),
            y + (1.1) * distance * np.sin(angle))

        segment_land = LineString([curr_point, final_point])
        start_time = time()
        if_encloses = self.shapely_polygon.contains(segment_land)
        end_time = time()
        # print("time: ", end_time - start_time)
        return if_encloses

    def search_landing_points(self, landing_points, curr_loc, map):
        for landing_point in landing_points:
            angle, distance = self.calculate_angle_and_distance(curr_loc, landing_point)
            if self.is_roll_in_polygon(curr_loc, distance, angle, map):
                return distance, angle

        return 0, 0  # TODO: bad position, go back

    def search_targets(self, target, curr_loc, map):
        required_dist = curr_loc.distance(target)
        roll_factor = 1.1
        if required_dist < 20:
            roll_factor = 1.0
        max_dist_traveled = 200 + self.skill
        distance = sympy.Min(max_dist_traveled, required_dist / roll_factor)
        angle = sympy.atan2(target.y - curr_loc.y, target.x - curr_loc.x)

        r = float(distance)
        while r >= 0:
            theta = 0
            while theta < sympy.pi / 2:
                if theta == 0:
                    if self.is_roll_in_polygon(curr_loc, r, angle, map):
                        return r, angle
                else:
                    if self.is_roll_in_polygon(curr_loc, r, angle + theta, map):
                        return distance, angle + theta
                    if self.is_roll_in_polygon(curr_loc, r, angle - theta, map):
                        return r, angle - theta
                theta += float(sympy.pi / 120)
                # theta += float(random.uniform(-.1, 0.1))
                # print("theta loop")
            # print("distance loop")
            r -= 20
            # print("distance", r)
        return 0, 0

    def get_targets(self, target: sympy.geometry.Point2D, curr_loc: sympy.geometry.Point2D):
        required_dist = curr_loc.distance(target)
        roll_factor = 1.1
        if required_dist < 20:
            roll_factor = 1.0
        max_dist_traveled = 200 + self.skill
        distance = sympy.Min(max_dist_traveled, required_dist / roll_factor)
        angle = sympy.atan2(target.y - curr_loc.y, target.x - curr_loc.x)
        farthest_point = sympy.Point2D(curr_loc.x + distance * sympy.cos(angle),
                                       curr_loc.y + distance * sympy.sin(angle))
        cone_angle = sympy.atan2(curr_loc.y - farthest_point.y, curr_loc.x - farthest_point.x)

        point_list = []
        r = 0
        while r < distance:
            theta = 0
            while theta < sympy.pi / 4:
                if theta == 0:
                    point_list.append(sympy.Point2D(farthest_point.x + r * sympy.cos(cone_angle),
                                                    farthest_point.y + r * sympy.sin(cone_angle)))
                else:
                    point_list.append(sympy.Point2D(farthest_point.x + r * sympy.cos(cone_angle + theta),
                                                    farthest_point.y + r * sympy.sin(cone_angle + theta)))
                    point_list.append(sympy.Point2D(farthest_point.x + r * sympy.cos(cone_angle - theta),
                                                    farthest_point.y + r * sympy.sin(cone_angle - theta)))
                theta += 0.1  # sympy.pi / 24
            r += 1
        return point_list

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

        points = self.search_points(curr_loc, target)

        return self.search_targets(target, curr_loc, golf_map)
