import numpy as np
import sympy
import logging
from typing import Tuple
import constants
from time import time
from shapely.geometry import Polygon, Point
from math import pi, atan2, inf, sqrt

from sklearn.neighbors import BallTree

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

        self.np_curr_loc = np.empty(2)
        self.np_target = np.empty(2)

        self.in_polygon = None
        self.origin = None

        self.n_distances = 20
        self.n_angles = 40

        self.angle_offset = pi*0.75


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
            shapely_polygon = Polygon([(p.x,p.y) for p in golf_map.vertices])

            x,y = list(zip(*golf_map.vertices))
            self.origin = (int(min(x)),int(min(y)))
            self.in_polygon = [[False]*int(max(y)-self.origin[1]+1) for _ in range(int(max(x)-self.origin[0]+1))]

            for i in range(len(self.in_polygon)):
                for j in range(len(self.in_polygon[0])):
                    px = self.origin[0]+i
                    py = self.origin[1]+j
                    self.in_polygon[i][j] = shapely_polygon.contains(Point(px, py))     
                    
        np.copyto(self.np_curr_loc, curr_loc.coordinates, casting='unsafe')
        np.copyto(self.np_target, target.coordinates, casting='unsafe')

        required_dist = np.linalg.norm(self.np_target - self.np_curr_loc)
        roll_factor = 1.1
        if required_dist < 20:
            roll_factor  = 1.0
        max_distance = min(200+self.skill, required_dist/roll_factor)
        target_angle = atan2(target.y - curr_loc.y, target.x - curr_loc.x)

        best_shot = (0,0)
        max_metric = 0

        for distance in np.linspace(max_distance, max_distance / self.n_distances, num=self.n_distances):
            for angle in np.linspace(target_angle-self.angle_offset, target_angle+self.angle_offset, num=self.n_angles):
                
                conf = self.est_shot_conf(distance, angle)
                
                p = self.np_curr_loc + distance * np.array([np.cos(angle), np.sin(angle)])
                target_dist = np.linalg.norm(self.np_target - p)
                
                metric = self.compute_metric(conf, target_dist)

                if metric > max_metric:
                    max_metric = metric
                    best_shot = (distance, angle)

        distance = (required_dist/roll_factor)
        if distance < (200+self.skill)*0.75:

            if distance < 20:
                distance *= 1.10
            else:
                distance *= 1.05

            conf = self.est_shot_conf(distance, target_angle)
            metric = self.compute_metric(conf, 0)

            if metric > max_metric:
                max_metric = metric
                best_shot = (distance, target_angle)

        print('shot', best_shot)
        return best_shot

    def compute_metric(self, conf, target_dist):
        return conf**3 / (target_dist+0.00001)

    def est_shot_conf(self, distance: float, angle: float, n_tries: int = 60):
        start_time = time()
        n_valid = 0

        # create memory now to save time creating lots of new np arrays
        landing_point = np.empty(2)
        final_point = np.empty(2)
        rot = np.empty(2)
        temp = np.empty(2)

        if constants.min_putter_dist <= distance <= constants.max_dist + self.skill: # normal shot
            case = 0
        elif distance < constants.min_putter_dist: # putter
            case = 1
        else: # invalid, doesn't move
            case = 2

        for _ in range(n_tries):
            
            actual_distance = self.rng.normal(distance, distance/self.skill)
            actual_angle = self.rng.normal(angle, 1/(2*self.skill))

            if case == 0: # normal shot
                rot[:2] = np.cos(actual_angle), np.sin(actual_angle)

                np.multiply(rot, actual_distance, out=temp)
                np.add(self.np_curr_loc, temp, out=landing_point)

                np.multiply(rot, (1.0 + constants.extra_roll)*actual_distance, out=temp)
                np.add(self.np_curr_loc, temp, out=final_point)

            elif case == 1: # putter
                np.copyto(landing_point, self.np_curr_loc)

                rot[:2] = np.cos(actual_angle), np.sin(actual_angle)
                np.multiply(rot, actual_distance, out=temp)
                np.add(self.np_curr_loc, temp, out=final_point)

            else: # invalid, doesn't move
                np.copyto(landing_point, self.np_curr_loc)
                np.copyto(final_point, self.np_curr_loc)

            if self.line_segment_in_polygon(landing_point, final_point, n_points_on_seg=5):
                n_valid += 1

        # t = time() - start_time
        # print('est_shot_conf time:', t)
        return n_valid / n_tries

    # Approx line-seg polygon intersection by checking points along the segment are inside polygon
    def line_segment_in_polygon(self, p1, p2, n_points_on_seg):
        direction = np.empty(2)
        temp = np.empty(2)
        p = np.empty(2)

        np.subtract(p2, p1, out=direction)
        for i in np.linspace(0, 1, num=n_points_on_seg):
            np.multiply(i, direction, out=temp)
            np.add(p1, temp, out=p)
            if not self.point_in_polygon(p):
                return False
        return True

    def point_in_polygon(self, p):
        x,y = int(p[0])-self.origin[0],int(p[1])-self.origin[1]
        return 0<=x<len(self.in_polygon) and 0<=y<len(self.in_polygon[0]) and self.in_polygon[x][y]


    class Node:
        def __init__(self, tup, target):
            self.x = tup[0]
            self.y = tup[1]
            self.came_from = None
            self.gscore = inf
            self.fscore = inf
            self.hscore = sqrt((self.x - target.x)**2 + (self.y - target.y)**2)
            self.neighbors = []

        def __eq__(self, other):
            if self.fscore == other.fscore:
                return True

        def __gt__(self, other):
            if self.fscore > other.fscore:
                return True
            else:
                return False

        def __lt__(self, other):
            if self.fscore < other.fscore:
                return True
            else:
                return False

