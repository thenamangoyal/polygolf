import numpy as np
import sympy
import logging
import math
from typing import Tuple
from shapely.geometry import Polygon, Point
import shapely.affinity
from collections import defaultdict
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
        self.grid = None 
        self.golf_map = None
        self.valueMap = defaultdict(lambda: float('inf'))
        self.graph = {}

    def create_grid(self, polygon):
        self.grid = []
        self.golf_map, bounding_box = PolygonUtility.convert_sympy_to_shapely(polygon)
        minX, minY, maxX, maxY = bounding_box
        granularity = 10
        for x in range(math.floor(minX), math.ceil(maxX + 1), granularity):
            for y in range(math.floor(minY), math.ceil(maxY + 1), granularity):
                p = Point(x,y)
                if self.golf_map.contains(p):
                    self.logger.info((x,y))
                    self.grid.append(p)

    def risk_estimation(self, point):
        return 0.0

    def value_estimation(self, target):
        not_visited = set(self.grid)
        def assign_value_est(p, v):
            # generate a circle around the target
            circle = Point(p.x, p.y).buffer(200 + self.skill)
            coveredPoints = []
            for point in self.grid:
                if circle.contains(point):
                    # alpha(risk) + (1-alpha)(ve)
                    distance_to =  point.distance(p)
                    ph = PolygonUtility.point_hash(p)
                    value_estimate = self.risk_estimation(p) + distance_to / 300 + 1 + v
                    if value_estimate < self.valueMap[ph]:
                        self.valueMap[ph] = value_estimate
                        self.graph[ph] = p
                    if point in not_visited:
                        not_visited.remove(point)
                        coveredPoints.append((point, self.valueMap[ph]))
            return sorted(coveredPoints, lambda x: x[1])
        ALPHA = 0.5
        best_locations = assign_value_est(target, 0)
        while len(best_locations) > 0:
            newBest = []
            self.logger.info(len(not_visited))
            for point, value in best_locations:
                if len(not_visited) == 0:
                    break
                newBest.extend(assign_value_est(point, value))
            best_locations = sorted(newBest, lambda x: x[1])





    def get_location_from_shot(self, distance, angle, curr_loc):
        # angle is in rads
        y_delta = distance * math.sin(angle) 
        x_delta = distance * math.cos(angle)
        new_x = curr_loc.x + x_delta
        new_y = curr_loc.y + y_delta
        return Point(new_x, new_y)

    def check_shot(self, distance, angle, curr_loc, roll_factor, golf_map):
        final_location = self.get_location_from_shot(distance * roll_factor, angle, curr_loc)
        drop_location = self.get_location_from_shot(distance, angle, curr_loc)
        return golf_map.contains(final_location) and golf_map.contains(drop_location)

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
        if not self.grid:
            self.create_grid(golf_map)
            self.value_estimation(Point(target.x, target.y))
        required_dist = curr_loc.distance(target)
        roll_factor = 1.0 if required_dist < 20 else 1.1 
        distance = sympy.Min(200+self.skill, required_dist/roll_factor)
        angle = sympy.atan2(target.y - curr_loc.y, target.x - curr_loc.x)
        new_distance, new_angle = self.find_shot(distance, angle, curr_loc, target, roll_factor, self.golf_map)

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

class PolygonUtility:

    @staticmethod
    def convert_sympy_to_shapely(sympy_polygon):
        '''
        return polygon + bounding box
        '''
        points = []
        maxX, minX = float('-inf'), float('inf') 
        maxY, minY = float('-inf'), float('inf')
        for p in sympy_polygon.vertices:
            floatX, floatY = float(p.x), float(p.y)
            maxX = max(maxX, floatX)
            minX = min(minX, floatX)
            maxY = max(maxY, floatY)
            minY = min(minY, floatY)
            points.append((p.x,p.y))
        return Polygon(points), (minX, minY, maxX, maxY)

    @staticmethod
    def point_hash(point):
        return (point.x, point.y)

