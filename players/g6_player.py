import numpy as np
import sympy
from sympy import Point, Polygon
import math
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

    def validate_node(self, golf_map, x, y, step):
        """ Function which determines if a node of size step x step centered at (x, y) is a valid node in our 
        golf_map

        Args:
            golf_map (sympy.Polygon): Golf Map polygon
            x (float): x-coordinate of node
            y (float): y-coordinate of node
            step (float): size of node
         """

        # 1. Node center must be inside graph
        valid_edge = 0
        if (golf_map.encloses_point(Point(x, y))):
            # 2. 7/8 points on edge of node must be in graph (we'll count as 8/9, including center)
            for i in np.arange(y - (step / 2), y + step, step / 2):
                for j in np.arange(x - (step / 2), x + step, step / 2):
                    if (golf_map.encloses_point(Point(j , i))):
                        valid_edge += 1
            #return True
        if (valid_edge >= 8):
            return True
        else:
            return False
        

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
        """ 
        Case 1: required_dist > 20m
            - create a map with 1m-sized nodes

        Case 2: requier_dist < 20m (putting dist)
            - more specific map with .05m-sized nodes 20m around from current point
         """

        # Case 1: required_dist > 20m
        if (required_dist > 20):
            step = 20
            # Detect edges of map
            map_ver = golf_map.vertices
            center = golf_map.centroid
            leftest = math.floor(center[0])
            rightest = math.floor(center[0])
            highest = math.floor(center[1])
            lowest = math.floor(center[1])
            for point in map_ver:
                # leftest
                if (point[0] < leftest):
                    leftest = math.floor(point[0])
                # rightest
                if (point[0] > rightest):
                    rightest = math.ceil(point[0])
                # highest
                if (point[1] > highest):
                    highest = math.ceil(point[1])
                # lowest
                if (point[1] < lowest):
                    lowest = math.floor(point[1])
        
        # Case 2: requier_dist < 20m (putting dist)
        else: 
            step = 0.5
            leftest = math.floor(curr_loc.x - 20)
            rightest = math.ceil(curr_loc.x + 20)
            highest = math.ceil(curr_loc.y + 20)
            lowest = math.floor(curr_loc.y - 20)


        """ Graph Creation
        - Goes from left lower to right upper of designated range
        - Select nodes of size (step x step)
            1. Node center must be inside graph
            2. 7 / 8 points on edge of node must be in graph (we'll count as 8/9, including center)
            3. nodes can extend horizontally to create one big node
        """
        graph = {}

        for y in np.arange(lowest, highest + step, step):
            print("this y: ", y)
            print("x range: ", [leftest, rightest])
            # 3. nodes can extend horizontally to create one big node
            x = leftest
            while(x < rightest + step):
                
                if (self.validate_node(golf_map, x, y, step)):
                    print("this x: ", x)
                    more_nodes = 0
                    while (self.validate_node(golf_map, x + step, y, step)):
                        more_nodes += 1
                        x += step
                
                    center_x = x - (more_nodes * step) / 2

                    graph[Point(center_x, y)] = []
                x += step
        
        # add target point as node
        graph[target] = []

        roll_factor = 1.1
        if required_dist < 20:
            roll_factor  = 1.0
        distance = sympy.Min(200+self.skill, required_dist/roll_factor)
        angle = sympy.atan2(target.y - curr_loc.y, target.x - curr_loc.x)
        return (distance, angle)
