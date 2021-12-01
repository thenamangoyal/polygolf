import numpy as np
import sympy
import logging
from typing import Tuple
from shapely.geometry import shape, Polygon
from sympy import Point2D
import math
import matplotlib.pyplot as plt
import constants

import heapq

class Cell:
    def __init__(self,point, target, actual_cost,previous ):
        self.point = point
        self.heuristic_cost = np.linalg.norm(np.array(target).astype(float) - np.array(self.point).astype(float))
        self.actual_cost = actual_cost
        self.previous = previous
    
    def total_cost(self):
        return self.heuristic_cost + self.actual_cost

    def __lt__(self, other):
        return self.total_cost() < other.total_cost()

    def __eq__(self, other):
        return self.point == other.point
    


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
        self.centers =[]
        self.centerset =set()
        self.centers2 = []
        self.target = (0,0)
        self.turns = 0
        self.map = None
    def point_inside_polygon(self,poly, p) -> bool:
    # http://paulbourke.net/geometry/polygonmesh/#insidepoly
        n = len(poly)
        inside = False
        p1 = poly[0]
        for i in range(1, n + 1):
            p2 = poly[i % n]
            if min(p1.y, p2.y) < p.y <= max(p1.y, p2.y) and p.x <= max(p1.x, p2.x) and p1.x != p2.y:
                xints = (p.y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y) + p1.x
                if p1.x == p2.x or p.x <= xints:
                    inside = not inside
            p1 = p2
        return inside
        
    def segmentize_map(self, golf_map ):
        area_length = 5
        beginx = area_length/2
        beginy = area_length/2
        endx = constants.vis_width
        endy = constants.vis_height
        node_centers = []
        node_centers2 =[]   
        for i in range(int(beginx), int(endx), area_length):
            tmp = []    
            for j in range(int(beginy), int(endy), area_length):
                representative_point = Point2D(i,j)
                #maybe if its not in the polygon check points around in order to use those
                if self.point_inside_polygon(golf_map.vertices,sympy.geometry.Point2D(i , j)):
                    tmp.append(representative_point)
                    node_centers.append(representative_point)
                    self.centerset.add((i,j))
                else:
                    tmp.append(None)
            node_centers2.append(tmp)
        self.centers = node_centers
        self.centers2 = node_centers2

    def sector(self, center, start_angle, end_angle, radius):
        def polar_point(origin_point, angle,  distance):
            return [origin_point.x + math.sin(math.radians(angle)) * distance, origin_point.y + math.cos(math.radians(angle)) * distance]
        steps=50
        if start_angle > end_angle:
            t = start_angle
            start_angle = end_angle
            end_angle = t
        else:
            pass
        step_angle_width = (end_angle-start_angle) / steps
        sector_width = (end_angle-start_angle) 
        segment_vertices = []

        segment_vertices.append(polar_point(center, 0,0))
        segment_vertices.append(polar_point(center, start_angle,radius))
        

        for z in range(1, steps):
            segment_vertices.append((polar_point(center, start_angle + z * step_angle_width,radius)))
        segment_vertices.append(polar_point(center, start_angle+sector_width,radius))
        #print(segment_vertices)
        return Polygon(segment_vertices)
    def is_landing_pt_safe(self, iteration: int, golf_map: sympy.Polygon, curr_loc: sympy.geometry.Point2D,
                       distance: float,
                       angle: float) -> bool:

        valid_cnt = 0
        for _ in range(iteration):
            # From golf_game.py
            actual_distance = self.rng.normal(distance, distance / self.skill)
            actual_angle = self.rng.normal(angle, 1 / (2 * self.skill))

            if constants.max_dist + self.skill >= distance >= constants.min_putter_dist:
                landing_point = Point2D(curr_loc[0] + actual_distance * sympy.cos(actual_angle),
                                    curr_loc[1] + actual_distance * sympy.sin(actual_angle))
                final_point = Point2D(
                    curr_loc[0] + (1. + constants.extra_roll) * actual_distance * sympy.cos(actual_angle),
                    curr_loc[1] + (1. + constants.extra_roll) * actual_distance * sympy.sin(actual_angle))
            else:
                landing_point = curr_loc
                final_point = sympy.Point2D(curr_loc[0] + actual_distance * sympy.cos(actual_angle),
                                        curr_loc[1] + actual_distance * sympy.sin(actual_angle))

            segment_land = sympy.geometry.Segment2D(landing_point, final_point)
            if golf_map.encloses(segment_land):
                valid_cnt += 1

        return valid_cnt == iteration
    def positionSafety(self, d, angle, start_point, golf_map):
        #CIRCLE of radiues = 2 standand deviations
        angle_2std = math.degrees(2*(1/self.skill))
        distance_2std = 2*(d/self.skill)
        center = start_point
        print("start ")
        print(angle + angle_2std)
        print("end ")
        print(angle - angle_2std)
        #print("end "+ angle - angle_2std)
        sector1 = self.sector(center, angle + angle_2std, angle - angle_2std, d - distance_2std)
        sector2 = self.sector(center, angle + angle_2std, angle - angle_2std, d + distance_2std )
        probable_landing_region = sector1.intersection(sector2)
        shape_map = golf_map.vertices
        x,y = probable_landing_region.exterior.xy
        shape_map_work = Polygon(shape_map)
        #fig = plt.figure()
        #plt.plot(x,y)
        #plt.show()
        
        area_inside_the_polygon =  (probable_landing_region.intersection(shape_map_work).area)/probable_landing_region.area
        print(area_inside_the_polygon)

    def travel_extra_10percent_safety(self, d, angle, start_point, golf_map):
        angle_2std = math.degrees(2*(1/self.skill))
        distance_2std = 2*(d/self.skill)
        center = start_point
        sector1 = self.sector(center, angle + angle_2std, angle - angle_2std, d + distance_2std + 0.1*d)
        sector2 = self.sector(center, angle + angle_2std, angle - angle_2std, d + distance_2std )
        probable_landing_region = sector1.intersection(sector2)
        shape_map = golf_map.vertices
        x,y = probable_landing_region.exterior.xy
        shape_map_work = Polygon(shape_map)
        return (area_inside_the_polygon)


    def is_neighbour(self, curr_loc, target_loc):
        current_point = curr_loc
        target_point = target_loc
        current_point = np.array(current_point).astype(float)
        target_point = np.array(target_point).astype(float)
        max_dist = 200 + self.skill
        required_dist = np.linalg.norm(current_point - target_point)
        angle = sympy.atan2(target_point[1] - current_point[1], target_point[0] - current_point[0])
        #is reachable
        if (np.linalg.norm(current_point - target_point) < max_dist):
            #is safe to land
            #if (self.is_landing_pt_safe(10,self.map, curr_loc, required_dist, angle )):
            return 1
            #else:
                #return 0
        else:
            return 0

    def adjacent_cells(self, point):
        if self.is_neighbour(point, self.target):
            print('target close!')
            return [self.target]
        neighbours = []
        for center in self.centers:
            if self.is_neighbour(point, center):
                neighbours.append( tuple(center))
        return neighbours

  
    def aStar( self, current, end):
        cur_loc = tuple(current)
        current = Cell(cur_loc, self.target, 0.0 , cur_loc )
        openSet = set()
        openHeap = []
        closedSet = set()
        openSet.add(cur_loc)
        openHeap.append(current)
        while openSet:
            next_pointC = heapq.heappop(openHeap)
            next_point = next_pointC.point
            #print(next_point)
            #reached the goal
            if np.linalg.norm(np.array(self.target).astype(float) - np.array(next_point).astype(float)) <= 5.4 / 100.0:
                while next_pointC.previous.point != cur_loc:
                    next_pointC = next_pointC.previous
                return next_pointC.point
            openSet.remove(next_point)
            closedSet.add(next_point)
            neighbours = self.adjacent_cells(next_point)
            for n in neighbours :
                if n not in closedSet:
                    cell = Cell(n, self.target, next_pointC.actual_cost +1 , next_pointC)
                    if n not in openSet and (next_pointC.actual_cost +1 <10 - self.turns):
                        openSet.add(n)
                        heapq.heappush(openHeap, cell )
        return []




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
        print(prev_loc)
        if (prev_loc == None):
            
            self.segmentize_map(golf_map)
            self.target = tuple(target)
            self.map = golf_map

        next_point = self.aStar(curr_loc, target )


        print(next_point)
        required_dist = curr_loc.distance(next_point)
        angle = sympy.atan2(next_point[1] - curr_loc.y, next_point[0] - curr_loc.x)
        #angle2  = math.degrees(angle)
        #a =  self.positionSafety( distance, angle2, curr_loc.evalf(), golf_map)
        self.turns = self.turns +1
        return (required_dist, angle)
