import numpy as np
import sympy
import logging
from typing import Tuple
from shapely.geometry import shape, Polygon, LineString , Point
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
    def __hash__(self):
        return hash(self.point)
    


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
        self.map_shapely =None
        self.initial_path = []
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

   
        
    def is_safe(self, d, angle, start_point):
        #to do add confidence bounds
        angle_2std = ((1/(self.skill)))
        distance_2std = (2*d/self.skill)
        min_distance = d-distance_2std
        max_distance = d+(d*0.1)+distance_2std
        begin_line1 = (start_point.x + (min_distance)*math.cos(angle - angle_2std ), start_point.y + (min_distance)*math.sin(angle -angle_2std ))
        begin_line2 = (start_point.x + (min_distance)*math.cos(angle + angle_2std), start_point.y + (min_distance)*math.sin(angle + angle_2std))
        end_line1 = (start_point.x + (max_distance)*math.cos(angle - angle_2std ), start_point.y + (max_distance)*math.sin(angle - angle_2std))
        end_line2 = (start_point.x + (max_distance)*math.cos(angle + angle_2std ), start_point.y + (max_distance)*math.sin(angle + angle_2std))
        L1 = LineString([Point(begin_line1), Point(end_line1)])
        L2 = LineString([Point(begin_line2), Point(end_line2)])
        check1 = L1.within(self.map_shapely)
        check2 = L2.within(self.map_shapely)
        xs=[]
        ys=[]
        step = 2*angle_2std/4
        angles = [angle - angle_2std +step , angle - angle_2std +2*step ,angle - angle_2std +3*step ]
        p = []
        for a in angles:
            x = start_point.x + max_distance * math.cos(a)
            y = start_point.y + max_distance * math.sin(a)
            p.append(Point2D(x,y))

        for a in reversed(angles):
            x = start_point.x + min_distance * math.cos(a)
            y = start_point.y + min_distance * math.sin(a)
            p.append(Point2D(x,y))
        contains =0
        for i in p:
            if(self.point_inside_polygon(self.map.vertices, i)):
                contains +=1
        if (check1 &   check2):
            return 1
        else:
            return 0



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
            #if(Point2D(self.target).equals(Point2D(target_loc))):
                #return 1
            if (self.is_safe(required_dist,angle,Point2D(curr_loc))):
                return 1
            else:
                return 0
            #return 1
        else:
            return 0

    def adjacent_cells(self, point, closedSet):
        neighbours = []
        if self.is_neighbour(point, self.target):
            print('target close!')
            neighbours.append(self.target)
        for center in self.centers:
            print("here")
            if center.equals(Point2D(point)):
                continue
            if tuple(center) in closedSet:
                continue
            if self.is_neighbour(point, center):
                neighbours.append( tuple(center))
        return neighbours

  
    def aStar( self, current, end):
        self.initial_path =[]
        cur_loc = tuple(current)
        current = Cell(cur_loc, self.target, 0.0 , cur_loc )
        openSet = set()
        node_dict = {}
        node_dict[(cur_loc)] = 0.0
        openHeap = []
        closedSet = set()
        openSet.add(cur_loc)
        openHeap.append(current)
        while len(openHeap)>0:
            next_pointC = heapq.heappop(openHeap)
            next_point = next_pointC.point
            print(next_point)
            #reached the goal
            if np.linalg.norm(np.array(self.target).astype(float) - np.array(next_point).astype(float)) <= 5.4 / 100.0:
                while next_pointC.previous.point != cur_loc:
                    self.initial_path.append(next_pointC.point)
                    next_pointC = next_pointC.previous
                self.initial_path.reverse()
                return next_pointC.point
            openSet.remove(next_point)
            closedSet.add(next_point)
            neighbours = self.adjacent_cells(next_point, closedSet)
            for n in neighbours :
                if n not in closedSet:
                    cell = Cell(n, self.target, next_pointC.actual_cost +1 , next_pointC)
                    if n not in openSet and (next_pointC.actual_cost +1 <=10 - self.turns):
                        if (n not in node_dict or cell.total_cost() < node_dict(n)):
                            openSet.add(n)
                            node_dict[n] = cell.total_cost()
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
        if (prev_loc == None):
            
            self.segmentize_map(golf_map)
            self.target = tuple(target)
            self.map = golf_map
            shape_map = golf_map.vertices 
            self.map_shapely = Polygon(shape_map)
        print(self.centers)
        if(self.turns>0):
            next_point = self.initial_path[0] if len(self.initial_path)>0 else self.target
            print(next_point)
            if len(self.initial_path)>0:
                del self.initial_path[0]
            required_dist = curr_loc.distance(next_point)
            print(required_dist)
            angle = sympy.atan2(next_point[1] - curr_loc.y, next_point[0] - curr_loc.x)
            if (self.is_safe(required_dist,angle,curr_loc) and required_dist<=200+self.skill):
                self.turns = self.turns +1  
                if (next_point[1] == self.target[1] and next_point[0] == self.target[0]):
                    if(required_dist>20):
                        required_dist = 0.9*required_dist
                return (required_dist, angle)
            else:
                next_point = self.aStar(curr_loc, target )
                required_dist = curr_loc.distance(next_point)
                angle = sympy.atan2(next_point[1] - curr_loc.y, next_point[0] - curr_loc.x)
                if (next_point[1] == self.target[1] and next_point[0] == self.target[0]):
                    if(required_dist>20):
                        required_dist = 0.9*required_dist

                self.turns = self.turns +1  
                return (required_dist, angle)
        else:
            next_point = self.aStar(curr_loc, target )
            print(next_point[0])
            required_dist = curr_loc.distance(next_point)
            angle = sympy.atan2(next_point[1] - curr_loc.y, next_point[0] - curr_loc.x)
            if (next_point[1] == self.target[1] and next_point[0] == self.target[0]):
                if(required_dist>20):
                    required_dist = 0.9*required_dist

            self.turns = self.turns +1  
            print(next_point)
            return (required_dist, angle)