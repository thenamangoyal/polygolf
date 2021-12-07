import os
import pickle
import numpy as np
import sympy
import logging
from typing import Tuple
import shapely
import random
from matplotlib import pyplot as plt
from collections import deque
import math
from queue import PriorityQueue
import heapq
from heapq import heappush, heappop

class Player:
    def __init__(self, skill: int, rng: np.random.Generator, logger: logging.Logger, golf_map: sympy.Polygon, start: sympy.geometry.Point2D, target: sympy.geometry.Point2D, map_path: str, precomp_dir: str) -> None:
        """Initialise the player with given skill.

        Args:
            skill (int): skill of your player
            rng (np.random.Generator): numpy random number generator, use this for same player behvior across run
            logger (logging.Logger): logger use this like logger.info("message")
            golf_map (sympy.Polygon): Golf Map polygon
            start (sympy.geometry.Point2D): Start location
            target (sympy.geometry.Point2D): Target location
            map_path (str): File path to map
            precomp_dir (str): Directory path to store/load precomputation
        """
        # # if depends on skill
        # precomp_path = os.path.join(precomp_dir, "{}_skill-{}.pkl".format(map_path, skill))
        # # if doesn't depend on skill
        # precomp_path = os.path.join(precomp_dir, "{}.pkl".format(map_path))
        
        # # precompute check
        # if os.path.isfile(precomp_path):
        #     # Getting back the objects:
        #     with open(precomp_path, "rb") as f:
        #         self.obj0, self.obj1, self.obj2 = pickle.load(f)
        # else:
        #     # Compute objects to store
        #     self.obj0, self.obj1, self.obj2 = _

        #     # Dump the objects
        #     with open(precomp_path, 'wb') as f:
        #         pickle.dump([self.obj0, self.obj1, self.obj2], f)
        self.skill = skill
        self.rng = rng
        self.logger = logger
        self.quick_map = None
        self.dmap = None
        self.pmap = None
        self.cell_width = 1
        self.rows = None
        self.cols = None
        self.max_distance = 200 + skill
        self.distances = [200 + skill, (200 + skill) / 2, (200 + skill) / 4]
        self.angles = [0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 3 * np.pi / 4, 5 * np.pi / 6, np.pi, 7 * np.pi / 6, 5 * np.pi / 4, 4 * np.pi / 3, 3 * np.pi / 2, 5 * np.pi / 3, 7 * np.pi / 4, 11 * np.pi / 6]
        self.path = None
        self.tmp = 'fire.txt'
        
    def get_landing_point(self, curr_loc: shapely.geometry.Point, distance: float, angle: float):
        """
    	    Args:
    	        curr_loc (sympy.geometry.Point2D): current location
    	        distance (float): The distance to next potential landing point
    	        angle (float): The angle coordinate to the next potential landing point
    	    Returns:
    	        the potential landing point as a sympy.Point2D object
    	"""
        return shapely.geometry.Point(curr_loc.x + distance * np.cos(angle), curr_loc.y + distance * np.sin(angle))

    
    def get_center(self, r: int, c: int):
        x = self.zero_center.x + c * self.cell_width
        y = self.zero_center.y - r * self.cell_width
        return shapely.geometry.Point(x, y)

    def get_row_col(self, x, y): #will return row, col closest to point (if within dmap)
        c = round((x - self.zero_center.x)/self.cell_width)
        r = round((self.zero_center.y - y)/self.cell_width)

        return r, c

    def get_corners(self, r: int, c: int):
        center = self.get_center(r,c)
        offset = self.cell_width / 2
        x = center.x
        y = center.y
        upper_left = shapely.geometry.Point(x - offset, y + offset)
        upper_right = shapely.geometry.Point(x + offset, y + offset)
        lower_left = shapely.geometry.Point(x - offset, y - offset)
        lower_right = shapely.geometry.Point(x + offset, y - offset)
        return [upper_left, upper_right, lower_left, lower_right]

    def in_bounds(self, row, col):
        if row >= 0 and col >=0 and row < self.rows and col < self.cols:
            return True
        return False

    def get_neighbors(self, row, col):
        neighbors_list = []
        for i in range(-1,2):
            for j in range(-1,2):
                if i == 0 and j == 0:
                    continue
                neighborRow = row + i
                neighborCol = col + j

                if self.in_bounds(neighborRow, neighborCol):
                    neighbors_list.append((neighborRow, neighborCol))

        return neighbors_list



    def brushfire(self, q):
        # print("starting brushfire")
        while(len(q) != 0):
            row, col, dist, waterPoint = q.popleft()

            neighbors_list = self.get_neighbors(row, col)
            for nRow, nCol in neighbors_list:
                if self.dmap[nRow][nCol] == 1: #is on land
                    nCenter = self.get_center(nRow, nCol)
                    ndist = nCenter.distance(waterPoint) #distance from neighbor to water
                    pValue = self.pmap[nRow, nCol]
                    if (pValue == -1) or (pValue > ndist): #either uninitialized or current distance is smaller
                        self.pmap[nRow, nCol] = ndist
                        q.append((nRow, nCol, ndist, waterPoint))


    def p_in_water(self,start, end):
        d = start.distance(end)
        sigma = d/self.skill
        r, c = self.get_row_col(end.x, end.y)
        dw = self.pmap[r, c] #distance to water for end point
        alpha = dw/sigma #number of standard deviations to water for end point
        if 1 - alpha > 0.999:
            alpha = 0.001

        if alpha <= 1:
            p = 1 - (1.848*math.exp(-1/alpha))
            if 1 - p < 0.001:
                return 0.999
            else:
                return p
        elif alpha < 2:
            return 1 - (0.68 + 0.27*(1- alpha))
        else:
            return 0

    def expected_strokes(self, start, end):
        ratio = self.p_in_water(start,end)

        return 1 + (ratio)/(1-ratio)

    
    def precompute(self):
        minx, miny, maxx, maxy = self.quick_map.bounds
        self.minx = minx
        self.miny = miny


        testpt = shapely.geometry.Point(50,160)

        width = maxx - minx
        height = maxy - miny
        self.cols = int(np.ceil(width / self.cell_width))
        self.rows = int(np.ceil(height / self.cell_width))
        self.zero_center = shapely.geometry.Point(minx + self.cell_width / 2, maxy - self.cell_width / 2)

        self.dmap = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.pmap = np.zeros((self.rows, self.cols), dtype=np.int8)


        q = deque() #used for brushfire

        for row in range(self.rows):
            for col in range(self.cols):
                corners = self.get_corners(row, col)
                water = 0
                land = 0
                for point in corners:
                    if self.quick_map.contains(point):
                        land += 1
                    else:
                        water += 1

                if land == 4:
                    # if all four points on land, then set dmap to 1
                    self.dmap[row, col] = 1
                    self.pmap[row, col] = -1

                elif water == 4:
                    # if all four points on water, then set dmap to 0
                    self.dmap[row, col] = 0
                    self.pmap[row, col] = -1

                else:
                    # in else ==> some points on land, some in water ==> we are on an edge cell
                    self.pmap[row, col] = 0
                    self.dmap[row, col] = 0
                    q.append((row,col,0, self.get_center(row, col))) #place edge in queue

        ### Commented out code saves precomputed brushfire map for testing purposes
        # if os.path.exists(self.tmp):
        #     with open(self.tmp, 'rb') as f:
        #         self.pmap = np.load(f)
        #         print("loaded from file")
        #         return
        self.brushfire(q)
        # with open(self.tmp, 'wb') as f:
        #     np.save(f, self.pmap)


    # Generate potential branches for the A* searching algorithm
    def generate_branches(self, points):
        pt = points[-1]
        li = []
        for distance in self.distances:
            for angle in self.angles:
                new_point = self.get_landing_point(pt, distance, angle)
                valid = True
                for i in range(len(points) - 1):
                    if points[i].distance(new_point) < (self.max_distance/2):
                        valid = False
                        break
                if valid:
                    li.append(new_point)

        return li

    def a_star(self, start, target):
        count = 0
        secondCount = 0
        # print("starting a star")

        #pq = PriorityQueue() #convention for data insertion is [heuristic, path, estimated total strokes for this path]

        E = start.distance(target)/(200 + self.skill)
        s = 0 #no shots taken so far
        path = [start]
        #pq.put([E, path, 0])
        heap = [(E, secondCount, 0, path)]

        while heap:
            count += 1
            h, _, strokes, path = heappop(heap)

            lastPoint = path[-1]
            if lastPoint.distance(target) <= 200 + self.skill:
                # print("found path")
                # print(count)
                path.append(target)
                return path


            possibleShots = self.generate_branches(path)


            for shot in possibleShots:
                secondCount += 1

                r,c = self.get_row_col(shot.x, shot.y)
                if(self.in_bounds(r, c)):
                    s = strokes + self.expected_strokes(lastPoint, shot)
                    d = shot.distance(target)
                    E = d/(200 + self.skill)
                    h = s + E #heuristic
                    newPath = path.copy()
                    newPath.append(shot)
                    heappush(heap, (h, secondCount, s, newPath))







    
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
        if not self.quick_map:
            self.quick_map = shapely.geometry.Polygon([(p.x,p.y) for p in golf_map.vertices])
            self.precompute()

        if not self.path:
            s = shapely.geometry.Point(curr_loc.x, curr_loc.y)
            t = shapely.geometry.Point(target.x, target.y)
            self.path = self.a_star(s, t)

        # for p in self.path:
        #     print(p.x, p.y)
        #     print()
        # print(self.path)

        # Testing
        count = 0
        # .bounds function can get (minx, miny, maxx, maxy) tuple (float values) that bounds the object

        s = shapely.geometry.Point(curr_loc.x, curr_loc.y)
        t = shapely.geometry.Point(target.x, target.y)
        if s.distance(t) <= self.max_distance:
            angle = sympy.atan2(target.y - curr_loc.y, target.x - curr_loc.x)
            distance = s.distance(t)
            return (distance, angle)

        minDist = float('inf')
        closestI = None
        for i in range(len(self.path)):
            if s.distance(self.path[i]) < minDist:
                minDist, closestI = s.distance(self.path[i]), i
        pt = self.path[closestI + 1]

        angle = sympy.atan2(pt.y - curr_loc.y, pt.x - curr_loc.x)
        distance = min(self.max_distance, s.distance(self.path[closestI + 1]))

        return (distance, angle)
