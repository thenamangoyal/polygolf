import numpy as np
import sympy
import logging
from typing import Tuple
def point_inside_polygon(poly, p) -> bool:
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
        self.maplimit = None
        self.modified_goal = None
        self.modified_start = None
        self.unit = 10
    def generate_grid(self,map,target,current,unit):
        maxx,maxy = 0.0,0.0
        for point in map.vertices:
            if point.x>maxx:
                maxx = point.x
            if point.y>maxy:
                maxy = point.y
        maxx = unit * int(maxx/unit+1) * 1.0
        maxy = unit * int(maxy/unit+1) * 1.0
        self.maplimit = sympy.geometry.Point2D(maxx, maxy)
        self.modified_goal = sympy.geometry.Point2D(int(target.x / unit) * unit * 1.0, int(target.y / unit) * unit * 1.0)
        self.modified_start = sympy.geometry.Point2D(int(current.x / unit) * unit * 1.0, int(current.y / unit) * unit * 1.0)

    def generate_shortest_path(self,map):
        movement = [(1,0),(-1,0),(0,1),(0,-1)]
        dist = []
        for i in range(0,self.maplimit.x,self.unit):
            temp = []
            for j in range(0, self.maplimit.y, self.unit):
                if point_inside_polygon(map.vertices,sympy.geometry.Point2D(i , j)):
                    temp.append(-1)
                else:
                    temp.append(-2)
            dist.append(temp)
        dist[int(self.modified_start.x / self.unit)][int(self.modified_start.y/self.unit)] = 0
        dist[int(self.modified_goal.x / self.unit)][int(self.modified_start.y / self.unit)] = -1
        current_points = [self.modified_start]
        next_points = []
        while dist[int(self.modified_goal.x / self.unit)][int(self.modified_goal.y / self.unit)]==-1:
            for current in current_points:
                for move in movement:
                    next_point = sympy.geometry.Point2D(current.x + move[0]*self.unit,current.y + move[1]*self.unit)
                    if int(next_point.x / self.unit)< len(dist) and int(next_point.x / self.unit)>=0 and \
                            int(next_point.y / self.unit) < len(dist[0]) and int(next_point.y / self.unit)>=0 and \
                            dist[int(next_point.x / self.unit)][int(next_point.y / self.unit)] == -1:
                        dist[int(next_point.x / self.unit)][int(next_point.y / self.unit)] = dist[int(current.x / self.unit)][int(current.y / self.unit)] + 1
                        next_points.append(next_point)
            current_points = next_points.copy()
            next_points = []
#-m maps/default/simple.json maps/g1/g1_map.json -p 1
        path = []
        current = [int(self.modified_goal.x / self.unit),int(self.modified_goal.y / self.unit)]
        path.append(sympy.geometry.Point2D(current[0]*self.unit*1.0,current[1]*self.unit*1.0))
        while dist[current[0]][current[1]]!=0:
            for move in movement:
                if current[0]-move[0]>=0 and current[0]-move[0]<len(dist) and \
                    current[1] - move[1] >= 0 and current[1] - move[1] < len(dist):
                    if dist[current[0]][current[1]] - 1 == dist[current[0]-move[0]][current[1]-move[1]]:
                        current[0] -= move[0]
                        current[1] -= move[1]
                        path.append(sympy.geometry.Point2D(current[0]*self.unit*1.0,current[1]*self.unit*1.0))
                        break
        path.reverse()

        self.shortest_path = path
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
        if self.maplimit == None:
            self.generate_grid(golf_map,target,curr_loc,self.unit)
            self.generate_shortest_path(golf_map)



        required_dist = curr_loc.distance(target)
        roll_factor = 1.1
        if required_dist < 20:
            roll_factor  = 1.0
            distance = required_dist
            angle = sympy.atan2(target.y - curr_loc.y, target.x - curr_loc.x)
            return (distance, angle)
        else:
            next_point = target
            for point in self.shortest_path:
                if curr_loc.distance(point) <= 200 + self.skill - 2*(200 + self.skill)/self.skill:
                    next_point = point
            distance = curr_loc.distance(next_point)
            angle = sympy.atan2(next_point.y - curr_loc.y, next_point.x - curr_loc.x)
            return (distance, angle)


