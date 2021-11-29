import numpy as np
import sympy
from shapely.geometry import Polygon, Point
import math
import logging
from typing import Tuple
from collections import defaultdict
import time

log_time = True


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

        self.shapely_poly = None
        self.graph = {}
        self.all_nodes_center = {}
        self.needs_edge_init = True

    def validate_node(self, x, y, step):
        """ Function which determines if a node of size step x step centered at (x, y) is a valid node in our 
        self.shapely_poly 

        Args:
            x (float): x-coordinate of node
            y (float): y-coordinate of node
            step (float): size of node

        Returns:
            Boolean: True if node is valid in our map
         """

        # 1. Node center must be inside graph
        valid_edge = 0
        if self.shapely_poly.contains(Point(x, y)):
            # 2. 7/8 points on edge of node must be in graph (we'll count as 8/9, including center)
            for i in np.arange(y - (step / 2), y + step, step / 2):
                for j in np.arange(x - (step / 2), x + step, step / 2):
                    if self.shapely_poly.contains(Point(j, i)):
                        valid_edge += 1
            # return True
        if valid_edge >= 8:
            # print("valid")
            return True
        else:
            return False

    def construct_nodes(self, golf_map, leftest, rightest, highest, lowest, step, target):
        """Function which creates a graph on self.graph based on constraints, with valid nodes

        Args:
            golf_map (sympy.Polygon): Golf Map polygon
            leftest, rightest, highest, lowest (int): limits of map
            step (float): size of node
            target (sympy.geometry.Point2D): Target location
        """
        since = time.time()
        self.graph = {'curr_loc': []}
        self.shapely_poly = Polygon([(p.x, p.y) for p in golf_map.vertices])

        """ Graph Creation: Nodes
        - Goes from left lower to right upper of designated range
        - Select nodes of size (step x step)
            1. Node center must be inside graph
            2. 7 / 8 points on edge of node must be in graph (we'll count as 8/9, including center)
            3. nodes can extend horizontally to create one big node
        - Each node is stored as [(x0, y0), (x1, y1), ...] or simply [(x0, y0)]
        """
        for y in np.arange(lowest + (step / 2), highest + step, step):
            # 3. nodes can extend horizontally to create one big node
            x = leftest + (step / 2)
            while x < rightest + step:

                if self.validate_node(x, y, step):
                    more_nodes = 0
                    while self.validate_node(x + step, y, step):
                        more_nodes += 1
                        x += step

                    # center_x = x - (more_nodes * step) / 2
                    left = x - (more_nodes * step)
                    right = x + step
                    new_key = ()
                    for i in np.arange(left, right, step):
                        new_key = new_key + ((i, y),)

                    self.graph[new_key] = []
                    self.all_nodes_center[new_key] = self._get_node_center(new_key)
                x += step
            """ for x in np.arange(leftest, rightest + step, step):
                if (self.validate_node(x, y, step)):
                    graph[(x, y)] = [] """

        # add target point as node
        self.graph[((target.x, target.y),)] = []
        self.all_nodes_center[((target.x, target.y),)] = self._get_node_center(((target.x, target.y),))

        if log_time:
            print("time for construct_nodes:", time.time() - since)

    def construct_edges(self, curr_loc, only_construct_from_source=False):
        """Graph Creation: Edges
        - In short, we construct directional Edge e: (n1, n2) if our skill level allows us to reach n2 from n1
        - For edges going from:
            - the Node containing the current position:
            use the exact coordinate for the current position as the origin of our circular range
            - a Node that doesn’t contain the current position:
            use the midpoint of that Node (not the midpoint of some unit grid within the Node) as the origin of our
            circular range
        """
        since = time.time()
        source_completed = False

        for from_node in self.graph.keys():

            # constructing an Edge from curr_loc to another non-curr_loc Node
            if from_node == 'curr_loc':
                # clear existing adjacency list of this from_node
                self.graph[from_node] = []

                for to_node in self.graph.keys():
                    if to_node == from_node:  # 'curr_loc' can't have an Edge with itself
                        continue
                    can_reach_one_unit = False  # whether at least one Unit in to_node is reachable from from_node
                    for unit_center in to_node:
                        if not isinstance(unit_center, tuple):
                            print(unit_center)
                        if curr_loc.distance(
                                sympy.geometry.Point2D(unit_center)) > self.skill:  # if outside our skill range
                            continue
                        can_reach_one_unit = True
                        break

                    # if Edge (from_node, to_node) is valid, add to from_node's adjacency list
                    if can_reach_one_unit:
                        self.graph[from_node].append(to_node)

                source_completed = True

            # constructing an Edge from a non-curr_loc Node to another non-curr_loc Node
            else:
                if only_construct_from_source and source_completed:  # if only constructing from source, skip this part
                    break

                # clear existing adjacency list of this from_node
                self.graph[from_node] = []

                for to_node in self.graph.keys():
                    # never treat 'curr_loc' as a destination Node; from_node and to_node need to be different
                    if to_node == 'curr_loc' or to_node == from_node:
                        continue
                    from_node_center = self.all_nodes_center[from_node]
                    to_node_center = self.all_nodes_center[to_node]

                    # if the distance between the two Node centers is reachable, add to from_node's adjacency list
                    if sympy.geometry.Point2D(from_node_center).distance(
                            sympy.geometry.Point2D(to_node_center)) <= self.skill:
                        self.graph[from_node].append(to_node)

        if log_time:
            print("time for construct_edges:", time.time() - since)

    @staticmethod
    def _get_node_center(unit_centers):
        if len(unit_centers) % 2:  # if number of Units is odd, take the middle Unit center as the Node center
            node_center = unit_centers[int(len(unit_centers) / 2)]
        else:  # if number of Units is even, average the middle two Unit centers as the Node center
            mid_left_unit_center = unit_centers[int(len(unit_centers) / 2) - 1]
            mid_right_unit_center = unit_centers[int(len(unit_centers) / 2)]
            node_center_x = (mid_left_unit_center[0] + mid_right_unit_center[0]) / 2
            node_center_y = (mid_left_unit_center[1] + mid_right_unit_center[1]) / 2
            node_center = (node_center_x, node_center_y)
        return node_center

    def BFS(self, target):
        """Function that performs BFS on the graph of nodes to find a path to the target, prioritizing minimum
        moves in order to minimize our score.
        
        Returns:
            Node to make a move towards.
        """
        since = time.time()

        visited = defaultdict(int)
        compare_target = ((target.x, target.y),)
        queue = []
        queue.append(['curr_loc'])
        final_path = []
        while queue:
            path = queue.pop(0)
            node = path[-1]
            if node == compare_target:
                # print(path)
                final_path = path
                break

            if visited[node] > 0 and visited[node] <= len(path):
                continue

            visited[node] = len(path)

            adj = self.graph[node]
            for a in adj:
                new_path = list(path)
                new_path.append(a)
                queue.append(new_path)
        if len(final_path) < 2:
            return "default"
        move = final_path[1]

        if log_time:
            print("time for bfs:", time.time() - since)
        return sympy.geometry.Point2D(self.all_nodes_center[move][0], self.all_nodes_center[move][1])

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

        required_dist = curr_loc.distance(target)
        """ 
        Case 1: required_dist > 20m
            - create a map with 1m-sized nodes

        Case 2: requier_dist < 20m (putting dist)
            - more specific map with .05m-sized nodes 20m around from current point
         """

        # Case 1: required_dist > 20m
        if required_dist > 20:
            step = 10
            # Detect edges of map
            map_ver = golf_map.vertices
            center = golf_map.centroid
            leftest = math.floor(center[0])
            rightest = math.floor(center[0])
            highest = math.floor(center[1])
            lowest = math.floor(center[1])
            for point in map_ver:
                # leftest
                if point[0] < leftest:
                    leftest = math.floor(point[0])
                # rightest
                if point[0] > rightest:
                    rightest = math.ceil(point[0])
                # highest
                if point[1] > highest:
                    highest = math.ceil(point[1])
                # lowest
                if point[1] < lowest:
                    lowest = math.floor(point[1])

            if score == 1:
                self.construct_nodes(golf_map, leftest, rightest, highest, lowest, step, target)
                if self.needs_edge_init:
                    self.construct_edges(curr_loc, only_construct_from_source=False)  # construct all edges
                    self.needs_edge_init = False
            else:
                self.construct_edges(curr_loc,
                                     only_construct_from_source=True)  # only construct outgoing edges of curr_loc

        # Case 2: required_dist < 20m (putting dist)
        else:
            step = 0.05
            leftest = math.floor(curr_loc.x - 20)
            rightest = math.ceil(curr_loc.x + 20)
            highest = math.ceil(curr_loc.y + 20)
            lowest = math.floor(curr_loc.y - 20)
            self.construct_nodes(golf_map, leftest, rightest, highest, lowest, step, target)
            self.construct_edges(curr_loc, only_construct_from_source=False)
        move = self.BFS(target)

        roll_factor = 1.1
        if required_dist < 20:
            roll_factor = 1.0

        if move == "default":
            distance = sympy.Min(200 + self.skill, required_dist / roll_factor)
            angle = sympy.atan2(target.y - curr_loc.y, target.x - curr_loc.x)
            return (distance, angle)

        distance = curr_loc.distance(move) / roll_factor
        angle = sympy.atan2(move.y - curr_loc.y, move.x - curr_loc.x)
        # distance = sympy.Min(200 + self.skill, required_dist / roll_factor)
        # angle = sympy.atan2(target.y - curr_loc.y, target.x - curr_loc.x)
        return (distance, angle)
