import numpy as np
import sympy
from shapely.geometry import Polygon, Point
import math
import logging
from typing import Tuple
from collections import defaultdict
import time

DEBUG_MSG = True  # enable print messages


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
        self.graph = {}  # self.graph[node_i] contains a list of edges where each edge_j = (node_j, weight, f_count)
        # self.all_nodes_center = {}
        self.needs_edge_init = True
        self.critical_pts = []

    def draw_skeleton(self, polygon, skeleton, show_time=False):
        draw(polygon)
        self.critical_pts = []
        for v in skeleton.vertices:
            if v.point not in polygon.vertices:
                self.critical_pts.append([float(v.point.x()), float(v.point.y())])

        out_count = 0
        for point in self.critical_pts:
            if not self.shapely_poly.contains(Point(point[0], point[1])):
                out_count += 1
        print("out count: ", str(out_count))
        if out_count:
            skel = sg.skeleton.create_exterior_straight_skeleton(self.scikit_poly, 0.1)
            self.draw_skeleton(self.scikit_poly, skel)

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

        for point in self.critical_pts:
            if self.shapely_poly.contains(Point(point[0], point[1])):
                self.graph[(point[0], point[1])] = []
        # add target point as node
        self.graph[(float(target.x), float(target.y))] = []

        if DEBUG_MSG:
            print("time for construct_nodes:", time.time() - since)

    def construct_more_nodes(self, curr_loc):
        """Function that builds 'bridges nodes' nearby polygon edges
        to provide options for crossing water.

        Args:
            curr_loc (sympy.geometry.Point2D): Current location
        """
        since = time.time()

        runs = 1  # change based on need
        skill_dist_range = 200 + self.skill
        hyp = 0.5  # vary this based on skill?

        for i in range(runs):
            new_nodes = []
            for from_node in self.graph.keys():
                if from_node == 'curr_loc':
                    for to_node in self.graph.keys():
                        if to_node == from_node:  # 'curr_loc' can't have an Edge with itself
                            continue
                        distance = self._euc_dist((int(curr_loc.x), int(curr_loc.y)), to_node)
                        line = LineString([(int(curr_loc.x), int(curr_loc.y)), to_node])

                        # a. If edge_len < 200: keep as is, regardless of going over water
                        if distance < skill_dist_range:
                            continue

                        # b. If edge_len > 200: check if it goes over water
                        # via line intersecting map
                        else:
                            # i. If yes, calculate bank_distance for this edge
                            if self.shapely_edges.intersects(line):
                                # this should be a LineString obj with 2n points
                                intersection = list(self.shapely_edges.intersection(line).geoms)

                                for i in range(int(len(intersection) / 2)):
                                    inter_0 = list(intersection[2 * i].coords)[0]
                                    inter_1 = list(intersection[(2 * i) + 1].coords)[0]
                                    bank_distance = self._euc_dist(inter_0, inter_1)

                                    # - bank_distance > 200: discard edge
                                    if bank_distance > skill_dist_range:
                                        continue
                                    # - bank_distance < 200: create two new nodes at the edge of bank 
                                    else:
                                        # delta y
                                        d_y = inter_1[1] - inter_0[1]
                                        # delta x
                                        d_x = inter_1[0] - inter_0[0]

                                        theta = math.atan(d_y / d_x)

                                        offset_y = hyp * math.sin(theta)
                                        offset_x = hyp * math.cos(theta)

                                        bridge_0 = (inter_0[0] - offset_x, inter_0[1] - offset_y)
                                        bridge_1 = (inter_1[0] + offset_x, inter_1[1] + offset_y)
                                        if (self.shapely_poly.contains(Point(bridge_0[0], bridge_0[1])) and
                                                self.shapely_poly.contains(Point(bridge_1[0], bridge_1[1]))):
                                            new_nodes.append(bridge_0)
                                            new_nodes.append(bridge_1)

                else:
                    for to_node in self.graph.keys():
                        if to_node == 'curr_loc' or to_node == from_node:
                            continue
                        distance = self._euc_dist(from_node, to_node)
                        line = LineString([from_node, to_node])
                        # a. If edge_len < 200: keep as is, regardless of going over water
                        if distance < skill_dist_range:
                            continue

                        # b. If edge_len > 200: check if it goes over water
                        # via line intersecting map
                        else:
                            # i. If yes, calculate bank_distance for this edge
                            if self.shapely_edges.intersects(line):
                                intersection = list(self.shapely_edges.intersection(line).geoms)

                                for i in range(int(len(intersection) / 2)):
                                    inter_0 = list(intersection[2 * i].coords)[0]
                                    inter_1 = list(intersection[(2 * i) + 1].coords)[0]
                                    bank_distance = self._euc_dist(inter_0, inter_1)

                                    # - bank_distance > 200: discard edge
                                    if bank_distance > skill_dist_range:
                                        continue
                                    # - bank_distance < 200: create two new nodes at the edge of bank 
                                    else:
                                        # delta y
                                        d_y = inter_1[1] - inter_0[1]
                                        # delta x
                                        d_x = inter_1[0] - inter_0[0]

                                        theta = math.atan(d_y / d_x)

                                        offset_y = hyp * math.sin(theta)
                                        offset_x = hyp * math.cos(theta)

                                        bridge_0 = (inter_0[0] - offset_x, inter_0[1] - offset_y)
                                        bridge_1 = (inter_1[0] + offset_x, inter_1[1] + offset_y)
                                        if (self.shapely_poly.contains(Point(bridge_0[0], bridge_0[1])) and
                                                self.shapely_poly.contains(Point(bridge_1[0], bridge_1[1]))):
                                            new_nodes.append(bridge_0)
                                            new_nodes.append(bridge_1)

            for node in new_nodes:
                self.graph[node] = []
        print("# nodes: " + str(len(self.graph.keys())))
        if DEBUG_MSG:
            print("time for additional_nodes:", time.time() - since)

    def construct_edges(self, curr_loc, target, only_construct_from_source=False):
        """Function which creates edges for every node with each other under the following conditions:
            - distance between two nodes < skill_dist_range
            - if the node is <20m from target, there cannot be water in the way.

        Args:
            curr_loc (sympy.geometry.Point2D): Current location
            target (sympy.geometry.Point2D): Target location
        """

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
        skill_dist_range = 200 + self.skill
        epsilon = 0.01

        for from_node in self.graph.keys():

            # constructing an Edge from curr_loc to another non-curr_loc Node
            if from_node == 'curr_loc':
                # clear existing adjacency list of this from_node
                self.graph[from_node] = []

                for to_node in self.graph.keys():
                    if to_node == from_node:  # 'curr_loc' can't have an Edge with itself
                        continue
                    """
                    can_reach_one_unit = False  # whether at least one Unit in to_node is reachable from from_node
                    for unit_center in to_node:
                        if self._euc_dist((int(curr_loc.x), int(curr_loc.y)), unit_center) > skill_dist_range:
                            continue
                        can_reach_one_unit = True
                        break

                    # if Edge (from_node, to_node) is valid, add to from_node's adjacency list
                    if can_reach_one_unit:
                        self.graph[from_node].append(to_node)
                    """
                    to_node_center = self.all_nodes_center[to_node]
                    if self._euc_dist((int(curr_loc.x), int(curr_loc.y)), to_node_center) <= skill_dist_range + epsilon:
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

                    if to_node == (float(target.x), float(target.y)):
                        if self._euc_dist(from_node, to_node) <= 20:
                            line = LineString([from_node, to_node])
                            # i. If yes, calculate bank_distance for this edge
                            if self.shapely_edges.intersects(line):
                                continue
                            else:
                                self.graph[from_node].append(to_node)
                    # if the distance between the two Node centers is reachable, add to from_node's adjacency list
                    if self._euc_dist(from_node_center, to_node_center) <= skill_dist_range:
                        self.graph[from_node].append(to_node)

        if DEBUG_MSG:
            print("time for construct_edges:", time.time() - since)

    @staticmethod
    def _euc_dist(pt1, pt2):
        pt1 = np.array((float(pt1[0]), float(pt1[1])))
        pt2 = np.array((float(pt2[0]), float(pt2[1])))
        return np.linalg.norm(pt1 - pt2)

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

            if 0 < visited[node] <= len(path):
                continue

            visited[node] = len(path)

            adj = self.graph[node]
            for a in adj:
                new_path = list(path)
                new_path.append(a)
                queue.append(new_path)
        if len(final_path) < 2:
            if DEBUG_MSG:
                print("time for bfs:", time.time() - since)
            return "default"

        move = final_path[1]

        if DEBUG_MSG:
            print("time for bfs:", time.time() - since)
            print("final_path:", [move for move in final_path[1:]])
        return sympy.geometry.Point2D(move[0], move[1])

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

        if DEBUG_MSG:
            print("curr_loc", float(curr_loc.x), float(curr_loc.y))

        required_dist = curr_loc.distance(target)
        """ 
        Case 1: required_dist > 20m
            - create a map with 1m-sized nodes

        Case 2: requier_dist < 20m (putting dist)
            - more specific map with .05m-sized nodes 20m around from current point
         """

        if score == 1:  # turn 1
            self.shapely_poly = Polygon([(p.x, p.y) for p in golf_map.vertices])
            self.shapely_edges = LineString(list(self.shapely_poly.exterior.coords))
            self.scikit_poly = sg.Polygon([(p.x, p.y) for p in golf_map.vertices])
            skel = sg.skeleton.create_interior_straight_skeleton(self.scikit_poly)
            self.draw_skeleton(self.scikit_poly, skel)
            self.construct_nodes(target)

            draw(self.scikit_poly)

            for v in self.graph.keys():
                plt.plot(v[0], v[1], 'bo')
            plt.savefig('test.png')

            self.construct_more_nodes(curr_loc)

            if self.needs_edge_init:
                self.construct_edges(curr_loc, target, only_construct_from_source=False)
                self.needs_edge_init = False

        else:
            self.construct_edges(curr_loc, target,
                                 only_construct_from_source=True)  # only construct outgoing edges of curr_loc

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
