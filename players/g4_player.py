import time
import numpy as np
import sympy
import logging
from typing import Tuple
import constants
from shapely import geometry
import pdb


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
        # TODO: define the risk rate, which means we can only take risks at 10%
        self.risk = 0.1
        self.simulate_times = 10
        self.tolerant_times = self.simulate_times * self.risk
        self.remember_middle_points = []

        self.turn = 0

    def water_boolean(self, poly, grid_points):
        water_grid = []
        
        for i,row in enumerate(grid_points): 
            water_grid.append([])
            for j,point in enumerate(row):
                
                thebool = poly.contains(point)
                
                water_grid[i].append(thebool)
            
        print(np.array(water_grid))
        return np.array(water_grid)



    def make_grid(self, golf_map: sympy.Polygon,target: sympy.geometry.Point2D,
        curr_loc: sympy.geometry.Point2D, prev_loc: sympy.geometry.Point2D):

        poly=geometry.Polygon([p.x, p.y] for p in golf_map.vertices)
        target_shapely = geometry.Point(target[0], target[1])
        print("convert")

        (xmin, ymin, xmax, ymax) = golf_map.bounds
        list_of_lists = []
        list_of_distances=[]
        
        queue = []
        dimension = 10
        allowed_distance = 100
        threshold = 20.0
        amt=1
        grid_of_scores = np.array(np.ones((dimension,dimension))*100)
        
        
        xmin = float(xmin)
        ymin = float(ymin)
        xmax = float(xmax)
        ymax = float(ymax)
        
        print(xmin,xmax,ymin,ymax)
        
        xcoords,ycoords = np.meshgrid(np.linspace(xmin,xmax, dimension),np.linspace(ymin,ymax, dimension))

        for x_index in range(len(xcoords)):
            list_of_lists.append([])
            list_of_distances.append([])
            for y_index in range(len(ycoords)):
                #considered_point = sympy.geometry.Point2D(xcoords[y_index,x_index],ycoords[y_index,x_index])
                considered_point = geometry.Point(xcoords[y_index,x_index],ycoords[y_index,x_index])
                list_of_lists[x_index].append(considered_point)
                #thedistance = curr_loc.distance(considered_point)
                #list_of_distances[x_index].append(thedistance)
                #print(float(considered_point[0]),float(considered_point[1]), float(curr_loc[0]), float(curr_loc[1]))
        print("hi")
        self.test()
        self.real_bfs(poly, list_of_lists, target_shapely, allowed_distance, grid_of_scores)
        """print("test1")
        water_grid = self.water_boolean(poly, list_of_lists) # True if on LAND
        for x_index in range(len(list_of_lists)):
            for y_index in range(len(list_of_lists[0])): 
                thedistance = target_shapely.distance(list_of_lists[x_index][y_index])
                print(thedistance)
                print(allowed_distance)
                print(water_grid[x_index][y_index] and thedistance < allowed_distance)
                if (thedistance < allowed_distance) and water_grid[x_index][y_index]:
                    queue.append((x_index,y_index))
                    print(queue, "what")"""


        print(queue)
        print("HELL")
        print(water_grid)

        """BFS stuff"""
        if thedistance < threshold:
            grid_of_scores[x_index,y_index] = amt
        return "hi"
    def test(self):
        print("um")

    def real_bfs(self, poly, list_of_lists, target_shapely, allowed_distance, grid_of_scores):
        queue = []
        water_grid = self.water_boolean(poly, list_of_lists) # True if on LAND
        for x_index in range(len(list_of_lists)):
            for y_index in range(len(list_of_lists[0])): 
                thedistance = target_shapely.distance(list_of_lists[x_index][y_index])
                if (thedistance < allowed_distance) and water_grid[x_index][y_index]:
                    queue.append((x_index,y_index))
                    grid_of_scores[x_index][y_index] = 1

                    print(queue, "what")
        print(grid_of_scores)

        while(len(queue) != 0):
            
            elem = queue.pop()
            # get all points that are < distance from elem
            elem_score = grid_of_scores[elem[0]][elem[1]]
            points_to_consider = []
            for x_index in range(len(list_of_lists)):
                for y_index in range(len(list_of_lists[0])):
                    elem_point = list_of_lists[elem[0]][elem[1]]
                    distance = elem_point.distance(list_of_lists[x_index][y_index])
                    if distance < allowed_distance:
                        print("hi")
                        points_to_consider.append((x_index,y_index))
            for point in points_to_consider:
                if water_grid[point[0],point[1]]:
                    (x_index,y_index) = point
                    if elem_score + 1 < grid_of_scores[x_index][y_index]:

                        grid_of_scores[x_index][y_index] = elem_score + 1
                        queue.append(point)
            print(grid_of_scores)
            

                
            #bfs


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
        print("test")
        if self.turn == 0:
            a = self.make_grid(golf_map,target,curr_loc, prev_loc)

        self.turn += 1
        # 1. always try greedy first
        required_dist = curr_loc.distance(target)
        roll_factor = 1. + constants.extra_roll
        if required_dist < constants.min_putter_dist:
            roll_factor = 1.0
        distance = sympy.Min(constants.max_dist + self.skill, required_dist / roll_factor)
        angle = sympy.atan2(target.y - curr_loc.y, target.x - curr_loc.x)

        is_greedy = True
        failed_times = 0
        # simulate the actual situation to ensure fail times will not be larger than self.tolerant_times
        for _ in range(self.simulate_times):
            is_succ, final_point = self.simulate_once(distance, angle, curr_loc, golf_map)
            if not is_succ:
                failed_times += 1
                if failed_times > self.tolerant_times:
                    is_greedy = False
                    break

        if is_greedy:
            self.logger.info(str(self.turn) + "select greedy strategy to go")
            return (distance, angle)

        # 2. if we cannot use greedy, we try to find the points intersected with the golf map
        if prev_admissible is None or prev_admissible or not self.remember_middle_points:
            circle = sympy.Circle(curr_loc, distance)
            # TODO: cost about 6-8 seconds, too slow
            intersect_points_origin = circle.intersection(golf_map)

            intersect_points_num = len(intersect_points_origin)
            temp_middle_points = []

            for i in range(intersect_points_num):
                for j in range(i + 1, intersect_points_num):
                    middle_point = sympy.Point2D(float(intersect_points_origin[i].x + intersect_points_origin[j].x) / 2,
                                                 float(intersect_points_origin[i].y + intersect_points_origin[j].y) / 2)
                    # find points that in the golf map polygon
                    if golf_map.encloses(middle_point):
                        temp_middle_points.append(middle_point)

            if len(temp_middle_points) == 0:
                self.logger.error(str(self.turn) + "cannot find any middle point, BUG!!!")
                return (distance, angle)

            # if there are many ways to go, delete the points that can go back
            middle_points = []
            for i, middle_point in enumerate(temp_middle_points):
                if middle_point.distance(target) > required_dist:
                    continue
                middle_points.append(middle_point)

            # if there we delete every point in temp_middle_points,
            # which means we could go longer ways than expected, we need to add back those points
            if len(middle_points) == 0:
                middle_points = temp_middle_points

            self.remember_middle_points = middle_points
        else:
            middle_points = list(self.remember_middle_points)

        middle_points_num = len(middle_points)
        mid_to_target_distance = [0] * middle_points_num
        for i, middle_point in enumerate(middle_points):
            mid_to_target_distance[i] = middle_point.distance(target)

        distance_sorted_indexes = sorted(range(middle_points_num), key=lambda x: mid_to_target_distance[x])

        middle_failed_times = [0] * middle_points_num

        midd_index = -1
        for i in distance_sorted_indexes:
            middle_point = middle_points[i]
            angle = sympy.atan2(middle_point.y - curr_loc.y, middle_point.x - curr_loc.x)
            for _ in range(self.simulate_times):
                is_succ, final_point = self.simulate_once(distance, angle, curr_loc, golf_map)
                if not is_succ:
                    middle_failed_times[i] += 1
                    if middle_failed_times[i] > self.tolerant_times:
                        middle_failed_times[i] = -1
                        break

            if middle_failed_times[i] != -1:
                midd_index = i
                break

        desire_distance = distance
        if midd_index != -1:
            self.logger.info(str(self.turn) + "select largest distance to middle point to go")
            desire_angle = sympy.atan2(middle_points[midd_index].y - curr_loc.y,
                                       middle_points[midd_index].x - curr_loc.x)

            return (desire_distance, desire_angle)

        # 3. if middle points are still not safe, choose the closest one to the target
        closest_index = distance_sorted_indexes[0]
        closest_middle_point = middle_points[closest_index]

        curr_to_mid = closest_middle_point.distance(curr_loc)
        desire_distance = sympy.Min(constants.max_dist + self.skill, curr_to_mid / roll_factor)
        desire_angle = sympy.atan2(closest_middle_point.y - curr_loc.y, closest_middle_point.x - curr_loc.x)
        self.logger.info(str(self.turn) + "risky!!! select closest middle point to go")
        return (desire_distance, desire_angle)

    def simulate_once(self, distance, angle, curr_loc, golf_map):
        actual_distance = self.rng.normal(distance, distance / self.skill)
        actual_angle = self.rng.normal(angle, 1 / (2 * self.skill))

        # landing_point means the land point(the golf can skip for a little distance),
        # final_point means the final stopped point, it is not equal
        if distance < constants.min_putter_dist:
            landing_point = curr_loc
            final_point = sympy.Point2D(curr_loc.x + actual_distance * sympy.cos(actual_angle),
                                        curr_loc.y + actual_distance * sympy.sin(actual_angle))

        else:
            landing_point = sympy.Point2D(curr_loc.x + actual_distance * sympy.cos(actual_angle),
                                          curr_loc.y + actual_distance * sympy.sin(actual_angle))
            final_point = sympy.Point2D(
                curr_loc.x + (1. + constants.extra_roll) * actual_distance * sympy.cos(actual_angle),
                curr_loc.y + (1. + constants.extra_roll) * actual_distance * sympy.sin(actual_angle))

        segment_land = sympy.geometry.Segment2D(landing_point, final_point)
        return golf_map.encloses(segment_land), final_point

