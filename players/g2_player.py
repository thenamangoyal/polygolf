import numpy as np
import sympy
import logging
from typing import Tuple, Iterator, List
from sympy.geometry import Polygon, Point2D
from scipy.stats import norm
from matplotlib.path import Path


def splash_zone(distance: float, angle: float, conf: float, skill: int, current_point: Point2D) -> List[Tuple[float, float]]:
    curr_x, curr_y = current_point.x, current_point.y
    conf_points = np.linspace(1 - conf, conf, 100)
    d_dist = norm(distance, distance/skill)
    a_dist = norm(angle, 1/(2*skill))
    distances = np.vectorize(d_dist.ppf)(conf_points)
    angles = np.vectorize(a_dist.ppf)(conf_points)
    xs = []
    ys = []
    scale = 1.1
    if distance <= 20:
        scale = 1.0
    max_distance = distances[-1]*scale
    for a in angles:
      x = curr_x + max_distance * np.cos(a)
      y = curr_y + max_distance * np.sin(a)

      xs.append(x)
      ys.append(y)

    min_distance = distances[0]
    for a in reversed(angles):
      x = curr_x + min_distance * np.cos(a)
      y = curr_y + min_distance * np.sin(a)

      xs.append(x)
      ys.append(y)

    # return Polygon(*zip(xs,ys), evaluate=False)
    return list(zip(xs, ys))


def poly_to_points(poly: Polygon) -> Iterator[Tuple[float, float]]:
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = float('-inf'), float('-inf')
    for point in poly.vertices:
        x_min = min(point.x, x_min)
        x_max = max(point.x, x_max)
        y_min = min(point.y, y_min)
        y_max = max(point.y, y_max)
    x_step = 1  # meter
    y_step = 1  # meter

    x_current = x_min + x_step
    y_current = y_min + y_step
    while x_current < x_max:
        while y_current < y_max:
            yield x_current, y_current
            y_current += y_step
        y_current = y_min
        x_current += x_step


def point_within_polygon(target_point: Point2D, poly: Polygon) -> bool:
    return poly.encloses_point(target_point)



# fn map to points
# fn points to points in range (with confidence n)
# fn points where splash zone is enclosed by polygon

# ^^ perform A* search over these points



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

    def reachable_point(self, current_point: Point2D, target_point: Point2D, conf: float) -> bool:
        """Determine whether the point is reachable with confidence [conf] based on our player's skill"""
        max_dist = 200 + self.skill
        d_dist = norm(max_dist, max_dist / self.skill)
        return current_point.distance(target_point) <= d_dist.ppf(1.0 - conf)
    
    def splash_zone_within_polygon(self, current_point: Point2D, target_point: Point2D, poly: Polygon, conf: float) -> bool:
        distance = current_point.distance(target_point)
        angle = sympy.atan2(target_point.y - current_point.y, target_point.x - current_point.x)
        splash_zone_polygon = splash_zone(float(distance), float(angle), float(conf), self.skill, current_point)
        vertices = poly.vertices
        vertices.append(vertices[0])
        path_poly = Path(vertices, closed=True)
        return path_poly.contains_points(splash_zone_polygon).all()
        # return poly.encloses(splash_zone_polygon)

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
        roll_factor = 1.1
        if required_dist < 20:
            roll_factor  = 1.0
        distance = sympy.Min(200+self.skill, required_dist/roll_factor)
        angle = sympy.atan2(target.y - curr_loc.y, target.x - curr_loc.x)

        return (distance, angle)




# === Unit Tests ===

def test_reachable():
    current_point = Point2D(0, 0, evaluate=False)
    target_point = Point2D(0, 250, evaluate=False)
    player = Player(50, 0xdeadbeef, None)
    
    assert not player.reachable_point(current_point, target_point, 0.80)


def test_point_within_polygon():
    poly = Polygon((0,0), (0, 300), (300, 300), (300, 0), evaluate=False)

    # Just checking points inside and outside
    inside = Point2D(0.1, 1, evaluate=False)
    outside = Point2D(301, 301, evaluate=False)

    assert point_within_polygon(inside, poly)
    assert not point_within_polygon(outside, poly)


def test_splash_zone_within_polygon():
    poly = Polygon((0,0), (0, 300), (300, 300), (300, 0), evaluate=False)

    current_point = Point2D(0, 0, evaluate=False)

    # Just checking polygons inside and outside
    inside_target_point = Point2D(150, 150, evaluate=False)
    outside_target_point = Point2D(299, 100, evaluate=False)

    player = Player(50, 0xdeadbeef, None)
    assert player.splash_zone_within_polygon(current_point, inside_target_point, poly, 0.8)
    assert not player.splash_zone_within_polygon(current_point, outside_target_point, poly, 0.8)


def test_poly_to_points():
    poly = Polygon((0,0), (0, 10), (10, 10), (10, 0))
    points = set(poly_to_points(poly))
    for x in range(1, 10):
        for y in range(1, 10):
            assert (x,y) in points
    assert len(points) == 81
