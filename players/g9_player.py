import numpy as np
import sympy
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
        
    def get_landing_point(self, curr_loc: sympy.geometry.Point2D, distance: float, angle: float):
    	"""
    	    Args:
    	        curr_loc (sympy.geometry.Point2D): current location
    	        distance (float): The distance to next potential landing point
    	        angle (float): The angle coordinate to the next potential landing point
    	    Returns:
    	        the potential landing point as a sympy.Point2D object
    	"""
    	return sympy.Point2D(curr_loc.x + distance * sympy.cos(angle), curr_loc.y + distance * sympy.sin(angle))

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
        
        # The potential landing point
        landing_point = self.get_landing_point(curr_loc, distance, angle)
        delta_angle = 5 * sympy.pi / 180
        
        
        # if we're landing in water, then try different angles until we land on grass.
        if golf_map.encloses_point(landing_point) == False:
            potential_angle = delta_angle
            while(potential_angle <= sympy.pi):
                # Check in one direction
                if golf_map.encloses_point(self.get_landing_point(curr_loc, distance, angle + potential_angle)) == True:
                    angle = angle + potential_angle
                    break
                # Check in the other direction
                if golf_map.encloses_point(self.get_landing_point(curr_loc, distance, angle - potential_angle)) == True:
                    angle = angle - potential_angle
                    break
                    
                potential_angle += delta_angle
        
        #print("-------------------------------------------------------------------------------")
        #print(distance, angle)
        #print("-------------------------------------------------------------------------------")
        return (distance, angle)
