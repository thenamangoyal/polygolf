import logging
import os
import numpy as np
import sympy
import logging
import json
import constants

class GolfMap:
    def __init__(self, map_filepath, logger) -> None:
        self.logger = logger

        self.logger.info("Map file loaded: {}".format(map_filepath))
        with open(map_filepath, "r") as f:
            json_obj = json.load(f)
        self.start = sympy.geometry.Point2D(*json_obj["start"])
        self.target = sympy.geometry.Point2D(*json_obj["target"])
        self.golf_map = sympy.Polygon(*json_obj["map"])
        assert self.golf_map.encloses(self.start), "Start point doesn't lie inside map polygon"
        assert self.golf_map.encloses(self.target), "Target point doesn't lie inside map polygon"