from p5 import *
import json
import sympy
import numpy as np
import argparse
import os

args = None
FILE = None
map = []
golf_start = None
golf_target = None
f = None  # STEP 1 Declare PFont variable


def draw_polygon(poly):
    begin_shape()
    for v in poly.vertices:
        vertex(float(v.x), float(v.y))
    end_shape()


def draw_point(p):
    point(float(p.x), float(p.y))


def draw_line(l):
    line((float(l.points[0].x), float(l.points[0].y)), (float(l.points[1].x), float(l.points[1].y)))


def draw_circle(c):
    circle(float(c.center.x), float(c.center.y), float(c.radius))


def setup():
    global f, args
    size(args.width, args.height)
    f = create_font("Arial.ttf", 16)  # STEP 2 Create Font


def draw():
    global map, golf_start, golf_target, f, args
    background(102)
    text_font(f, 16)
    text_align("CENTER")
    fill(0)
    text("Press e to save and exit, s for start (red) and t for target (green)", (width/2, args.height*0.9))

    stroke(0)
    fill(255)

    if map:
        poly = sympy.geometry.Polygon(*map)
        if len(map) == 1:
            draw_point(poly)
        elif len(map) == 2:
            draw_line(poly)
        elif len(map) > 2:

            draw_polygon(poly)
    if golf_start:
        sc = sympy.geometry.Circle(golf_start, 10)
        fill(255, 0, 0)
        draw_circle(sc)
        fill(255)
    if golf_target:
        ec = sympy.geometry.Circle(golf_target, 10)
        fill(0, 255, 0)
        draw_circle(ec)
        fill(255)


def save():
    global map, golf_start, golf_target, FILE
    map_list = []
    for p in map:
        tup = (float(p.x), float(p.y))
        map_list.append(tup)

    save_dict = dict()
    save_dict["map"] = map_list
    save_dict["start"] = (float(golf_start.x), float(golf_start.y))
    save_dict["target"] = (float(golf_target.x), float(golf_target.y))

    with open(FILE, "w") as f:
        json.dump(save_dict, f)
    print("saved file {}".format(FILE))
    exit()


def mouse_pressed():
    global map, golf_start, golf_target
    p = sympy.geometry.Point2D(mouse_x, mouse_y)
    print(mouse_x, mouse_y)
    map.append(p)
    print("New Map", map)


def key_pressed():
    global map, golf_start, golf_target
    p = sympy.geometry.Point2D(mouse_x, mouse_y)
    # print(mouse_x, mouse_y)
    if key == "e":
        save()
    elif key == "s":
        golf_start = p
        print("Start assigned")
    elif key == "t":
        golf_target = p
        print("Target assigned")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", default="map.json", help="Path to export generated map")
    parser.add_argument("--width", help="Width", type=int, default=800)
    parser.add_argument("--height", help="Height", type=int, default=600)

    args = parser.parse_args()
    FILE = args.file

    golf_start = sympy.geometry.Point2D(args.width*0.2, args.height*0.2)
    golf_target = sympy.geometry.Point2D(args.width*0.8, args.height*0.8)

    dir = os.path.dirname(FILE)
    if dir:
        os.makedirs(dir, exist_ok=True)

    run(frame_rate=60)
