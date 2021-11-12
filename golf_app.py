import logging
import os
import numpy as np
import sympy
import constants

from remi import App, gui


class GolfApp(App):
    def __init__(self, *args):
        res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res')
        super(GolfApp, self).__init__(*args, static_file_path={'res': res_path})

    def draw_polygon(self, poly):
        p = gui.SvgPolygon(len(poly.vertices))
        for v in poly.vertices:
            p.add_coord(float(v.x), float(v.y))
        return p

    def draw_point(self, point):
        return gui.SvgCircle(float(point.x), float(point.y), 1.0)

    def draw_line(self, line):
        return gui.SvgLine(float(line.points[0].x), float(line.points[0].y), float(line.points[1].x), float(line.points[1].y))

    def draw_circle(self, circle):
        return gui.SvgCircle(float(circle.center.x), float(circle.center.y), float(circle.radius))

    def main(self, *userdata):
        self.golf_game, start_automatic = userdata
        self.golf_game.set_app(self)

        mainContainer = gui.Container(style={'width': '100%', 'height': '100%', 'overflow': 'auto', 'text-align': 'center'})
        mainContainer.style['justify-content'] = 'center'
        mainContainer.style['align-items'] = 'center'

        bt_hbox = gui.HBox()
        bt_hbox.attributes["class"] = "tophbox"
        play_step_bt = gui.Button("Play Step")
        play_turn_bt = gui.Button("Play Turn")
        play_all_bt = gui.Button("Play All")

        self.automatic_play = gui.CheckBoxLabel("Play Automatically", checked=start_automatic)
        self.automatic_play.attributes["class"] = "checkbox"

        self.view_drop_down = gui.DropDown(style={'padding': '5px', 'text-align': 'center', })
        for player_idx, player_name in enumerate(self.golf_game.player_names):
            self.view_drop_down.append(player_name, player_idx)

        self.view_drop_down.select_by_value(self.golf_game.get_current_player())
        self.view_drop_down.onchange.do(self.view_drop_down_changed)

        bt_hbox.append([play_step_bt, play_turn_bt, play_all_bt, self.automatic_play, self.view_drop_down])

        play_step_bt.onclick.do(self.play_step_bt_press)
        play_turn_bt.onclick.do(self.play_turn_bt_press)
        play_all_bt.onclick.do(self.play_all_bt_press)
        mainContainer.append(bt_hbox)
        self.labels = []
        self.labels.append(gui.Label("Polygolf: Ready to start", style={'margin': '5px auto'}))
        if self.golf_game.next_player is not None:
            self.golf_game.logger.debug("First turn {}".format(self.golf_game.get_current_player()))
            self.set_label_text("{}, First turn {}".format(self.get_label_text(), self.golf_game.get_current_player()))
        for label in self.labels:
            mainContainer.append(label)

        self.score_table = gui.TableWidget(2, len(self.golf_game.players), style={'margin': '5px auto'})
        self.update_score_table()

        for player_idx, _ in enumerate(self.golf_game.scores):
            self.score_table.item_at(0, player_idx).set_style("padding:0 10px")
            self.score_table.item_at(1, player_idx).set_style("padding:0 10px")
        mainContainer.append(self.score_table)

        self.svgplot = gui.Svg(width="80%", height="80vh", style={'background-color': '#BBDDFF', 'margin': '0 auto'})
        # screen_width = 1000
        # screen_height = 600
        # svgplot.set_viewbox(0, 0, screen_width, screen_height)

        self.reset_svgplot()

        mainContainer.append(self.svgplot)
        return mainContainer

    def idle(self):
        if not self.golf_game.is_game_ended():
            if self.automatic_play.get_value():
                self.golf_game.play(run_stepwise=True)
        else:
            self.automatic_play.set_value(False)

    def reset_svgplot(self):
        self.svgplot.empty()

        bounds = self.golf_game.golf.golf_map.bounds
        xmin, ymin, xmax, ymax = list(map(float, bounds))
        p = self.draw_polygon(self.golf_game.golf.golf_map)
        # p.set_stroke(1, "black")
        p.set_fill("#BBFF66")
        self.svgplot.append(p)

        golf_start = self.golf_game.golf.start
        golf_target = self.golf_game.golf.target

        unit_t = sympy.geometry.Triangle(asa=(60, 10, 60))
        golf_start_triangle = sympy.geometry.Triangle(*list(map(lambda p: p.translate(golf_start.x-unit_t.circumcenter.x, golf_start.y-unit_t.circumcenter.y), unit_t.vertices)))
        t = self.draw_polygon(golf_start_triangle)
        t.set_fill("red")
        self.svgplot.append(t)

        golf_target_circle = sympy.geometry.Circle(golf_target, constants.target_radius)
        c = self.draw_circle(golf_target_circle)
        c.set_fill("rgba(0,0,0,0.3)")
        self.svgplot.append(c)

        ps = self.draw_point(golf_start)
        self.svgplot.append(ps)

        pe = self.draw_point(golf_target)
        self.svgplot.append(pe)

    def view_drop_down_changed(self, widget, value):
        self.reset_svgplot()
        player_idx = widget.get_key()

    def play_step_bt_press(self, widget):
        self.golf_game.play(run_stepwise=True)

    def play_turn_bt_press(self, widget):
        self.golf_game.play(run_stepwise=False)

    def play_all_bt_press(self, widget):
        self.golf_game.play_all()

    def plot(self, segment_air, segment_land, admissible):
        if admissible:
            if isinstance(segment_air, sympy.geometry.Segment2D):
                sa = self.draw_line(segment_air)
                sa.set_stroke(3, "rgba(0,0,0, 0.1)")
                self.svgplot.append(sa)
            if isinstance(segment_land, sympy.geometry.Segment2D):
                sl = self.draw_line(segment_land)
                sl.set_stroke(3, "rgba(0,0,0, 1)")
                self.svgplot.append(sl)
        else:
            if isinstance(segment_air, sympy.geometry.Segment2D):
                sa = self.draw_line(segment_air)
                sa.set_stroke(3, "rgba(255,0,0, 0.2)")
                self.svgplot.append(sa)
            if isinstance(segment_land, sympy.geometry.Segment2D):
                sl = self.draw_line(segment_land)
                sl.set_stroke(3, "rgba(255,0,0, 1)")
                self.svgplot.append(sl)

    def update_score_table(self):
        for player_idx, score in enumerate(self.golf_game.scores):
            self.score_table.item_at(0, player_idx).set_text("{}, {}".format(self.golf_game.player_names[player_idx], self.golf_game.skills[player_idx]))
            self.score_table.item_at(1, player_idx).set_text("{}, {}".format(score, self.golf_game.player_states[player_idx]))

    def set_label_text(self, text, label_num=0):
        self.labels[label_num].set_text(text)

    def get_label_text(self, label_num=0):
        return self.labels[label_num].get_text()
