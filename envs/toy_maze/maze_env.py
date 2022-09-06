# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import numpy as np
from envs.toy_maze.mazes import mazes_dict, make_crazy_maze, make_experiment_maze, make_hallway_maze, make_u_maze


class Env:
    def __init__(self, n=None, maze_type=None, use_antigoal=True, ddiff=False, ignore_reset_start=False):
        self.n = n

        self._mazes = mazes_dict
        self.maze_type = maze_type.lower()

        self._ignore_reset_start = bool(ignore_reset_start)

        # Generate a crazy maze specified by its size and generation seed
        if self.maze_type.startswith('crazy'):
            _, size, seed = self.maze_type.split('_')
            size = int(size)
            seed = int(seed)
            self._mazes[self.maze_type] = {'maze': make_crazy_maze(size, seed), 'action_range': 0.95}

        # Generate an "experiment" maze specified by its height, half-width, and size of starting section
        if self.maze_type.startswith('experiment'):
            _, h, half_w, sz0 = self.maze_type.split('_')
            h = int(h)
            half_w = int(half_w)
            sz0 = int(sz0)
            self._mazes[self.maze_type] = {'maze': make_experiment_maze(h, half_w, sz0), 'action_range': 0.25}


        if self.maze_type.startswith('corridor'):
            corridor_length = int(self.maze_type.split('_')[1])
            self._mazes[self.maze_type] = {'maze': make_hallway_maze(corridor_length), 'action_range': 0.95}

        if self.maze_type.startswith('umaze'):
            corridor_length = int(self.maze_type.split('_')[1])
            self._mazes[self.maze_type] = {'maze': make_u_maze(corridor_length), 'action_range': 0.95}

        assert self.maze_type in self._mazes

        self.use_antigoal = bool(use_antigoal)
        self.ddiff = bool(ddiff)

        self._state = dict(s0=None, prev_state=None, state=None, goal=None, n=None, done=None, d_goal_0=None, d_antigoal_0=None)

        self.dist_threshold = 0.15

    @property
    def state_size(self):
        return 2

    @property
    def goal_size(self):
        return 2

    @property
    def action_size(self):
        return 2

    @property
    def maze(self):
        return self._mazes[self.maze_type]['maze']

    @property
    def action_range(self):
        return self._mazes[self.maze_type]['action_range']

    @property
    def state(self):
        return self._state['state'].view(-1).detach()

    @property
    def goal(self):
        return self._state['goal'].view(-1).detach()

    @property
    def antigoal(self):
        return self._state['antigoal'].view(-1).detach()

    @property
    def achieved(self):
        return self.goal if self.is_success else self.state

    @property
    def is_done(self):
        return bool(self._state['done'])

    @property
    def d_goal_0(self):
        return self._state['d_goal_0'].item()

    @property
    def d_antigoal_0(self):
        return self._state['d_antigoal_0'].item()

    @property
    def next_phase_reset(self):
        return {'state': self._state['s0'].detach(), 'goal': self.goal, 'antigoal': self.achieved}

    @property
    def sibling_reset(self):
        return {'state': self._state['s0'].detach(), 'goal': self.goal}
