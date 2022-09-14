# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

from .maze_env import Env
import gym
import numpy as np


class PointMaze2D(gym.GoalEnv):
  """Wraps the Sibling Rivalry 2D point maze in a gym goal env.
  Keeps the first visit done and uses -1/0 rewards.
  """
  def __init__(self, env_max_steps, test=False):
    super().__init__()
    # n is the maximum steps - should be controlled.
    # during training, the tasks is marked as never done.
    # num_steps is not configurable.
    self._env = Env(n=env_max_steps, maze_type='square_large', use_antigoal=False, ddiff=False, ignore_reset_start=True)
    self.maze = self._env.maze
    self.dist_threshold = 0.15

    self.action_space = gym.spaces.Box(-0.95, 0.95, (2, ))
    observation_space = gym.spaces.Box(-np.inf, np.inf, (2, ))
    goal_space = gym.spaces.Box(-np.inf, np.inf, (2, ))
    self.observation_space = gym.spaces.Dict({
        'observation': observation_space,
        'desired_goal': goal_space,
        'achieved_goal': goal_space
    })

    self.s_xy = np.array(self.maze.sample_start())
    self.g_xy = np.array(self.maze.sample_goal(min_wall_dist=0.025 + self.dist_threshold))
    self.max_steps = env_max_steps # 50 most of the time.
    self.num_steps = 0
    self.test = test

  def seed(self, seed=None):
    self.action_space.seed(seed=seed)
    return self.maze.seed(seed=seed)

  def step(self, action):
    try:
      s_xy = np.array(self.maze.move(tuple(self.s_xy), tuple(action)))
    except:
      print('failed to move', tuple(self.s_xy), tuple(action))
      raise

    self.s_xy = s_xy
    reward = self.compute_reward(s_xy, self.g_xy, None)
    info = {}
    self.num_steps += 1

    if self.test:
      done = np.allclose(0., reward)
      info['is_success'] = done
    else:
      done = False
      info['is_success'] = np.allclose(0., reward)

    if self.num_steps >= self.max_steps and not done:
      done = True
      info['TimeLimit.truncated'] = True

    # obs = {
    #     'observation': s_xy,
    #     'achieved_goal': s_xy,
    #     'desired_goal': self.g_xy,
    # }

    obs = {
      'image': s_xy,
      'state': s_xy,
      'goal': self.g_xy,
      'image_goal': self.g_xy,
      'achieved_goal': s_xy
    }


    return obs, reward, done, info

  def reset(self):
    self.num_steps = 0
    s_xy = np.array(self.maze.sample_start())
    self.s_xy = s_xy
    g_xy = np.array(self.maze.sample_goal(min_wall_dist=0.025 + self.dist_threshold))
    self.g_xy = g_xy
    # return {
        # 'observation': s_xy,
        # 'achieved_goal': s_xy,
        # 'desired_goal': g_xy,
    # }
    obs = {
      'image': s_xy,
      'state': s_xy,
      'goal': self.g_xy,
      'image_goal': self.g_xy,
      'achieved_goal': s_xy
    }
    return obs


  def render(self):
    raise NotImplementedError

  def compute_reward(self, achieved_goal, desired_goal, info):
    d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
    return -(d >= self.dist_threshold).astype(np.float32)

