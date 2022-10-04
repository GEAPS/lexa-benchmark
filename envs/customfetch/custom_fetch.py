import numpy as np
from gym.envs.robotics.fetch_env import goal_distance
from gym.envs.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv

from gym.envs.robotics import fetch_env
import os
from gym.utils import EzPickle
from enum import Enum
from scipy.linalg import block_diag

from gym.envs.robotics import rotations, utils

dir_path = os.path.dirname(os.path.realpath(__file__))
STACKXML = os.path.join(dir_path, 'xmls', 'FetchStack#.xml')
ORIGPUSHXML = os.path.join(dir_path, 'xmls', 'Push.xml')
ORIGSLIDEXML = os.path.join(dir_path, 'xmls', 'Slide.xml')
PUSHXML = os.path.join(dir_path, 'xmls', 'CustomPush.xml')
PPXML = os.path.join(dir_path, 'xmls', 'CustomPP.xml')
SLIDEXML = os.path.join(dir_path, 'xmls', 'CustomSlide.xml')
SLIDE_N_XML = os.path.join(dir_path, 'xmls', 'FetchSlide#.xml')
PUSH_N_XML = os.path.join(dir_path, 'xmls', 'FetchPush#.xml')

INIT_Q_POSES = [
    [1.3, 0.6, 0.41, 1., 0., 0., 0.],
    [1.3, 0.9, 0.41, 1., 0., 0., 0.],
    [1.2, 0.68, 0.41, 1., 0., 0., 0.],
    [1.4, 0.82, 0.41, 1., 0., 0., 0.],
    [1.4, 0.68, 0.41, 1., 0., 0., 0.],
    [1.2, 0.82, 0.41, 1., 0., 0., 0.],
]
INIT_Q_POSES_SLIDE = [
    [1.3, 0.7, 0.42, 1., 0., 0., 0.],
    [1.3, 0.9, 0.42, 1., 0., 0., 0.],
    [1.25, 0.8, 0.42, 1., 0., 0., 0.],
    [1.35, 0.8, 0.42, 1., 0., 0., 0.],
    [1.35, 0.7, 0.42, 1., 0., 0., 0.],
    [1.25, 0.9, 0.42, 1., 0., 0., 0.],
]


class GoalType(Enum):
  OBJ = 1
  GRIP = 2
  OBJ_GRIP = 3
  ALL = 4
  OBJSPEED = 5
  OBJSPEED2 = 6
  OBJ_GRIP_GRIPPER = 7


def compute_reward(achieved_goal, goal, internal_goal, distance_threshold, per_dim_threshold,
                   compute_reward_with_internal, mode):
  # Always require internal success.
  internal_success = 0.
  if internal_goal == GoalType.OBJ_GRIP:
    if len(achieved_goal.shape) == 1:
      d = goal_distance(achieved_goal[:6], goal[:6])
    else:
      d = goal_distance(achieved_goal[:, :6], goal[:, :6])
  elif internal_goal in [GoalType.GRIP, GoalType.OBJ]:
    if len(achieved_goal.shape) == 1:
      d = goal_distance(achieved_goal[:3], goal[:3])
    else:
      d = goal_distance(achieved_goal[:, :3], goal[:, :3])
  else:
    raise

  internal_success = (d <= distance_threshold).astype(np.float32)

  if compute_reward_with_internal:
    return internal_success - (1. - mode)

  # use per_dim_thresholds for other dimensions
  success = np.all(np.abs(achieved_goal - goal) < per_dim_threshold, axis=-1)
  success = np.logical_and(internal_success, success).astype(np.float32)
  return success - (1. - mode)


def get_obs(sim, external_goal, goal, subtract_obj_velp=True):
  # positions
  grip_pos = sim.data.get_site_xpos('robot0:grip')
  dt = sim.nsubsteps * sim.model.opt.timestep
  grip_velp = sim.data.get_site_xvelp('robot0:grip') * dt
  robot_qpos, robot_qvel = utils.robot_get_obs(sim)

  object_pos = sim.data.get_site_xpos('object0').ravel()
  # rotations
  object_rot = rotations.mat2euler(sim.data.get_site_xmat('object0')).ravel()
  # velocities
  object_velp = (sim.data.get_site_xvelp('object0') * dt).ravel()
  object_velr = (sim.data.get_site_xvelr('object0') * dt).ravel()
  # gripper state
  object_rel_pos = object_pos - grip_pos
  if subtract_obj_velp:
    object_velp -= grip_velp

  gripper_state = robot_qpos[-2:]
  gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

  items = [
      grip_pos,
      object_pos,
      object_rel_pos,
      gripper_state,
      object_rot,
      object_velp,
      object_velr,
      grip_velp,
      gripper_vel,
  ]

  obs = np.concatenate(items)

  if external_goal == GoalType.ALL:
    achieved_goal = np.concatenate([
        object_pos,
        grip_pos,
        object_rel_pos,
        gripper_state,
        object_rot,
        object_velp,
        object_velr,
        grip_velp,
        gripper_vel,
    ])
  elif external_goal == GoalType.OBJ:
    achieved_goal = object_pos
  elif external_goal == GoalType.OBJ_GRIP:
    achieved_goal = np.concatenate([object_pos, grip_pos])
  elif external_goal == GoalType.OBJ_GRIP_GRIPPER:
    achieved_goal = np.concatenate([object_pos, grip_pos, gripper_state])
  elif external_goal == GoalType.OBJSPEED:
    achieved_goal = np.concatenate([object_pos, object_velp])
  elif external_goal == GoalType.OBJSPEED2:
    achieved_goal = np.concatenate([object_pos, object_velp, object_velr])
  else:
    raise ValueError('unsupported goal type!')
  return {
    'observation': obs, # not used
    'image': obs,
    'state': obs,
    'goal': goal.copy(),
    'image_goal': goal.copy(),
    'achieved_goal': achieved_goal,}
  


def sample_goal(initial_gripper_xpos, np_random, target_range, target_offset, height_offset, internal_goal,
                external_goal, grip_offset, gripper_goal):
  obj_goal = initial_gripper_xpos[:3] + np_random.uniform(-target_range, target_range, size=3)
  obj_goal += target_offset
  obj_goal[2] = height_offset

  if internal_goal in [GoalType.GRIP, GoalType.OBJ_GRIP]:
    grip_goal = initial_gripper_xpos[:3] + np_random.uniform(-0.15, 0.15, size=3) + np.array([0., 0., 0.15])
    obj_rel_goal = obj_goal - grip_goal
  else:
    grip_goal = obj_goal + grip_offset
    obj_rel_goal = -grip_offset

  if external_goal == GoalType.ALL:
    return np.concatenate([obj_goal, grip_goal, obj_rel_goal, gripper_goal, [0.] * 14])
  elif external_goal == GoalType.OBJ:
    return obj_goal
  elif external_goal == GoalType.OBJ_GRIP_GRIPPER:
    return np.concatenate([obj_goal, grip_goal, gripper_goal])
  elif external_goal == GoalType.OBJ_GRIP:
    return np.concatenate([obj_goal, grip_goal])
  elif external_goal == GoalType.OBJSPEED:
    return np.concatenate([obj_goal, [0.] * 3])
  elif external_goal == GoalType.OBJSPEED2:
    return np.concatenate([obj_goal, [0.] * 6])
  elif external_goal == GoalType.GRIP:
    raise NotImplementedError

  raise ValueError("BAD external goal value")


class StackEnv(fetch_env.FetchEnv, EzPickle):
  def __init__(self,
               max_step=50,
               n=1,
               internal_goal=GoalType.OBJ,
               external_goal=GoalType.OBJ,
               mode="-1/0",
               compute_reward_with_internal=False,
               per_dim_threshold=None,
               hard=False,
               distance_threshold=0.,
               range_min=None,
               range_max=None):
    self.internal_goal = internal_goal
    self.external_goal = external_goal
    self.n = n
    self.hard = hard

    initial_qpos = {
        'robot0:slide0': 0.405,
        'robot0:slide1': 0.48,
        'robot0:slide2': 0.0,
    }
    for i in range(self.n):
      k = 'object{}:joint'.format(i)
      initial_qpos[k] = INIT_Q_POSES[i]

    fetch_env.FetchEnv.__init__(self,
                                STACKXML.replace('#', '{}'.format(n)),
                                has_object=True,
                                block_gripper=False,
                                n_substeps=20,
                                gripper_extra_height=0.2,
                                target_in_the_air=False,
                                target_offset=0.0,
                                obj_range=0.15,
                                target_range=0.1,
                                distance_threshold=0.05,
                                initial_qpos=initial_qpos,
                                reward_type='sparse')

    EzPickle.__init__(self)

    self.max_step = max(50 * (n - 1), 50)
    self.num_step = 0

    self.mode = 0
    if mode == "0/1" or mode == 1:
      self.mode = 1

    if self.external_goal == internal_goal:
      self.compute_reward_with_internal = True
    else:
      self.compute_reward_with_internal = compute_reward_with_internal

    self.per_dim_threshold = np.sqrt(self.distance_threshold**2 / 3)
    if per_dim_threshold:
      raise NotImplementedError

  def compute_reward(self, achieved_goal, goal, info):
    # Compute distance between goal and the achieved goal.

    if self.external_goal == GoalType.OBJ_GRIP:
      if len(achieved_goal.shape) == 1:
        actual_internal_goals = np.split(goal[:-3], self.n)
        achieved_internal_goals = np.split(achieved_goal[:-3], self.n)
      else:
        actual_internal_goals = np.split(goal[:, :-3], self.n, axis=1)
        achieved_internal_goals = np.split(achieved_goal[:, :-3], self.n, axis=1)
    elif self.external_goal == GoalType.OBJ:
      if len(achieved_goal.shape) == 1:
        actual_internal_goals = np.split(goal, self.n)
        achieved_internal_goals = np.split(achieved_goal, self.n)
      else:
        actual_internal_goals = np.split(goal, self.n, axis=1)
        achieved_internal_goals = np.split(achieved_goal, self.n, axis=1)
    else:
      raise

    if len(achieved_goal.shape) == 1:
      success = 1.
    else:
      success = np.ones(achieved_goal.shape[0], dtype=np.float32)

    for b, g in zip(achieved_internal_goals, actual_internal_goals):
      d = goal_distance(b, g)
      success *= (d <= self.distance_threshold).astype(np.float32)

    if self.compute_reward_with_internal:
      return success - (1. - self.mode)

    # use per_dim_thresholds for other dimensions
    if len(achieved_goal.shape) == 1:
      d = goal_distance(achieved_goal[-3:], goal[-3:])
    else:
      d = goal_distance(achieved_goal[:, -3:], goal[:, -3:])
    success *= (d <= self.distance_threshold).astype(np.float32)

    return success - (1. - self.mode)

  def _get_obs(self):
    # positions
    grip_pos = self.sim.data.get_site_xpos('robot0:grip')
    dt = self.sim.nsubsteps * self.sim.model.opt.timestep
    grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
    robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

    obj_feats = []
    obj_poses = []

    for i in range(self.n):
      obj_labl = 'object{}'.format(i)
      object_pos = self.sim.data.get_site_xpos(obj_labl).ravel()
      object_pos[2] = max(object_pos[2], self.height_offset)
      # rotations
      object_rot = rotations.mat2euler(self.sim.data.get_site_xmat(obj_labl)).ravel()
      # velocities
      object_velp = (self.sim.data.get_site_xvelp(obj_labl) * dt).ravel()
      object_velr = (self.sim.data.get_site_xvelr(obj_labl) * dt).ravel()
      # gripper state
      object_rel_pos = object_pos - grip_pos
      #object_velp -= grip_velp

      obj_feats.append([object_pos, object_rel_pos, object_rot, object_velp, object_velr])
      obj_poses.append(object_pos)

    gripper_state = robot_qpos[-2:]
    gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

    if self.external_goal == GoalType.OBJ_GRIP:
      achieved_goal = np.concatenate(obj_poses + [grip_pos])
    else:
      achieved_goal = np.concatenate(obj_poses)
    obs = np.concatenate([grip_pos, gripper_state, grip_velp, gripper_vel] + sum(obj_feats, []))



    return {
        'observation': obs,
        'image': obs,
        'state': obs,
        'goal': self.goal.copy(),
        'image_goal': self.goal.copy(),
        'achieved_goal': achieved_goal,
    }

  def _render_callback(self):
    # Visualize target.
    sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos)
    if self.external_goal == GoalType.OBJ_GRIP:
      goals = np.split(self.goal[:-3], self.n)
    else:
      goals = np.split(self.goal, self.n)

    for i in range(self.n):
      site_id = self.sim.model.site_name2id('target{}'.format(i))
      self.sim.model.site_pos[site_id] = goals[i] - sites_offset[i]
    self.sim.forward()

  def _reset_sim(self):
    self.sim.set_state(self.initial_state)

    # Only a little randomize about the start state
    for i in range(self.n):
      object_qpos = self.sim.data.get_joint_qpos('object{}:joint'.format(i))
      object_qpos[:2] += self.np_random.uniform(-0.03, 0.03, size=2)
      self.sim.data.set_joint_qpos('object{}:joint'.format(i), object_qpos)

    bad_poses = [self.initial_gripper_xpos[:2]]
    # Randomize start positions of boxes.
    # for i in range(self.n):
    #   object_xpos = self.initial_gripper_xpos[:2]
    #   while min([np.linalg.norm(object_xpos - p) for p in bad_poses]) < 0.1:
    #       object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
    #   bad_poses.append(object_xpos)

    #   object_qpos = self.sim.data.get_joint_qpos('object{}:joint'.format(i))
    #   assert object_qpos.shape == (7,)
    #   object_qpos[:2] = object_xpos
    #   self.sim.data.set_joint_qpos('object{}:joint'.format(i), object_qpos)

    self.sim.forward()
    return True

  def _sample_goal(self):
    bottom_box = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
    bottom_box[2] = self.height_offset  #self.sim.data.get_joint_qpos('object0:joint')[:3]

    goal = []
    for i in range(self.n):
      goal.append(bottom_box + (np.array([0., 0., 0.05]) * i))

    if self.external_goal == GoalType.OBJ_GRIP:
      goal.append(goal[-1] + np.array([-0.01, 0., 0.008]))

    return np.concatenate(goal)

  def step(self, action):
    obs, reward, _, info = super().step(action)
    self.num_step += 1
    done = True if self.num_step >= self.max_step else False
    if done: info['TimeLimit.truncated'] = True

    if self.mode == 1 and reward:
      done = True

    info['is_success'] = np.allclose(reward, self.mode)
    return obs, reward, done, info

  def reset(self):
    obs = super().reset()
    self.num_step = 0
    return obs


class PickPlaceEnv(FetchPickAndPlaceEnv):
  def __init__(self,
               max_step=51,
               internal_goal=GoalType.OBJ,
               external_goal=GoalType.OBJ,
               mode="-1/0",
               compute_reward_with_internal=False,
               per_dim_threshold=None,
               hard=False,
               distance_threshold=0.,
               n=0.5,
               range_min=0.2,
               range_max=0.45):
    self.internal_goal = internal_goal
    self.external_goal = external_goal
    if hard:
      self.minimum_air = range_min
      self.maximum_air = range_max
    else:
      self.minimum_air = 0.
      self.maximum_air = range_max
    self.in_air_percentage = n
    super().__init__(reward_type='sparse')

    if distance_threshold > 1e-5:
      self.distance_threshold = distance_threshold

    self.max_step = max_step
    self.num_step = 0
    self.mode = 0
    if mode == "0/1" or mode == 1:
      self.mode = 1

    if self.external_goal == internal_goal:
      self.compute_reward_with_internal = True
    else:
      self.compute_reward_with_internal = compute_reward_with_internal

    self.per_dim_threshold = np.sqrt(self.distance_threshold**2 / 3)
    if per_dim_threshold:
      self.per_dim_threshold = per_dim_threshold
    print('PER DIM THRESHOLD:', self.per_dim_threshold)

  def compute_reward(self, achieved_goal, goal, info):
    return compute_reward(achieved_goal, goal, self.internal_goal, self.distance_threshold, self.per_dim_threshold,
                          self.compute_reward_with_internal, self.mode)

  def _render_callback(self):
    # Visualize target.
    sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
    site_id = self.sim.model.site_name2id('target0')

    goal = self.goal[:3]

    self.sim.model.site_pos[site_id] = goal - sites_offset[0]
    self.sim.forward()

  def _get_obs(self):
    return get_obs(self.sim, self.external_goal, self.goal)

  def _sample_goal(self):
    height_offset = self.height_offset
    if self.np_random.uniform() < self.in_air_percentage:
      height_offset += self.np_random.uniform(self.minimum_air, self.maximum_air)
    return sample_goal(self.initial_gripper_xpos, self.np_random, self.target_range, self.target_offset, height_offset,
                       self.internal_goal, self.external_goal, np.array([-0.01, 0., 0.008]), [0.024273, 0.024273])

  def step(self, action):
    obs, reward, _, info = super().step(action)
    self.num_step += 1
    done = True if self.num_step >= self.max_step else False
    if done: info['TimeLimit.truncated'] = True

    if self.mode == 1 and reward:
      done = True

    info['is_success'] = np.allclose(reward, self.mode)
    return obs, reward, done, info

  def reset(self):
    obs = super().reset()
    self.num_step = 0
    return obs
