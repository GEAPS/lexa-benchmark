from gym.spaces import Discrete, Box
import numpy as np
from envs.customfetch.custom_fetch import StackEnv, PickPlaceEnv, GoalType

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def discrete_to_box_wrapper(env, bound=4.):
  """takes a discrete environment, and turns it into a box environment"""
  assert isinstance(env.action_space, Discrete), "must pass a discrete environment!"
  old_step = env.step
  n = env.action_space.n
  env.action_space = Box(low = -bound, high = bound, shape=(n,))

  def step(action):
    action = np.clip(action, -bound, bound)
    action = softmax(action)
    action = np.random.choice(range(n), p=action)
    
    return old_step(action)

  env.step = step

  return env

class Fpp(PickPlaceEnv):
  def __init__(self, env_max_steps, test=False):
    super().__init__(max_step=env_max_steps, internal_goal = GoalType.OBJ,
     external_goal = GoalType.OBJ, mode=0, compute_reward_with_internal=test,
     per_dim_threshold=0.0, hard=True, distance_threshold=0.0, n=1,
     range_min=0.2, range_max=0.45)


class Fsk(StackEnv):
  def __init__(self, env_max_steps, test=False):
    super().__init__(max_step=env_max_steps, internal_goal = GoalType.OBJ,
     external_goal = GoalType.OBJ, mode=0, compute_reward_with_internal=test,
     per_dim_threshold=0.0, hard=False, distance_threshold=0.0, n=2,
     range_min=None, range_max=None)