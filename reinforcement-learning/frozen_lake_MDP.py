
# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import LEFT, RIGHT, DOWN, UP

import numpy as np
action_names = {LEFT: 'LEFT', RIGHT: 'RIGHT', DOWN: 'DOWN', UP: 'UP'}

def evaluate_policy(env, gamma, policy, value_func,max_iterations=int(1e3), tol=1e-3):
    """Evaluate the value of a policy.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    nS = env.nrow * env.ncol
    v = value_func

    #  goal is to update value function given policy
    # iterate max_iterations
      #  caculate value matrix
          #  foreach state and givien action            
   
    count = 0    
   
    for itr in range(max_iterations):
        delta = 0
        v_old = v.copy()
        count += 1
        for s in range(nS):                      
            a = policy[s]
            expectation = 0.0
            for prob, newstate, reward, terminated in env.P[s][a]:
                if terminated:
                    expectation += prob * (reward + 0.0)
                else:
                    expectation += prob * (reward + gamma* v_old[newstate])

            v[s] = expectation
            delta = max(delta, abs(v_old[s] - v[s]))
        
        if delta < tol:
            break
        
    return v, count



def value_function_to_policy(env, gamma, value_function):
    """Output action numbers for each state in value_function.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute policy for. Must have nS, nA, and P as
      attributes.
    gamma: float
      Discount factor. Number in range [0, 1)
    value_function: np.ndarray
      Value of each state.

    Returns
    -------
    np.ndarray
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """
    return np.zeros(env.nS, dtype='int')


def improve_policy(env, gamma, value_func, policy):
    """Given a policy and value function improve the policy.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

        Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.

    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """
    nS = env.nrow * env.ncol    
    is_stable = True        
    for s in range(nS):          
        max_value = float('-inf')
        old_action = policy[s]
        
        for a in action_names.keys():
          expectation = 0.0
          for prob, newstate, reward, terminated in env.P[s][a]:
              if terminated:
                  expectation += prob * (reward + 0.0)
              else:
                  expectation += prob * (reward + gamma* value_func[newstate])

          if expectation > max_value:
                policy[s] = a
                max_value = expectation                        

        if old_action != policy[s]:
            is_stable = False

    return is_stable, policy


def policy_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    You should use the improve_policy and evaluate_policy methods to
    implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    nS = env.nrow * env.ncol
    policy = np.zeros(nS, dtype='int')
    value_func = np.zeros(nS)    
    
    improve_iteration  = 0
    value_iterations = 0
    for itr in range(max_iterations):
        value_func, itr_c = evaluate_policy(env, gamma, policy, value_func, max_iterations, tol)
        is_stable, policy = improve_policy(env, gamma, value_func, policy)        
        value_iterations += itr_c
        improve_iteration += 1
        if is_stable:
            break        
        
    return policy, value_func,improve_iteration, value_iterations

def value_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.

    See page 90 (pg 108 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    value functions (numpy array), policy (numpy array), iteration (int)
    """
    nS = env.nrow * env.ncol
    v = np.zeros(nS, dtype='float')
    policy = np.zeros(nS, dtype='int')
    # init value matrix
    # iterate max_iterations
      #  caculate value matrix
          #  foreach state
            # caluclated expected value by taking each action based on probability
      # 
    count = 0    

    for itr in range(max_iterations):
        delta = 0
        v_old = v.copy()
        count += 1
        for s in range(nS):          
            max_value = float('-inf')
            for a in env.P[s].keys():              
                expectation = 0
                for prob, newstate, reward, terminated in env.P[s][a]:
                    if terminated:
                        expectation += prob * (reward + 0)
                    else:
                        expectation += prob * (reward + gamma* v_old[newstate])

                if expectation > max_value:
                      policy[s] = a
                      max_value = expectation
            v[s] = max_value
            delta = max(delta, abs(v_old[s] - max_value))
        
        if delta < tol:
            break
        
    return v, policy, count


def print_policy(policy, action_names):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """
    str_policy = policy.astype('str')
    for action_num, action_name in action_names.items():
        np.place(str_policy, policy == action_num, action_name)

    print(str_policy)

