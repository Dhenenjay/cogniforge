"""
Optimization utilities for waypoint and trajectory optimization.

This module provides optimization algorithms for finding optimal waypoints,
trajectories, and control parameters using various optimization methods including CMA-ES.
"""

import numpy as np
import cma
from typing import Callable, List, Tuple, Dict, Any, Optional, Union
import logging
import time
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.transform import Rotation
import json
from multiprocessing import Pool, Queue, Process, Manager, cpu_count
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
from queue import Empty
import signal
import sys

# Configure logging
logger = logging.getLogger(__name__)


def optimize_waypoints_cma(
    cost_fn: Callable[[np.ndarray], float],
    W0: Union[List[Tuple[float, float, float]], np.ndarray],
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    budget_iters: int = 40,
    popsize: int = 12,
    sigma0: float = 0.1,
    restarts: int = 0,
    parallel: bool = False,
    verbose: bool = True,
    callback: Optional[Callable] = None,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Optimize waypoints using CMA-ES to minimize cost function.
    
    CMA-ES is a powerful evolutionary optimization algorithm that adapts
    the covariance matrix of the search distribution, making it effective
    for non-convex optimization problems like waypoint optimization.
    
    Args:
        cost_fn: Cost function that takes flattened waypoint array and returns scalar cost
                 Should handle waypoints as flat array of shape (n_waypoints * 3,)
        W0: Initial waypoints as list of (x,y,z) tuples or numpy array
        bounds: Optional bounds as (lower_bounds, upper_bounds) arrays
        budget_iters: Maximum number of iterations (generations)
        popsize: Population size (number of candidate solutions per generation)
        sigma0: Initial standard deviation for sampling
        restarts: Number of restarts with increasing population size
        parallel: Whether to evaluate population in parallel
        verbose: Whether to print optimization progress
        callback: Optional callback function called after each iteration
        options: Additional CMA-ES options
        
    Returns:
        Dictionary containing:
        - 'optimal_waypoints': Optimized waypoints as list of tuples
        - 'optimal_waypoints_array': Optimized waypoints as numpy array
        - 'optimal_cost': Final cost value
        - 'initial_cost': Initial cost value
        - 'convergence_history': Cost values over iterations
        - 'solution': CMA-ES solution object
        - 'n_evaluations': Total number of function evaluations
        - 'optimization_time': Total optimization time in seconds
        
    Example:
        def trajectory_cost(W_flat):
            W = W_flat.reshape(-1, 3)
            length = np.sum(np.linalg.norm(np.diff(W, axis=0), axis=1))
            smoothness = np.sum(np.linalg.norm(np.diff(W, n=2, axis=0), axis=1))
            return length + 0.1 * smoothness
        
        W0 = [(0, 0, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0.3)]
        result = optimize_waypoints_cma(
            trajectory_cost, W0,
            bounds=(np.array([-1, -1, 0]*3), np.array([1, 1, 1]*3)),
            budget_iters=50,
            popsize=16
        )
        
        optimal_waypoints = result['optimal_waypoints']
        print(f"Optimized cost: {result['optimal_cost']:.3f}")
    """
    start_time = time.time()
    
    # Convert initial waypoints to numpy array
    if isinstance(W0, list):
        W0_array = np.array([list(w) for w in W0])
    else:
        W0_array = np.array(W0)
    
    n_waypoints = len(W0_array)
    n_dims = W0_array.size  # Total dimensions (n_waypoints * 3)
    
    # Flatten initial waypoints for optimization
    x0 = W0_array.flatten()
    
    # Evaluate initial cost
    initial_cost = cost_fn(x0)
    
    if verbose:
        logger.info(f"Starting CMA-ES optimization")
        logger.info(f"  Waypoints: {n_waypoints} ({n_dims} dimensions)")
        logger.info(f"  Initial cost: {initial_cost:.6f}")
        logger.info(f"  Budget: {budget_iters} iterations, population size: {popsize}")
    
    # Setup CMA-ES options
    cma_options = {
        'maxiter': budget_iters,
        'popsize': popsize,
        'verb_disp': 1 if verbose else 0,
        'verb_log': 0,
        'tolfun': 1e-11,
        'tolx': 1e-11,
    }
    
    # Add bounds if provided
    if bounds is not None:
        lower_bounds, upper_bounds = bounds
        if isinstance(lower_bounds, (list, tuple)):
            lower_bounds = np.array(lower_bounds)
        if isinstance(upper_bounds, (list, tuple)):
            upper_bounds = np.array(upper_bounds)
        
        # Ensure bounds have correct shape
        if len(lower_bounds) != n_dims:
            if len(lower_bounds) == 3:  # Per-dimension bounds
                lower_bounds = np.tile(lower_bounds, n_waypoints)
            else:
                raise ValueError(f"Lower bounds shape mismatch: got {len(lower_bounds)}, expected {n_dims}")
        
        if len(upper_bounds) != n_dims:
            if len(upper_bounds) == 3:  # Per-dimension bounds
                upper_bounds = np.tile(upper_bounds, n_waypoints)
            else:
                raise ValueError(f"Upper bounds shape mismatch: got {len(upper_bounds)}, expected {n_dims}")
        
        cma_options['bounds'] = [lower_bounds.tolist(), upper_bounds.tolist()]
        
        # Clip initial point to bounds
        x0 = np.clip(x0, lower_bounds, upper_bounds)
    
    # Add any user-provided options
    if options:
        cma_options.update(options)
    
    # Setup parallel evaluation if requested
    if parallel:
        cma_options['EpochEval'] = True  # Evaluate whole population at once
    
    # Track convergence history
    convergence_history = []
    best_costs = []
    all_solutions = []
    n_evaluations = 0
    
    # Define wrapped cost function with tracking
    def wrapped_cost_fn(x):
        nonlocal n_evaluations
        n_evaluations += 1
        try:
            cost = cost_fn(x)
            if not np.isfinite(cost):
                cost = 1e10  # Penalty for invalid solutions
            return cost
        except Exception as e:
            logger.warning(f"Cost evaluation failed: {e}")
            return 1e10
    
    # Run CMA-ES optimization with optional restarts
    best_solution = None
    best_cost = float('inf')
    
    for restart in range(restarts + 1):
        if restart > 0:
            if verbose:
                logger.info(f"Restart {restart}/{restarts}")
            # Increase population size for restarts
            cma_options['popsize'] = popsize * (2 ** restart)
            cma_options['maxiter'] = budget_iters // (restart + 1)
        
        # Create CMA-ES instance
        es = cma.CMAEvolutionStrategy(x0, sigma0, cma_options)
        
        # Optimization loop
        iteration = 0
        while not es.stop() and iteration < cma_options['maxiter']:
            # Get population
            if parallel:
                # Get all solutions at once for parallel evaluation
                solutions = es.ask()
                # Evaluate in parallel (user should handle parallelization in cost_fn)
                costs = [wrapped_cost_fn(x) for x in solutions]
            else:
                # Sequential evaluation
                solutions = es.ask()
                costs = [wrapped_cost_fn(x) for x in solutions]
            
            # Update CMA-ES
            es.tell(solutions, costs)
            
            # Track best solution
            min_idx = np.argmin(costs)
            if costs[min_idx] < best_cost:
                best_cost = costs[min_idx]
                best_solution = solutions[min_idx].copy()
            
            # Record convergence history
            convergence_history.append(costs)
            best_costs.append(best_cost)
            
            # Callback
            if callback is not None:
                callback_info = {
                    'iteration': iteration,
                    'best_cost': best_cost,
                    'mean_cost': np.mean(costs),
                    'std_cost': np.std(costs),
                    'best_solution': best_solution,
                    'cma_sigma': es.sigma
                }
                callback(callback_info)
            
            # Logging
            if verbose and (iteration % 10 == 0 or iteration == 0):
                logger.info(f"  Iter {iteration:3d}: Best={best_cost:.6f}, "
                           f"Mean={np.mean(costs):.6f}, Std={np.std(costs):.6f}, "
                           f"Sigma={es.sigma:.4f}")
            
            iteration += 1
        
        # Store solution
        all_solutions.append({
            'solution': es.result.xbest,
            'cost': es.result.fbest,
            'iterations': iteration,
            'restart': restart
        })
        
        # Update best if needed
        if es.result.fbest < best_cost:
            best_cost = es.result.fbest
            best_solution = es.result.xbest.copy()
    
    # Reshape optimal solution to waypoints
    optimal_waypoints_array = best_solution.reshape(n_waypoints, 3)
    optimal_waypoints = [tuple(w) for w in optimal_waypoints_array]
    
    # Compute improvement
    improvement = initial_cost - best_cost
    improvement_pct = (improvement / initial_cost) * 100 if initial_cost != 0 else 0
    
    optimization_time = time.time() - start_time
    
    if verbose:
        logger.info(f"Optimization complete:")
        logger.info(f"  Initial cost: {initial_cost:.6f}")
        logger.info(f"  Optimal cost: {best_cost:.6f}")
        logger.info(f"  Improvement: {improvement:.6f} ({improvement_pct:.1f}%)")
        logger.info(f"  Function evaluations: {n_evaluations}")
        logger.info(f"  Time: {optimization_time:.2f} seconds")
    
    return {
        'optimal_waypoints': optimal_waypoints,
        'optimal_waypoints_array': optimal_waypoints_array,
        'optimal_cost': best_cost,
        'initial_cost': initial_cost,
        'improvement': improvement,
        'improvement_percent': improvement_pct,
        'convergence_history': convergence_history,
        'best_costs': best_costs,
        'all_solutions': all_solutions,
        'n_evaluations': n_evaluations,
        'optimization_time': optimization_time,
        'final_sigma': es.sigma if 'es' in locals() else sigma0
    }


def create_smooth_cost_function(
    length_weight: float = 1.0,
    smoothness_weight: float = 0.5,
    collision_weight: float = 10.0,
    clearance_weight: float = 0.1,
    obstacles: Optional[List[Dict[str, Any]]] = None,
    min_clearance: float = 0.05,
    start_point: Optional[np.ndarray] = None,
    goal_point: Optional[np.ndarray] = None,
    goal_weight: float = 5.0
) -> Callable[[np.ndarray], float]:
    """
    Create a cost function for smooth waypoint optimization.
    
    Args:
        length_weight: Weight for path length
        smoothness_weight: Weight for path smoothness (acceleration)
        collision_weight: Weight for collision penalty
        clearance_weight: Weight for obstacle clearance
        obstacles: List of obstacles (dicts with 'center' and 'radius')
        min_clearance: Minimum required clearance from obstacles
        start_point: Optional fixed start point
        goal_point: Optional fixed goal point
        goal_weight: Weight for reaching goal
        
    Returns:
        Cost function for waypoint optimization
        
    Example:
        obstacles = [
            {'center': [0.3, 0.3, 0.5], 'radius': 0.1},
            {'center': [0.7, 0.7, 0.3], 'radius': 0.15}
        ]
        
        cost_fn = create_smooth_cost_function(
            obstacles=obstacles,
            goal_point=np.array([1.0, 1.0, 0.5])
        )
        
        result = optimize_waypoints_cma(cost_fn, initial_waypoints)
    """
    def cost_function(W_flat: np.ndarray) -> float:
        # Reshape waypoints
        W = W_flat.reshape(-1, 3)
        n_waypoints = len(W)
        
        # Override start/goal if provided
        if start_point is not None:
            W[0] = start_point
        if goal_point is not None:
            W[-1] = goal_point
        
        cost = 0.0
        
        # Path length cost
        if length_weight > 0:
            segments = np.diff(W, axis=0)
            lengths = np.linalg.norm(segments, axis=1)
            path_length = np.sum(lengths)
            cost += length_weight * path_length
        
        # Smoothness cost (minimize acceleration)
        if smoothness_weight > 0 and n_waypoints > 2:
            acceleration = np.diff(W, n=2, axis=0)
            smoothness = np.sum(np.linalg.norm(acceleration, axis=1))
            cost += smoothness_weight * smoothness
        
        # Collision and clearance costs
        if obstacles and (collision_weight > 0 or clearance_weight > 0):
            for obs in obstacles:
                center = np.array(obs['center'])
                radius = obs['radius']
                
                # Check each waypoint
                for w in W:
                    dist = np.linalg.norm(w - center)
                    
                    # Collision penalty
                    if dist < radius:
                        penetration = radius - dist
                        cost += collision_weight * (penetration ** 2)
                    
                    # Clearance penalty
                    elif dist < radius + min_clearance:
                        clearance_violation = (radius + min_clearance) - dist
                        cost += clearance_weight * clearance_violation
                
                # Check segments between waypoints
                for i in range(n_waypoints - 1):
                    # Simple approximation: check midpoint
                    midpoint = (W[i] + W[i+1]) / 2
                    dist = np.linalg.norm(midpoint - center)
                    
                    if dist < radius:
                        penetration = radius - dist
                        cost += collision_weight * (penetration ** 2) * 0.5
        
        # Goal reaching cost
        if goal_point is not None and goal_weight > 0:
            goal_error = np.linalg.norm(W[-1] - goal_point)
            cost += goal_weight * goal_error
        
        return cost
    
    return cost_function


def trajectory_cost(
    W: np.ndarray,
    target_pose: Optional[Dict[str, np.ndarray]] = None,
    obstacles: Optional[List[Dict[str, Any]]] = None,
    env: Optional[Any] = None,
    length_weight: float = 1.0,
    collision_weight: float = 100.0,
    near_collision_weight: float = 10.0,
    pose_position_weight: float = 5.0,
    pose_orientation_weight: float = 2.0,
    smoothness_weight: float = 0.5,
    collision_threshold: float = 0.02,
    near_collision_distance: float = 0.05,
    n_ray_samples: int = 10,
    use_parallel_rays: bool = False,
    ray_offsets: Optional[List[np.ndarray]] = None,
    end_effector_radius: float = 0.02,
    debug: bool = False
) -> float:
    """
    Comprehensive trajectory cost function with ray-based collision detection.
    
    This function evaluates a trajectory (sequence of waypoints) based on:
    1. Path length - minimize total distance traveled
    2. Collision penalty - heavy penalty for collisions detected via ray casting
    3. Near-collision penalty - penalty for getting too close to obstacles
    4. Final pose error - error in both position and orientation at goal
    5. Smoothness - minimize acceleration/jerk
    
    Args:
        W: Waypoints array of shape (n_waypoints, 3) or flattened (n_waypoints * 3,)
        target_pose: Target pose with 'position' and 'orientation' (quaternion or matrix)
        obstacles: List of obstacles for collision checking
        env: Environment object with collision checking methods (optional)
        length_weight: Weight for path length cost
        collision_weight: Weight for collision penalty
        near_collision_weight: Weight for near-collision penalty
        pose_position_weight: Weight for position error at goal
        pose_orientation_weight: Weight for orientation error at goal
        smoothness_weight: Weight for trajectory smoothness
        collision_threshold: Distance threshold for collision
        near_collision_distance: Distance for near-collision penalty
        n_ray_samples: Number of samples along each ray for collision checking
        use_parallel_rays: Whether to cast multiple parallel rays
        ray_offsets: Offsets for parallel rays (for thick robot links)
        end_effector_radius: Radius of end-effector for collision checking
        debug: Whether to print debug information
        
    Returns:
        Total cost (scalar)
        
    Example:
        # Define target pose
        target_pose = {
            'position': np.array([1.0, 1.0, 0.5]),
            'orientation': np.array([0, 0, 0, 1])  # Quaternion
        }
        
        # Define obstacles
        obstacles = [
            {'type': 'sphere', 'center': [0.5, 0.5, 0.5], 'radius': 0.1},
            {'type': 'box', 'min': [0.2, 0.2, 0], 'max': [0.3, 0.3, 1.0]}
        ]
        
        # Compute cost
        cost = trajectory_cost(
            waypoints,
            target_pose=target_pose,
            obstacles=obstacles,
            collision_weight=100.0,
            near_collision_weight=10.0
        )
    """
    # Handle flattened waypoints
    if W.ndim == 1:
        W = W.reshape(-1, 3)
    
    n_waypoints = len(W)
    total_cost = 0.0
    
    # Debug tracking
    cost_components = {} if debug else None
    
    # 1. Path Length Cost
    if length_weight > 0:
        segments = np.diff(W, axis=0)
        segment_lengths = np.linalg.norm(segments, axis=1)
        path_length = np.sum(segment_lengths)
        length_cost = length_weight * path_length
        total_cost += length_cost
        
        if debug:
            cost_components['path_length'] = path_length
            cost_components['length_cost'] = length_cost
    
    # 2. Collision and Near-Collision Costs via Ray Testing
    if (collision_weight > 0 or near_collision_weight > 0) and n_waypoints > 1:
        collision_cost = 0.0
        near_collision_cost = 0.0
        n_collisions = 0
        n_near_collisions = 0
        
        # Check each segment between waypoints
        for i in range(n_waypoints - 1):
            start_point = W[i]
            end_point = W[i + 1]
            
            # Perform ray collision test
            collision_info = _ray_collision_test(
                start_point, end_point,
                obstacles=obstacles,
                env=env,
                n_samples=n_ray_samples,
                use_parallel_rays=use_parallel_rays,
                ray_offsets=ray_offsets,
                end_effector_radius=end_effector_radius
            )
            
            # Check for collisions
            if collision_info['collision']:
                # Collision detected
                penetration = collision_info['penetration_depth']
                collision_cost += collision_weight * (1.0 + penetration ** 2)
                n_collisions += 1
            
            # Check for near-collisions
            min_distance = collision_info['min_distance']
            if min_distance < near_collision_distance and not collision_info['collision']:
                # Near collision
                proximity = (near_collision_distance - min_distance) / near_collision_distance
                near_collision_cost += near_collision_weight * (proximity ** 2)
                n_near_collisions += 1
        
        # Also check waypoints themselves
        for i, waypoint in enumerate(W):
            point_collision = _point_collision_test(
                waypoint,
                obstacles=obstacles,
                env=env,
                radius=end_effector_radius
            )
            
            if point_collision['collision']:
                collision_cost += collision_weight * 0.5  # Half weight for waypoint collisions
                n_collisions += 1
            elif point_collision['distance'] < near_collision_distance:
                proximity = (near_collision_distance - point_collision['distance']) / near_collision_distance
                near_collision_cost += near_collision_weight * (proximity ** 2) * 0.5
                n_near_collisions += 1
        
        total_cost += collision_cost + near_collision_cost
        
        if debug:
            cost_components['n_collisions'] = n_collisions
            cost_components['n_near_collisions'] = n_near_collisions
            cost_components['collision_cost'] = collision_cost
            cost_components['near_collision_cost'] = near_collision_cost
    
    # 3. Final Pose Error
    if target_pose is not None and (pose_position_weight > 0 or pose_orientation_weight > 0):
        final_position = W[-1]
        
        # Position error
        if 'position' in target_pose and pose_position_weight > 0:
            target_position = np.array(target_pose['position'])
            position_error = np.linalg.norm(final_position - target_position)
            position_cost = pose_position_weight * position_error ** 2
            total_cost += position_cost
            
            if debug:
                cost_components['position_error'] = position_error
                cost_components['position_cost'] = position_cost
        
        # Orientation error (if provided)
        if 'orientation' in target_pose and pose_orientation_weight > 0:
            # Note: This assumes the waypoints have associated orientations
            # For simplicity, we can penalize deviation from vertical or target heading
            # This is a simplified version - extend as needed for full pose
            
            # Simple approach: penalize deviation from target z-height
            if 'z_orientation' in target_pose:
                z_error = abs(final_position[2] - target_pose['z_orientation'])
                orientation_cost = pose_orientation_weight * z_error ** 2
                total_cost += orientation_cost
                
                if debug:
                    cost_components['orientation_cost'] = orientation_cost
    
    # 4. Smoothness Cost (minimize acceleration)
    if smoothness_weight > 0 and n_waypoints > 2:
        # Second derivative (acceleration)
        acceleration = np.diff(W, n=2, axis=0)
        smoothness = np.sum(np.linalg.norm(acceleration, axis=1) ** 2)
        smoothness_cost = smoothness_weight * smoothness
        total_cost += smoothness_cost
        
        if debug:
            cost_components['smoothness'] = smoothness
            cost_components['smoothness_cost'] = smoothness_cost
    
    # Debug output
    if debug:
        print("Trajectory Cost Breakdown:")
        for key, value in cost_components.items():
            print(f"  {key}: {value:.4f}")
        print(f"  TOTAL: {total_cost:.4f}")
    
    return total_cost


def _ray_collision_test(
    start: np.ndarray,
    end: np.ndarray,
    obstacles: Optional[List[Dict[str, Any]]] = None,
    env: Optional[Any] = None,
    n_samples: int = 10,
    use_parallel_rays: bool = False,
    ray_offsets: Optional[List[np.ndarray]] = None,
    end_effector_radius: float = 0.02
) -> Dict[str, Any]:
    """
    Perform ray-based collision testing between two points.
    
    Args:
        start: Start point of ray
        end: End point of ray
        obstacles: List of obstacle dictionaries
        env: Environment with collision checking methods
        n_samples: Number of samples along ray
        use_parallel_rays: Whether to use multiple parallel rays
        ray_offsets: Offsets for parallel rays
        end_effector_radius: Radius for collision checking
        
    Returns:
        Dictionary with collision information
    """
    collision = False
    min_distance = float('inf')
    penetration_depth = 0.0
    collision_point = None
    
    # If environment has ray casting method, use it
    if env is not None and hasattr(env, 'ray_cast'):
        result = env.ray_cast(start, end)
        return {
            'collision': result['hit'],
            'min_distance': result.get('distance', float('inf')),
            'penetration_depth': result.get('penetration', 0.0),
            'collision_point': result.get('hit_point', None)
        }
    
    # Otherwise, perform manual ray testing
    ray_direction = end - start
    ray_length = np.linalg.norm(ray_direction)
    if ray_length < 1e-6:
        return {
            'collision': False,
            'min_distance': float('inf'),
            'penetration_depth': 0.0,
            'collision_point': None
        }
    
    ray_direction = ray_direction / ray_length
    
    # Generate rays to test
    rays_to_test = [(start, end)]
    
    if use_parallel_rays and ray_offsets:
        for offset in ray_offsets:
            rays_to_test.append((start + offset, end + offset))
    
    # Test each ray
    for ray_start, ray_end in rays_to_test:
        # Sample points along the ray
        t_values = np.linspace(0, 1, n_samples)
        
        for t in t_values:
            point = ray_start + t * (ray_end - ray_start)
            
            # Check collision with obstacles
            if obstacles:
                for obstacle in obstacles:
                    dist, is_collision = _check_point_obstacle_distance(
                        point, obstacle, end_effector_radius
                    )
                    
                    if is_collision:
                        collision = True
                        penetration_depth = max(penetration_depth, end_effector_radius - dist)
                        if collision_point is None:
                            collision_point = point
                    
                    min_distance = min(min_distance, dist)
    
    return {
        'collision': collision,
        'min_distance': min_distance,
        'penetration_depth': penetration_depth,
        'collision_point': collision_point
    }


def _point_collision_test(
    point: np.ndarray,
    obstacles: Optional[List[Dict[str, Any]]] = None,
    env: Optional[Any] = None,
    radius: float = 0.02
) -> Dict[str, Any]:
    """
    Test if a point collides with obstacles.
    
    Args:
        point: 3D point to test
        obstacles: List of obstacles
        env: Environment object
        radius: Collision radius
        
    Returns:
        Dictionary with collision information
    """
    if env is not None and hasattr(env, 'check_collision'):
        result = env.check_collision(point, radius)
        return {
            'collision': result,
            'distance': 0.0 if result else float('inf')
        }
    
    min_distance = float('inf')
    collision = False
    
    if obstacles:
        for obstacle in obstacles:
            dist, is_collision = _check_point_obstacle_distance(
                point, obstacle, radius
            )
            
            if is_collision:
                collision = True
            min_distance = min(min_distance, dist)
    
    return {
        'collision': collision,
        'distance': min_distance
    }


def _check_point_obstacle_distance(
    point: np.ndarray,
    obstacle: Dict[str, Any],
    radius: float = 0.0
) -> Tuple[float, bool]:
    """
    Check distance from point to obstacle.
    
    Args:
        point: 3D point
        obstacle: Obstacle dictionary
        radius: Collision radius
        
    Returns:
        Tuple of (distance, is_collision)
    """
    obstacle_type = obstacle.get('type', 'sphere')
    
    if obstacle_type == 'sphere':
        center = np.array(obstacle['center'])
        obs_radius = obstacle['radius']
        dist = np.linalg.norm(point - center) - obs_radius
        return dist, dist < radius
    
    elif obstacle_type == 'box':
        box_min = np.array(obstacle['min'])
        box_max = np.array(obstacle['max'])
        
        # Find closest point on box to the query point
        closest = np.clip(point, box_min, box_max)
        dist = np.linalg.norm(point - closest)
        return dist, dist < radius
    
    elif obstacle_type == 'cylinder':
        base = np.array(obstacle['base'])
        axis = np.array(obstacle['axis'])
        height = obstacle['height']
        cyl_radius = obstacle['radius']
        
        # Project point onto cylinder axis
        to_point = point - base
        proj_length = np.dot(to_point, axis)
        proj_length = np.clip(proj_length, 0, height)
        
        closest_on_axis = base + proj_length * axis
        radial_dist = np.linalg.norm(point - closest_on_axis)
        
        if proj_length >= 0 and proj_length <= height:
            dist = max(0, radial_dist - cyl_radius)
        else:
            # Distance to cylinder caps
            cap_dist = min(
                np.linalg.norm(point - base),
                np.linalg.norm(point - (base + height * axis))
            )
            dist = cap_dist
        
        return dist, dist < radius
    
    elif obstacle_type == 'mesh' and 'vertices' in obstacle:
        # Simplified mesh collision - check distance to vertices
        vertices = np.array(obstacle['vertices'])
        distances = np.linalg.norm(vertices - point[None, :], axis=1)
        min_dist = np.min(distances)
        return min_dist, min_dist < radius
    
    else:
        # Unknown obstacle type - assume no collision
        return float('inf'), False


def create_trajectory_cost_function(
    target_pose: Dict[str, np.ndarray],
    obstacles: Optional[List[Dict[str, Any]]] = None,
    env: Optional[Any] = None,
    weights: Optional[Dict[str, float]] = None
) -> Callable[[np.ndarray], float]:
    """
    Create a trajectory cost function with fixed parameters.
    
    Args:
        target_pose: Target pose for the trajectory
        obstacles: List of obstacles
        env: Environment for collision checking
        weights: Dictionary of weight values
        
    Returns:
        Cost function that takes waypoints and returns cost
        
    Example:
        cost_fn = create_trajectory_cost_function(
            target_pose={'position': [1, 1, 0.5]},
            obstacles=obstacles,
            weights={'collision': 100, 'length': 1.0}
        )
        
        result = optimize_waypoints_cma(cost_fn, initial_waypoints)
    """
    # Default weights
    default_weights = {
        'length': 1.0,
        'collision': 100.0,
        'near_collision': 10.0,
        'pose_position': 5.0,
        'pose_orientation': 2.0,
        'smoothness': 0.5
    }
    
    if weights:
        default_weights.update(weights)
    
    def cost_function(W: np.ndarray) -> float:
        return trajectory_cost(
            W,
            target_pose=target_pose,
            obstacles=obstacles,
            env=env,
            length_weight=default_weights['length'],
            collision_weight=default_weights['collision'],
            near_collision_weight=default_weights['near_collision'],
            pose_position_weight=default_weights['pose_position'],
            pose_orientation_weight=default_weights['pose_orientation'],
            smoothness_weight=default_weights['smoothness']
        )
    
    return cost_function


def optimize_waypoints_scipy(
    cost_fn: Callable[[np.ndarray], float],
    W0: Union[List[Tuple[float, float, float]], np.ndarray],
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    method: str = 'L-BFGS-B',
    options: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Optimize waypoints using scipy optimization methods.
    
    Args:
        cost_fn: Cost function for waypoints
        W0: Initial waypoints
        bounds: Optional bounds
        method: Optimization method ('L-BFGS-B', 'SLSQP', 'trust-constr')
        options: Optimizer options
        verbose: Whether to print progress
        
    Returns:
        Optimization result dictionary
    """
    # Convert initial waypoints
    if isinstance(W0, list):
        W0_array = np.array([list(w) for w in W0])
    else:
        W0_array = np.array(W0)
    
    x0 = W0_array.flatten()
    n_waypoints = len(W0_array)
    
    # Prepare bounds for scipy
    scipy_bounds = None
    if bounds is not None:
        lower, upper = bounds
        if len(lower) == 3:  # Per-dimension
            lower = np.tile(lower, n_waypoints)
            upper = np.tile(upper, n_waypoints)
        scipy_bounds = list(zip(lower, upper))
    
    # Default options
    if options is None:
        options = {'maxiter': 1000, 'disp': verbose}
    
    # Track convergence
    convergence_history = []
    
    def wrapped_cost(x):
        cost = cost_fn(x)
        convergence_history.append(cost)
        return cost
    
    # Run optimization
    start_time = time.time()
    result = minimize(
        wrapped_cost,
        x0,
        method=method,
        bounds=scipy_bounds,
        options=options
    )
    
    # Format results
    optimal_waypoints_array = result.x.reshape(n_waypoints, 3)
    optimal_waypoints = [tuple(w) for w in optimal_waypoints_array]
    
    return {
        'optimal_waypoints': optimal_waypoints,
        'optimal_waypoints_array': optimal_waypoints_array,
        'optimal_cost': result.fun,
        'initial_cost': convergence_history[0] if convergence_history else None,
        'convergence_history': convergence_history,
        'success': result.success,
        'message': result.message,
        'n_evaluations': result.nfev,
        'optimization_time': time.time() - start_time
    }


def optimize_waypoints_genetic(
    cost_fn: Callable[[np.ndarray], float],
    W0: Union[List[Tuple[float, float, float]], np.ndarray],
    bounds: Tuple[np.ndarray, np.ndarray],
    popsize: int = 15,
    maxiter: int = 100,
    mutation: float = 0.5,
    recombination: float = 0.7,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Optimize waypoints using differential evolution (genetic algorithm).
    
    Args:
        cost_fn: Cost function for waypoints
        W0: Initial waypoints (used for shape)
        bounds: Required bounds for differential evolution
        popsize: Population size multiplier
        maxiter: Maximum iterations
        mutation: Mutation constant
        recombination: Crossover probability
        verbose: Whether to print progress
        
    Returns:
        Optimization result dictionary
    """
    # Convert initial waypoints
    if isinstance(W0, list):
        W0_array = np.array([list(w) for w in W0])
    else:
        W0_array = np.array(W0)
    
    n_waypoints = len(W0_array)
    n_dims = W0_array.size
    
    # Prepare bounds
    lower, upper = bounds
    if len(lower) == 3:  # Per-dimension
        lower = np.tile(lower, n_waypoints)
        upper = np.tile(upper, n_waypoints)
    
    bounds_list = list(zip(lower, upper))
    
    # Track convergence
    convergence_history = []
    
    def wrapped_cost(x):
        cost = cost_fn(x)
        convergence_history.append(cost)
        if verbose and len(convergence_history) % 100 == 0:
            logger.info(f"  Eval {len(convergence_history)}: Cost = {cost:.6f}")
        return cost
    
    # Run differential evolution
    start_time = time.time()
    result = differential_evolution(
        wrapped_cost,
        bounds_list,
        popsize=popsize,
        maxiter=maxiter,
        mutation=mutation,
        recombination=recombination,
        disp=verbose,
        workers=1
    )
    
    # Format results
    optimal_waypoints_array = result.x.reshape(n_waypoints, 3)
    optimal_waypoints = [tuple(w) for w in optimal_waypoints_array]
    
    return {
        'optimal_waypoints': optimal_waypoints,
        'optimal_waypoints_array': optimal_waypoints_array,
        'optimal_cost': result.fun,
        'convergence_history': convergence_history,
        'success': result.success,
        'message': result.message,
        'n_evaluations': result.nfev,
        'optimization_time': time.time() - start_time
    }


def visualize_waypoint_optimization(
    initial_waypoints: np.ndarray,
    optimal_waypoints: np.ndarray,
    convergence_history: List[float],
    obstacles: Optional[List[Dict[str, Any]]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize waypoint optimization results.
    
    Args:
        initial_waypoints: Initial waypoint positions
        optimal_waypoints: Optimized waypoint positions
        convergence_history: Cost values over iterations
        obstacles: Optional list of obstacles to visualize
        save_path: Optional path to save figure
    """
    fig = plt.figure(figsize=(15, 5))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Plot initial trajectory
    ax1.plot(initial_waypoints[:, 0], initial_waypoints[:, 1], initial_waypoints[:, 2],
             'b--o', label='Initial', alpha=0.5)
    
    # Plot optimal trajectory
    ax1.plot(optimal_waypoints[:, 0], optimal_waypoints[:, 1], optimal_waypoints[:, 2],
             'r-o', label='Optimal', linewidth=2)
    
    # Plot obstacles
    if obstacles:
        for obs in obstacles:
            center = obs['center']
            radius = obs['radius']
            
            # Draw sphere for obstacle
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
            y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
            z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
            
            ax1.plot_surface(x, y, z, alpha=0.3, color='gray')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Waypoint Optimization')
    ax1.legend()
    
    # 2D projection (XY plane)
    ax2 = fig.add_subplot(132)
    ax2.plot(initial_waypoints[:, 0], initial_waypoints[:, 1], 'b--o', 
             label='Initial', alpha=0.5)
    ax2.plot(optimal_waypoints[:, 0], optimal_waypoints[:, 1], 'r-o', 
             label='Optimal', linewidth=2)
    
    if obstacles:
        for obs in obstacles:
            circle = plt.Circle(obs['center'][:2], obs['radius'], 
                               color='gray', alpha=0.3)
            ax2.add_patch(circle)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Projection')
    ax2.legend()
    ax2.axis('equal')
    ax2.grid(True)
    
    # Convergence plot
    ax3 = fig.add_subplot(133)
    if isinstance(convergence_history[0], (list, np.ndarray)):
        # Plot mean and std if history contains populations
        means = [np.mean(gen) for gen in convergence_history]
        stds = [np.std(gen) for gen in convergence_history]
        iterations = range(len(means))
        
        ax3.plot(iterations, means, 'b-', label='Mean')
        ax3.fill_between(iterations, 
                         np.array(means) - np.array(stds),
                         np.array(means) + np.array(stds),
                         alpha=0.3)
        
        # Also plot best
        bests = [np.min(gen) for gen in convergence_history]
        ax3.plot(iterations, bests, 'r-', label='Best')
    else:
        # Simple convergence history
        ax3.plot(convergence_history, 'b-')
    
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Cost')
    ax3.set_title('Convergence')
    ax3.legend()
    ax3.grid(True)
    ax3.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved optimization visualization to {save_path}")
    
    plt.show()


@dataclass
class WaypointOptimizationProblem:
    """
    Data class for waypoint optimization problems.
    """
    initial_waypoints: np.ndarray
    cost_function: Callable
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    constraints: Optional[List[Dict[str, Any]]] = None
    obstacles: Optional[List[Dict[str, Any]]] = None
    start_point: Optional[np.ndarray] = None
    goal_point: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None


def batch_optimize_waypoints(
    problems: List[WaypointOptimizationProblem],
    method: str = 'cma',
    parallel: bool = True,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Optimize multiple waypoint problems in batch.
    
    Args:
        problems: List of optimization problems
        method: Optimization method ('cma', 'scipy', 'genetic')
        parallel: Whether to run in parallel
        **kwargs: Additional arguments for optimizer
        
    Returns:
        List of optimization results
    """
    results = []
    
    for i, problem in enumerate(problems):
        logger.info(f"Optimizing problem {i+1}/{len(problems)}")
        
        if method == 'cma':
            result = optimize_waypoints_cma(
                problem.cost_function,
                problem.initial_waypoints,
                problem.bounds,
                **kwargs
            )
        elif method == 'scipy':
            result = optimize_waypoints_scipy(
                problem.cost_function,
                problem.initial_waypoints,
                problem.bounds,
                **kwargs
            )
        elif method == 'genetic':
            result = optimize_waypoints_genetic(
                problem.cost_function,
                problem.initial_waypoints,
                problem.bounds,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        result['problem_idx'] = i
        results.append(result)
    
    return results


def parallel_rollout_optimization(
    rollout_fn: Callable,
    W0: Union[List[Tuple[float, float, float]], np.ndarray],
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    n_workers: Optional[int] = None,
    method: str = 'cma',
    budget_iters: int = 40,
    popsize: int = 12,
    sigma0: float = 0.1,
    early_stop_threshold: Optional[float] = None,
    early_stop_patience: int = 10,
    stream_progress: bool = True,
    progress_callback: Optional[Callable] = None,
    use_processes: bool = True,
    timeout_per_rollout: Optional[float] = None,
    verbose: bool = True,
    **optimizer_kwargs
) -> Dict[str, Any]:
    """
    Optimize waypoints using parallel rollout evaluation with streaming progress.
    
    This function runs trajectory rollouts in parallel across multiple processes/threads,
    streams the best cost per iteration, and supports early stopping.
    
    Args:
        rollout_fn: Function that takes waypoints and returns cost (simulation-based)
        W0: Initial waypoints
        bounds: Optional bounds for waypoints
        n_workers: Number of parallel workers (None = CPU count)
        method: Optimization method ('cma', 'scipy', 'genetic')
        budget_iters: Maximum iterations
        popsize: Population size for evolutionary methods
        sigma0: Initial step size for CMA-ES
        early_stop_threshold: Stop if cost goes below this threshold
        early_stop_patience: Iterations without improvement before stopping
        stream_progress: Whether to stream progress updates
        progress_callback: Callback for progress updates
        use_processes: Use processes (True) or threads (False)
        timeout_per_rollout: Maximum time per rollout evaluation
        verbose: Print progress
        **optimizer_kwargs: Additional optimizer arguments
        
    Returns:
        Optimization results with streaming history
        
    Example:
        def evaluate_trajectory(W_flat):
            # Run simulation with waypoints
            W = W_flat.reshape(-1, 3)
            success, _, trajectory = rollout_policy(
                create_waypoint_following_expert(W),
                env,
                max_steps=100,
                return_trajectory=True
            )
            return -trajectory['summary']['total_reward']
        
        result = parallel_rollout_optimization(
            evaluate_trajectory,
            initial_waypoints,
            n_workers=4,
            budget_iters=30,
            early_stop_threshold=-100.0,
            stream_progress=True
        )
    """
    # Setup workers
    if n_workers is None:
        n_workers = min(cpu_count(), popsize) if method == 'cma' else cpu_count()
    
    if verbose:
        logger.info(f"Starting parallel optimization with {n_workers} workers")
        logger.info(f"Method: {method}, Population: {popsize}, Iterations: {budget_iters}")
    
    # Convert initial waypoints
    if isinstance(W0, list):
        W0_array = np.array([list(w) for w in W0])
    else:
        W0_array = np.array(W0)
    
    # Setup progress tracking
    manager = Manager() if use_processes else None
    progress_queue = manager.Queue() if manager else Queue()
    best_cost_history = []
    iteration_times = []
    
    # Early stopping state
    best_cost = float('inf')
    patience_counter = 0
    should_stop = False
    
    # Progress streaming thread
    if stream_progress:
        stream_thread = threading.Thread(
            target=_stream_progress,
            args=(progress_queue, best_cost_history, iteration_times, verbose),
            daemon=True
        )
        stream_thread.start()
    
    # Create parallel evaluator
    evaluator = ParallelEvaluator(
        rollout_fn,
        n_workers=n_workers,
        use_processes=use_processes,
        timeout=timeout_per_rollout
    )
    
    try:
        # Define cost function with parallel evaluation
        def parallel_cost_fn(candidates):
            """Evaluate multiple candidates in parallel."""
            if isinstance(candidates, np.ndarray) and candidates.ndim == 1:
                # Single candidate
                candidates = [candidates]
            
            # Submit all rollouts
            costs = evaluator.evaluate_batch(candidates)
            
            # Update progress
            min_cost = min(costs)
            if stream_progress:
                progress_queue.put({
                    'type': 'iteration',
                    'best_cost': min_cost,
                    'mean_cost': np.mean(costs),
                    'std_cost': np.std(costs),
                    'timestamp': time.time()
                })
            
            # Check for improvement and early stopping
            nonlocal best_cost, patience_counter, should_stop
            
            if min_cost < best_cost:
                improvement = best_cost - min_cost
                best_cost = min_cost
                patience_counter = 0
                
                if verbose:
                    logger.info(f"New best cost: {best_cost:.6f} (improved by {improvement:.6f})")
            else:
                patience_counter += 1
            
            # Early stopping checks
            if early_stop_threshold is not None and best_cost <= early_stop_threshold:
                should_stop = True
                if verbose:
                    logger.info(f"Reached target threshold {early_stop_threshold}")
            
            if patience_counter >= early_stop_patience:
                should_stop = True
                if verbose:
                    logger.info(f"Early stopping: no improvement for {patience_counter} iterations")
            
            return costs
        
        # Custom callback that checks for early stopping
        def optimization_callback(info):
            if progress_callback:
                progress_callback(info)
            return should_stop  # Signal to stop if True
        
        # Run optimization based on method
        if method == 'cma':
            result = _optimize_cma_parallel(
                parallel_cost_fn,
                W0_array,
                bounds,
                budget_iters,
                popsize,
                sigma0,
                optimization_callback,
                should_stop,
                **optimizer_kwargs
            )
        
        elif method == 'genetic':
            result = _optimize_genetic_parallel(
                parallel_cost_fn,
                W0_array,
                bounds,
                popsize,
                budget_iters,
                optimization_callback,
                should_stop,
                **optimizer_kwargs
            )
        
        else:
            raise ValueError(f"Unsupported method for parallel optimization: {method}")
        
    finally:
        # Cleanup
        evaluator.shutdown()
        if stream_progress:
            progress_queue.put({'type': 'stop'})
    
    # Add streaming history to results
    result['best_cost_history'] = best_cost_history
    result['iteration_times'] = iteration_times
    result['n_workers'] = n_workers
    result['early_stopped'] = should_stop
    
    if verbose:
        logger.info(f"Optimization complete: Best cost = {result['optimal_cost']:.6f}")
        logger.info(f"Total evaluations: {result['n_evaluations']}")
        logger.info(f"Time: {result['optimization_time']:.2f} seconds")
    
    return result


class ParallelEvaluator:
    """
    Manages parallel evaluation of rollouts.
    """
    
    def __init__(self, eval_fn: Callable, n_workers: int = 4,
                 use_processes: bool = True, timeout: Optional[float] = None):
        self.eval_fn = eval_fn
        self.n_workers = n_workers
        self.timeout = timeout
        self.use_processes = use_processes
        
        if use_processes:
            # Workaround for Windows multiprocessing
            mp.set_start_method('spawn', force=True)
            self.executor = ProcessPoolExecutor(max_workers=n_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=n_workers)
    
    def evaluate_batch(self, candidates: List[np.ndarray]) -> List[float]:
        """Evaluate a batch of candidates in parallel."""
        futures = []
        
        # Submit all evaluations
        for candidate in candidates:
            future = self.executor.submit(self._safe_evaluate, candidate)
            futures.append(future)
        
        # Collect results
        costs = []
        for future in as_completed(futures, timeout=self.timeout):
            try:
                cost = future.result(timeout=self.timeout)
                costs.append(cost)
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")
                costs.append(float('inf'))  # Penalty for failed evaluation
        
        return costs
    
    def _safe_evaluate(self, candidate: np.ndarray) -> float:
        """Safely evaluate a single candidate with error handling."""
        try:
            cost = self.eval_fn(candidate)
            if not np.isfinite(cost):
                return 1e10
            return float(cost)
        except Exception as e:
            logger.warning(f"Rollout failed: {e}")
            return 1e10
    
    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)


def _stream_progress(queue: Queue, best_costs: List, times: List, verbose: bool):
    """Stream progress updates from queue."""
    start_time = time.time()
    
    while True:
        try:
            msg = queue.get(timeout=1.0)
            
            if msg['type'] == 'stop':
                break
            
            if msg['type'] == 'iteration':
                best_costs.append(msg['best_cost'])
                times.append(msg['timestamp'] - start_time)
                
                if verbose:
                    logger.info(f"Iteration {len(best_costs)}: "
                               f"Best={msg['best_cost']:.6f}, "
                               f"Mean={msg['mean_cost']:.6f}, "
                               f"Std={msg['std_cost']:.6f}")
        
        except Empty:
            continue
        except Exception as e:
            logger.error(f"Progress streaming error: {e}")
            break


def _optimize_cma_parallel(
    cost_fn: Callable,
    x0: np.ndarray,
    bounds: Optional[Tuple],
    budget_iters: int,
    popsize: int,
    sigma0: float,
    callback: Callable,
    should_stop_ref: bool,
    **kwargs
) -> Dict[str, Any]:
    """CMA-ES optimization with parallel evaluation."""
    import cma
    
    n_dims = x0.size
    start_time = time.time()
    
    # Setup CMA options
    cma_options = {
        'maxiter': budget_iters,
        'popsize': popsize,
        'verb_disp': 0,
        'verb_log': 0,
        'tolfun': 1e-11,
        'tolx': 1e-11,
    }
    
    if bounds is not None:
        lower, upper = bounds
        if len(lower) == 3:  # Per-dimension
            lower = np.tile(lower, len(x0) // 3)
            upper = np.tile(upper, len(x0) // 3)
        cma_options['bounds'] = [lower.tolist(), upper.tolist()]
        x0 = np.clip(x0, lower, upper)
    
    cma_options.update(kwargs)
    
    # Create CMA instance
    es = cma.CMAEvolutionStrategy(x0.flatten(), sigma0, cma_options)
    
    # Optimization loop
    iteration = 0
    best_solution = None
    best_cost = float('inf')
    n_evaluations = 0
    
    while not es.stop() and iteration < budget_iters and not should_stop_ref:
        # Get population
        solutions = es.ask()
        
        # Evaluate in parallel
        costs = cost_fn(solutions)
        n_evaluations += len(solutions)
        
        # Update CMA
        es.tell(solutions, costs)
        
        # Track best
        min_idx = np.argmin(costs)
        if costs[min_idx] < best_cost:
            best_cost = costs[min_idx]
            best_solution = solutions[min_idx].copy()
        
        # Callback
        if callback:
            stop_signal = callback({
                'iteration': iteration,
                'best_cost': best_cost,
                'mean_cost': np.mean(costs),
                'cma_sigma': es.sigma
            })
            if stop_signal:
                break
        
        iteration += 1
    
    # Format results
    n_waypoints = len(x0) // 3
    optimal_waypoints_array = best_solution.reshape(n_waypoints, 3)
    optimal_waypoints = [tuple(w) for w in optimal_waypoints_array]
    
    return {
        'optimal_waypoints': optimal_waypoints,
        'optimal_waypoints_array': optimal_waypoints_array,
        'optimal_cost': best_cost,
        'n_evaluations': n_evaluations,
        'optimization_time': time.time() - start_time,
        'iterations': iteration,
        'final_sigma': es.sigma
    }


def _optimize_genetic_parallel(
    cost_fn: Callable,
    x0: np.ndarray,
    bounds: Tuple,
    popsize: int,
    maxiter: int,
    callback: Callable,
    should_stop_ref: bool,
    **kwargs
) -> Dict[str, Any]:
    """Genetic algorithm optimization with parallel evaluation."""
    from scipy.optimize import differential_evolution
    
    start_time = time.time()
    n_evaluations = 0
    iteration = 0
    best_cost = float('inf')
    best_solution = None
    
    # Prepare bounds
    lower, upper = bounds
    if len(lower) == 3:
        lower = np.tile(lower, len(x0) // 3)
        upper = np.tile(upper, len(x0) // 3)
    bounds_list = list(zip(lower, upper))
    
    # Custom callback for scipy
    def scipy_callback(xk, convergence):
        nonlocal iteration, should_stop_ref
        iteration += 1
        
        if callback:
            stop_signal = callback({
                'iteration': iteration,
                'convergence': convergence
            })
            if stop_signal or should_stop_ref:
                return True
        return False
    
    # Parallel evaluation wrapper
    def parallel_eval(x):
        nonlocal n_evaluations
        n_evaluations += 1
        costs = cost_fn([x])
        return costs[0]
    
    # Run optimization
    result = differential_evolution(
        parallel_eval,
        bounds_list,
        popsize=popsize,
        maxiter=maxiter,
        callback=scipy_callback,
        workers=1,  # We handle parallelization ourselves
        **kwargs
    )
    
    # Format results
    n_waypoints = len(x0) // 3
    optimal_waypoints_array = result.x.reshape(n_waypoints, 3)
    optimal_waypoints = [tuple(w) for w in optimal_waypoints_array]
    
    return {
        'optimal_waypoints': optimal_waypoints,
        'optimal_waypoints_array': optimal_waypoints_array,
        'optimal_cost': result.fun,
        'n_evaluations': n_evaluations,
        'optimization_time': time.time() - start_time,
        'iterations': iteration,
        'success': result.success
    }


def create_rollout_cost_function(
    env: Any,
    expert_fn: Callable,
    max_steps: int = 100,
    n_rollouts: int = 1,
    aggregate: str = 'mean',
    success_bonus: float = 10.0,
    collision_penalty: float = 100.0,
    timeout_penalty: float = 50.0
) -> Callable[[np.ndarray], float]:
    """
    Create a cost function based on rollout evaluation.
    
    Args:
        env: Environment for rollouts
        expert_fn: Function to create expert from waypoints
        max_steps: Maximum steps per rollout
        n_rollouts: Number of rollouts per evaluation
        aggregate: How to aggregate multiple rollouts ('mean', 'min', 'max')
        success_bonus: Bonus for successful completion
        collision_penalty: Penalty for collisions
        timeout_penalty: Penalty for timeout
        
    Returns:
        Cost function for optimization
        
    Example:
        cost_fn = create_rollout_cost_function(
            env,
            lambda W: create_waypoint_following_expert(W.reshape(-1, 3)),
            max_steps=200
        )
        
        result = parallel_rollout_optimization(
            cost_fn,
            initial_waypoints,
            n_workers=4
        )
    """
    def rollout_cost(W_flat: np.ndarray) -> float:
        W = W_flat.reshape(-1, 3)
        
        # Create expert from waypoints
        try:
            expert = expert_fn(W)
        except Exception as e:
            logger.warning(f"Expert creation failed: {e}")
            return 1e10
        
        # Run rollouts
        costs = []
        for _ in range(n_rollouts):
            try:
                # Import here to avoid circular dependency
                from cogniforge.core.evaluation import rollout_policy
                
                success, _, trajectory = rollout_policy(
                    expert, env,
                    max_steps=max_steps,
                    render=False,
                    record_video=False,
                    verbose=False,
                    return_trajectory=True
                )
                
                # Compute cost
                total_reward = trajectory['summary']['total_reward']
                steps = trajectory['summary']['steps']
                
                # Base cost is negative reward
                cost = -total_reward
                
                # Add bonuses/penalties
                if success:
                    cost -= success_bonus
                
                if steps >= max_steps:
                    cost += timeout_penalty
                
                # Check for collisions in trajectory
                if 'infos' in trajectory:
                    for info in trajectory['infos']:
                        if info.get('collision', False):
                            cost += collision_penalty
                            break
                
                costs.append(cost)
                
            except Exception as e:
                logger.warning(f"Rollout failed: {e}")
                costs.append(1e10)
        
        # Aggregate costs
        if aggregate == 'mean':
            return np.mean(costs)
        elif aggregate == 'min':
            return np.min(costs)
        elif aggregate == 'max':
            return np.max(costs)
        else:
            raise ValueError(f"Unknown aggregation: {aggregate}")
    
    return rollout_cost


def stream_optimization_progress(
    optimization_fn: Callable,
    *args,
    websocket_port: int = 8765,
    sse_port: int = 5000,
    use_websocket: bool = False,
    use_sse: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Run optimization with real-time progress streaming.
    
    Args:
        optimization_fn: Optimization function to run
        *args: Arguments for optimization function
        websocket_port: Port for WebSocket streaming
        sse_port: Port for SSE streaming
        use_websocket: Enable WebSocket streaming
        use_sse: Enable SSE streaming
        **kwargs: Keyword arguments for optimization function
        
    Returns:
        Optimization results
        
    Example:
        result = stream_optimization_progress(
            parallel_rollout_optimization,
            rollout_cost,
            initial_waypoints,
            n_workers=4,
            use_sse=True,
            sse_port=5000
        )
    """
    from flask import Flask, Response
    import flask
    
    # Setup streaming servers
    servers = []
    progress_queue = Queue()
    
    if use_sse:
        app = Flask(__name__)
        
        @app.route('/optimization/stream')
        def stream():
            def generate():
                while True:
                    try:
                        msg = progress_queue.get(timeout=1.0)
                        if msg['type'] == 'stop':
                            break
                        yield f"data: {json.dumps(msg)}\n\n"
                    except Empty:
                        yield ": heartbeat\n\n"
            
            return Response(
                generate(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no',
                    'Access-Control-Allow-Origin': '*'
                }
            )
        
        sse_thread = threading.Thread(
            target=lambda: app.run(host='localhost', port=sse_port, debug=False),
            daemon=True
        )
        sse_thread.start()
        servers.append(('SSE', sse_port))
        logger.info(f"SSE streaming at http://localhost:{sse_port}/optimization/stream")
    
    # Custom callback to forward progress
    def progress_callback(info):
        progress_queue.put({
            'type': 'progress',
            'timestamp': time.time(),
            **info
        })
    
    # Add callback to kwargs
    kwargs['progress_callback'] = progress_callback
    kwargs['stream_progress'] = False  # We handle streaming ourselves
    
    try:
        # Run optimization
        result = optimization_fn(*args, **kwargs)
    finally:
        # Signal stop
        progress_queue.put({'type': 'stop'})
    
    # Add server info to results
    result['streaming_servers'] = servers
    
    return result


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Testing waypoint optimization with CMA-ES")
    print("=" * 60)
    
    # Define initial waypoints
    W0 = [
        (0.0, 0.0, 0.5),
        (0.3, 0.2, 0.5),
        (0.6, 0.4, 0.5),
        (0.9, 0.8, 0.3),
        (1.0, 1.0, 0.3)
    ]
    
    # Define obstacles
    obstacles = [
        {'center': [0.5, 0.5, 0.5], 'radius': 0.15},
        {'center': [0.8, 0.2, 0.4], 'radius': 0.1}
    ]
    
    # Create cost function
    cost_fn = create_smooth_cost_function(
        length_weight=1.0,
        smoothness_weight=0.5,
        collision_weight=20.0,
        obstacles=obstacles,
        goal_point=np.array([1.0, 1.0, 0.3])
    )
    
    # Define bounds
    bounds = (
        np.array([0.0, 0.0, 0.1]),  # Lower bounds per dimension
        np.array([1.0, 1.0, 1.0])   # Upper bounds per dimension
    )
    
    # Optimize with CMA-ES
    print("\nOptimizing with CMA-ES...")
    result = optimize_waypoints_cma(
        cost_fn,
        W0,
        bounds=bounds,
        budget_iters=40,
        popsize=12,
        sigma0=0.1,
        verbose=True
    )
    
    print(f"\nOptimization Results:")
    print(f"  Initial cost: {result['initial_cost']:.4f}")
    print(f"  Optimal cost: {result['optimal_cost']:.4f}")
    print(f"  Improvement: {result['improvement_percent']:.1f}%")
    print(f"  Evaluations: {result['n_evaluations']}")
    print(f"  Time: {result['optimization_time']:.2f} seconds")
    
    print("\nOptimal waypoints:")
    for i, w in enumerate(result['optimal_waypoints']):
        print(f"  W[{i}] = ({w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f})")
    
    # Visualize results
    initial_array = np.array([list(w) for w in W0])
    visualize_waypoint_optimization(
        initial_array,
        result['optimal_waypoints_array'],
        result['best_costs'],
        obstacles=obstacles,
        save_path='waypoint_optimization.png'
    )
    
    # Test scipy optimization for comparison
    print("\n" + "=" * 60)
    print("Testing scipy optimization (L-BFGS-B)")
    print("=" * 60)
    
    scipy_result = optimize_waypoints_scipy(
        cost_fn,
        W0,
        bounds=bounds,
        method='L-BFGS-B',
        verbose=True
    )
    
    print(f"\nScipy Results:")
    print(f"  Optimal cost: {scipy_result['optimal_cost']:.4f}")
    print(f"  Success: {scipy_result['success']}")
    print(f"  Message: {scipy_result['message']}")
    
    # Compare methods
    print("\n" + "=" * 60)
    print("Method Comparison:")
    print(f"  CMA-ES cost: {result['optimal_cost']:.4f}")
    print(f"  Scipy cost:  {scipy_result['optimal_cost']:.4f}")
    print(f"  CMA-ES is {'better' if result['optimal_cost'] < scipy_result['optimal_cost'] else 'worse'}")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)