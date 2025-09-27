"""
Post-optimization refinement utilities.

This module provides functions for refining optimized trajectories through
expert demonstration generation and quick behavior cloning updates.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, Union, List, Callable
import logging
import time
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


def refine_with_optimal_trajectory(
    optimal_waypoints: Union[List[Tuple[float, float, float]], np.ndarray],
    env: Any,
    policy: Optional[nn.Module] = None,
    n_demonstrations: int = 10,
    max_steps_per_demo: int = 200,
    bc_epochs: int = 5,
    bc_batch_size: int = 32,
    bc_lr: float = 1e-3,
    action_repeat: int = 1,
    augment_data: bool = True,
    noise_level: float = 0.01,
    save_demos: Optional[str] = None,
    save_refined_policy: Optional[str] = None,
    device: Optional[str] = None,
    verbose: bool = True,
    return_all_data: bool = False
) -> Dict[str, Any]:
    """
    Refine trajectory using optimal waypoints with expert demonstrations and BC.
    
    After optimization finds optimal waypoints W*, this function:
    1. Creates an expert from the optimal waypoints
    2. Generates demonstration trajectories
    3. Collects state-action pairs
    4. Performs quick BC refinement on a policy
    
    Args:
        optimal_waypoints: Optimized waypoints W* from optimization
        env: Environment for rollouts
        policy: Existing policy to refine (creates new if None)
        n_demonstrations: Number of expert demonstrations to generate
        max_steps_per_demo: Maximum steps per demonstration
        bc_epochs: Number of BC training epochs for refinement
        bc_batch_size: Batch size for BC training
        bc_lr: Learning rate for BC refinement
        action_repeat: Action repeat factor for demonstrations
        augment_data: Whether to augment collected data
        noise_level: Noise level for data augmentation
        save_demos: Optional path to save demonstrations
        save_refined_policy: Optional path to save refined policy
        device: Device for training ('cuda' or 'cpu')
        verbose: Whether to print progress
        return_all_data: Return all collected data and trajectories
        
    Returns:
        Dictionary containing:
        - 'refined_policy': Refined policy model
        - 'optimal_expert': Expert created from optimal waypoints
        - 'demonstration_stats': Statistics from demonstrations
        - 'bc_results': BC training results
        - 'evaluation_results': Evaluation of refined policy
        - 'collected_data': (Optional) All collected state-action pairs
        - 'trajectories': (Optional) All demonstration trajectories
        
    Example:
        # After optimization
        result = optimize_waypoints_cma(cost_fn, initial_waypoints)
        optimal_waypoints = result['optimal_waypoints']
        
        # Refine with BC
        refinement = refine_with_optimal_trajectory(
            optimal_waypoints,
            env,
            policy=existing_policy,  # Or None to create new
            n_demonstrations=20,
            bc_epochs=5
        )
        
        refined_policy = refinement['refined_policy']
    """
    start_time = time.time()
    
    if verbose:
        logger.info("Starting trajectory refinement with optimal waypoints")
        logger.info(f"Waypoints: {len(optimal_waypoints)} points")
    
    # Convert waypoints to array if needed
    if isinstance(optimal_waypoints, list):
        waypoints_array = np.array([list(w) for w in optimal_waypoints])
    else:
        waypoints_array = optimal_waypoints
    
    # Step 1: Create expert from optimal waypoints
    if verbose:
        logger.info("Step 1: Creating expert from optimal waypoints")
    
    from cogniforge.core.expert_script import create_waypoint_following_expert
    
    optimal_expert = create_waypoint_following_expert(
        waypoints_array,
        control_type='position',
        interpolation='cubic',  # Smooth interpolation
        waypoint_tolerance=0.02
    )
    
    # Step 2: Generate expert demonstrations
    if verbose:
        logger.info(f"Step 2: Generating {n_demonstrations} expert demonstrations")
    
    demonstrations = _generate_expert_demonstrations(
        optimal_expert,
        env,
        n_demonstrations=n_demonstrations,
        max_steps=max_steps_per_demo,
        action_repeat=action_repeat,
        verbose=verbose
    )
    
    # Compute demonstration statistics
    demo_stats = _compute_demonstration_stats(demonstrations)
    
    if verbose:
        logger.info(f"  Success rate: {demo_stats['success_rate']:.2%}")
        logger.info(f"  Avg reward: {demo_stats['mean_reward']:.3f}")
        logger.info(f"  Avg length: {demo_stats['mean_length']:.1f} steps")
    
    # Step 3: Collect and process state-action pairs
    if verbose:
        logger.info("Step 3: Collecting state-action pairs")
    
    X, Y = _collect_state_action_pairs(
        demonstrations,
        augment=augment_data,
        noise_level=noise_level
    )
    
    if verbose:
        logger.info(f"  Collected {len(X)} state-action pairs")
        logger.info(f"  State dim: {X.shape[1]}, Action dim: {Y.shape[1]}")
    
    # Save demonstrations if requested
    if save_demos:
        _save_demonstrations(demonstrations, X, Y, save_demos)
        if verbose:
            logger.info(f"  Saved demonstrations to {save_demos}")
    
    # Step 4: Create or use existing policy
    if policy is None:
        if verbose:
            logger.info("Step 4: Creating new policy network")
        
        from cogniforge.core.policy import SimplePolicy
        
        policy = SimplePolicy(
            obs_dim=X.shape[1],
            act_dim=Y.shape[1],
            hidden_dim=64,
            n_hidden_layers=2,
            continuous=True
        )
    else:
        if verbose:
            logger.info("Step 4: Using existing policy for refinement")
    
    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = policy.to(device)
    
    # Step 5: Quick BC refinement
    if verbose:
        logger.info(f"Step 5: BC refinement for {bc_epochs} epochs")
    
    from cogniforge.core.training import fit_bc
    
    bc_results = fit_bc(
        policy,
        X, Y,
        epochs=bc_epochs,
        batch_size=bc_batch_size,
        lr=bc_lr,
        val_split=0.2,
        early_stopping=False,  # Quick training, no early stopping
        verbose=verbose,
        device=device
    )
    
    if verbose:
        logger.info(f"  Final train loss: {bc_results['final_train_loss']:.6f}")
        if bc_results['final_val_loss'] is not None:
            logger.info(f"  Final val loss: {bc_results['final_val_loss']:.6f}")
    
    # Step 6: Evaluate refined policy
    if verbose:
        logger.info("Step 6: Evaluating refined policy")
    
    evaluation_results = _evaluate_refined_policy(
        policy,
        env,
        n_episodes=5,
        max_steps=max_steps_per_demo,
        device=device,
        verbose=verbose
    )
    
    if verbose:
        logger.info(f"  Success rate: {evaluation_results['success_rate']:.2%}")
        logger.info(f"  Mean reward: {evaluation_results['mean_reward']:.3f}")
    
    # Save refined policy if requested
    if save_refined_policy:
        policy.save(save_refined_policy)
        if verbose:
            logger.info(f"Saved refined policy to {save_refined_policy}")
    
    # Compile results
    total_time = time.time() - start_time
    
    results = {
        'refined_policy': policy,
        'optimal_expert': optimal_expert,
        'demonstration_stats': demo_stats,
        'bc_results': bc_results,
        'evaluation_results': evaluation_results,
        'n_demonstrations': n_demonstrations,
        'n_training_samples': len(X),
        'refinement_time': total_time
    }
    
    if return_all_data:
        results['collected_data'] = {'states': X, 'actions': Y}
        results['trajectories'] = demonstrations
    
    if verbose:
        logger.info(f"Refinement complete in {total_time:.1f} seconds")
    
    return results


def _generate_expert_demonstrations(
    expert: Callable,
    env: Any,
    n_demonstrations: int,
    max_steps: int,
    action_repeat: int,
    verbose: bool
) -> List[Dict[str, Any]]:
    """Generate expert demonstrations using the optimal expert."""
    from cogniforge.core.expert_script import run_expert_and_record
    
    demonstrations = []
    successful_demos = 0
    
    for i in range(n_demonstrations):
        try:
            # Run expert and record trajectory
            demo_data = run_expert_and_record(
                expert,
                env,
                max_steps=max_steps,
                record_frequency=1,  # Record every step
                verbose=False
            )
            
            # Check if successful
            success = demo_data.get('success', False)
            if success:
                successful_demos += 1
            
            demonstrations.append(demo_data)
            
            if verbose and (i + 1) % 5 == 0:
                logger.info(f"  Generated {i + 1}/{n_demonstrations} demonstrations "
                           f"({successful_demos} successful)")
        
        except Exception as e:
            logger.warning(f"Demonstration {i} failed: {e}")
            continue
    
    return demonstrations


def _collect_state_action_pairs(
    demonstrations: List[Dict[str, Any]],
    augment: bool = True,
    noise_level: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect and process state-action pairs from demonstrations."""
    all_states = []
    all_actions = []
    
    for demo in demonstrations:
        states = demo.get('states', [])
        actions = demo.get('actions', [])
        
        # Ensure equal length
        min_len = min(len(states) - 1, len(actions))  # -1 because states has final state
        
        for i in range(min_len):
            # Original pair
            state = _process_state(states[i])
            action = _process_action(actions[i])
            
            all_states.append(state)
            all_actions.append(action)
            
            # Data augmentation
            if augment and noise_level > 0:
                # Add noisy versions
                for _ in range(2):  # Add 2 augmented samples per original
                    noisy_state = state + np.random.normal(0, noise_level, state.shape)
                    noisy_action = action + np.random.normal(0, noise_level * 0.5, action.shape)
                    
                    all_states.append(noisy_state)
                    all_actions.append(noisy_action)
    
    return np.array(all_states, dtype=np.float32), np.array(all_actions, dtype=np.float32)


def _process_state(state: Any) -> np.ndarray:
    """Convert state to numpy array."""
    if isinstance(state, np.ndarray):
        return state.flatten()
    elif isinstance(state, (list, tuple)):
        return np.array(state, dtype=np.float32).flatten()
    elif isinstance(state, dict):
        # Extract numeric values from dict
        values = []
        for key in sorted(state.keys()):
            val = state[key]
            if isinstance(val, (int, float)):
                values.append(val)
            elif isinstance(val, (list, tuple, np.ndarray)):
                values.extend(np.array(val).flatten())
        return np.array(values, dtype=np.float32)
    else:
        return np.array([state], dtype=np.float32)


def _process_action(action: Any) -> np.ndarray:
    """Convert action to numpy array."""
    if isinstance(action, np.ndarray):
        return action.flatten()
    elif isinstance(action, (list, tuple)):
        return np.array(action, dtype=np.float32).flatten()
    elif isinstance(action, dict):
        # Common action format
        if 'position' in action:
            return np.array(action['position'], dtype=np.float32)
        elif 'velocity' in action:
            return np.array(action['velocity'], dtype=np.float32)
        else:
            # Extract all numeric values
            values = []
            for key in sorted(action.keys()):
                val = action[key]
                if isinstance(val, (int, float)):
                    values.append(val)
                elif isinstance(val, (list, tuple, np.ndarray)):
                    values.extend(np.array(val).flatten())
            return np.array(values, dtype=np.float32)
    else:
        return np.array([action], dtype=np.float32)


def _compute_demonstration_stats(demonstrations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute statistics from demonstrations."""
    if not demonstrations:
        return {
            'success_rate': 0.0,
            'mean_reward': 0.0,
            'std_reward': 0.0,
            'mean_length': 0.0,
            'std_length': 0.0,
            'n_demonstrations': 0
        }
    
    successes = []
    rewards = []
    lengths = []
    
    for demo in demonstrations:
        successes.append(demo.get('success', False))
        rewards.append(demo.get('total_reward', 0.0))
        lengths.append(demo.get('total_steps', 0))
    
    return {
        'success_rate': np.mean(successes),
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'n_demonstrations': len(demonstrations),
        'n_successful': sum(successes)
    }


def _evaluate_refined_policy(
    policy: nn.Module,
    env: Any,
    n_episodes: int,
    max_steps: int,
    device: str,
    verbose: bool
) -> Dict[str, Any]:
    """Evaluate the refined policy."""
    from cogniforge.core.evaluation import rollout_policy
    
    policy.eval()
    
    successes = []
    rewards = []
    lengths = []
    
    for i in range(n_episodes):
        try:
            success, _, trajectory = rollout_policy(
                policy, env,
                max_steps=max_steps,
                deterministic=True,
                device=device,
                verbose=False,
                return_trajectory=True
            )
            
            successes.append(success)
            rewards.append(trajectory['summary']['total_reward'])
            lengths.append(trajectory['summary']['steps'])
            
        except Exception as e:
            logger.warning(f"Evaluation episode {i} failed: {e}")
            successes.append(False)
            rewards.append(0.0)
            lengths.append(max_steps)
    
    return {
        'success_rate': np.mean(successes),
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'n_episodes': n_episodes,
        'all_rewards': rewards,
        'all_successes': successes
    }


def _save_demonstrations(
    demonstrations: List[Dict[str, Any]],
    X: np.ndarray,
    Y: np.ndarray,
    save_path: str
) -> None:
    """Save demonstrations and collected data."""
    import pickle
    
    save_data = {
        'demonstrations': demonstrations,
        'states': X,
        'actions': Y,
        'metadata': {
            'n_demonstrations': len(demonstrations),
            'n_samples': len(X),
            'state_dim': X.shape[1] if X.ndim > 1 else 1,
            'action_dim': Y.shape[1] if Y.ndim > 1 else 1,
            'timestamp': time.time()
        }
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)


def iterative_refinement(
    initial_waypoints: Union[List[Tuple[float, float, float]], np.ndarray],
    env: Any,
    cost_fn: Callable,
    n_iterations: int = 3,
    optimization_kwargs: Optional[Dict[str, Any]] = None,
    refinement_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform iterative optimization and refinement.
    
    Alternates between waypoint optimization and BC refinement for
    progressive improvement.
    
    Args:
        initial_waypoints: Starting waypoints
        env: Environment
        cost_fn: Cost function for optimization
        n_iterations: Number of optimization-refinement iterations
        optimization_kwargs: Arguments for optimization
        refinement_kwargs: Arguments for refinement
        verbose: Whether to print progress
        
    Returns:
        Results from iterative refinement
        
    Example:
        results = iterative_refinement(
            initial_waypoints,
            env,
            trajectory_cost_fn,
            n_iterations=3,
            optimization_kwargs={'budget_iters': 20, 'popsize': 8},
            refinement_kwargs={'n_demonstrations': 10, 'bc_epochs': 5}
        )
    """
    from cogniforge.core.optimization import optimize_waypoints_cma
    
    if optimization_kwargs is None:
        optimization_kwargs = {}
    if refinement_kwargs is None:
        refinement_kwargs = {}
    
    current_waypoints = initial_waypoints
    all_results = []
    best_policy = None
    best_reward = -float('inf')
    
    for iteration in range(n_iterations):
        if verbose:
            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {iteration + 1}/{n_iterations}")
            logger.info(f"{'='*60}")
        
        # Step 1: Optimize waypoints
        if verbose:
            logger.info("Optimizing waypoints...")
        
        opt_result = optimize_waypoints_cma(
            cost_fn,
            current_waypoints,
            verbose=verbose,
            **optimization_kwargs
        )
        
        optimal_waypoints = opt_result['optimal_waypoints']
        
        # Step 2: Refine with BC
        if verbose:
            logger.info("\nRefining with behavior cloning...")
        
        refine_result = refine_with_optimal_trajectory(
            optimal_waypoints,
            env,
            policy=best_policy,  # Use previous best policy
            verbose=verbose,
            **refinement_kwargs
        )
        
        # Check if improved
        mean_reward = refine_result['evaluation_results']['mean_reward']
        if mean_reward > best_reward:
            best_reward = mean_reward
            best_policy = refine_result['refined_policy']
            current_waypoints = optimal_waypoints
            
            if verbose:
                logger.info(f"New best reward: {best_reward:.3f}")
        
        # Store results
        all_results.append({
            'iteration': iteration,
            'optimization': opt_result,
            'refinement': refine_result
        })
    
    return {
        'best_policy': best_policy,
        'best_waypoints': current_waypoints,
        'best_reward': best_reward,
        'all_results': all_results,
        'n_iterations': n_iterations
    }


def post_optimization_pipeline(
    optimization_result: Dict[str, Any],
    env: Any,
    n_expert_demos: int = 20,
    bc_epochs: int = 5,
    bc_lr: float = 5e-4,
    evaluate_both: bool = True,
    record_video: bool = False,
    save_dir: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Complete post-optimization refinement pipeline.
    
    Takes optimization results and performs full refinement with evaluation.
    
    Args:
        optimization_result: Result from waypoint optimization
        env: Environment
        n_expert_demos: Number of expert demonstrations
        bc_epochs: BC training epochs
        bc_lr: BC learning rate
        evaluate_both: Evaluate both expert and refined policy
        record_video: Record evaluation videos
        save_dir: Directory to save artifacts
        verbose: Print progress
        
    Returns:
        Complete pipeline results
        
    Example:
        # After optimization
        opt_result = optimize_waypoints_cma(cost_fn, initial_waypoints)
        
        # Run complete pipeline
        pipeline_result = post_optimization_pipeline(
            opt_result,
            env,
            n_expert_demos=30,
            bc_epochs=5,
            evaluate_both=True,
            record_video=True
        )
    """
    start_time = time.time()
    
    if verbose:
        logger.info("Starting post-optimization refinement pipeline")
    
    # Setup save directory
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
    else:
        save_path = Path("refinement_results")
        save_path.mkdir(exist_ok=True)
    
    # Extract optimal waypoints
    optimal_waypoints = optimization_result['optimal_waypoints']
    
    # Step 1: Create expert and generate demonstrations
    refinement = refine_with_optimal_trajectory(
        optimal_waypoints,
        env,
        n_demonstrations=n_expert_demos,
        bc_epochs=bc_epochs,
        bc_lr=bc_lr,
        save_demos=str(save_path / "demonstrations.pkl") if save_dir else None,
        save_refined_policy=str(save_path / "refined_policy.pt") if save_dir else None,
        verbose=verbose,
        return_all_data=True
    )
    
    results = {
        'refinement': refinement,
        'optimization_result': optimization_result
    }
    
    # Step 2: Detailed evaluation
    if evaluate_both:
        if verbose:
            logger.info("\nEvaluating both expert and refined policy...")
        
        from cogniforge.core.evaluation import rollout_multiple
        
        # Evaluate expert
        expert_eval = rollout_multiple(
            refinement['optimal_expert'],
            env,
            n_rollouts=10,
            max_steps=200,
            record_best=record_video,
            verbose=verbose
        )
        
        # Evaluate refined policy
        policy_eval = rollout_multiple(
            refinement['refined_policy'],
            env,
            n_rollouts=10,
            max_steps=200,
            record_best=record_video,
            verbose=verbose
        )
        
        results['expert_evaluation'] = expert_eval
        results['policy_evaluation'] = policy_eval
        
        if verbose:
            logger.info("\nComparison:")
            logger.info(f"  Expert success rate: {expert_eval['success_rate']:.2%}")
            logger.info(f"  Expert mean reward: {expert_eval['mean_reward']:.3f}")
            logger.info(f"  Policy success rate: {policy_eval['success_rate']:.2%}")
            logger.info(f"  Policy mean reward: {policy_eval['mean_reward']:.3f}")
        
        # Save videos if created
        if record_video and save_dir:
            if 'best_video_path' in expert_eval:
                import shutil
                shutil.copy(expert_eval['best_video_path'], 
                           save_path / "expert_best.gif")
            if 'best_video_path' in policy_eval:
                import shutil
                shutil.copy(policy_eval['best_video_path'],
                           save_path / "policy_best.gif")
    
    # Step 3: Create comparison visualization
    if record_video:
        if verbose:
            logger.info("\nCreating comparison video...")
        
        from cogniforge.core.evaluation import create_comparison_video
        
        try:
            comparison_path = create_comparison_video(
                [refinement['optimal_expert'], refinement['refined_policy']],
                env,
                labels=['Expert (W*)', 'Refined Policy'],
                max_steps=200,
                video_path=str(save_path / "comparison.gif"),
                side_by_side=True
            )
            results['comparison_video'] = comparison_path
        except Exception as e:
            logger.warning(f"Failed to create comparison video: {e}")
    
    # Step 4: Save summary
    if save_dir:
        summary = {
            'optimal_waypoints': optimal_waypoints,
            'optimization_cost': optimization_result.get('optimal_cost'),
            'n_demonstrations': refinement['n_demonstrations'],
            'n_training_samples': refinement['n_training_samples'],
            'bc_final_loss': refinement['bc_results']['final_val_loss'],
            'expert_success_rate': expert_eval['success_rate'] if evaluate_both else None,
            'policy_success_rate': policy_eval['success_rate'] if evaluate_both else None,
            'expert_mean_reward': expert_eval['mean_reward'] if evaluate_both else None,
            'policy_mean_reward': policy_eval['mean_reward'] if evaluate_both else None,
            'total_time': time.time() - start_time
        }
        
        import json
        with open(save_path / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        if verbose:
            logger.info(f"\nResults saved to {save_path}")
    
    results['total_time'] = time.time() - start_time
    
    if verbose:
        logger.info(f"\nPipeline complete in {results['total_time']:.1f} seconds")
    
    return results


# Example usage
if __name__ == "__main__":
    # Example: Complete refinement workflow
    print("=" * 60)
    print("Testing Post-Optimization Refinement")
    print("=" * 60)
    
    # Create dummy environment
    class DummyEnv:
        def reset(self):
            self.step_count = 0
            return np.random.randn(4)
        
        def step(self, action):
            self.step_count += 1
            next_state = np.random.randn(4)
            reward = -np.linalg.norm(action)
            done = self.step_count >= 50
            return next_state, reward, done, {'success': done}
        
        def get_state(self):
            return np.random.randn(4)
    
    # Simulate optimization result
    mock_optimization_result = {
        'optimal_waypoints': [
            (0.0, 0.0, 0.5),
            (0.3, 0.2, 0.4),
            (0.6, 0.5, 0.3),
            (1.0, 1.0, 0.3)
        ],
        'optimal_cost': 1.234,
        'initial_cost': 5.678
    }
    
    env = DummyEnv()
    
    print("\nStep 1: Testing basic refinement")
    print("-" * 40)
    
    refinement_result = refine_with_optimal_trajectory(
        mock_optimization_result['optimal_waypoints'],
        env,
        n_demonstrations=5,
        bc_epochs=2,
        verbose=True
    )
    
    print(f"\nRefinement Results:")
    print(f"  Demonstrations generated: {refinement_result['n_demonstrations']}")
    print(f"  Training samples: {refinement_result['n_training_samples']}")
    print(f"  BC final loss: {refinement_result['bc_results']['final_train_loss']:.6f}")
    print(f"  Evaluation success rate: {refinement_result['evaluation_results']['success_rate']:.2%}")
    
    print("\nStep 2: Testing complete pipeline")
    print("-" * 40)
    
    pipeline_result = post_optimization_pipeline(
        mock_optimization_result,
        env,
        n_expert_demos=5,
        bc_epochs=2,
        evaluate_both=True,
        record_video=False,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)