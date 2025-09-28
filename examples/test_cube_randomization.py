#!/usr/bin/env python3
"""
Test script demonstrating cube position randomization at reset().

This script shows how the cube's initial position is randomized by ±2cm
each time the environment is reset, helping to train more robust policies.
"""

import numpy as np
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from cogniforge.environments.randomized_pick_place_env import (
    RandomizedPickPlaceEnv, 
    RandomizedEnvConfig,
    create_randomized_env
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def test_basic_randomization():
    """Test basic cube randomization functionality."""
    print("\n" + "="*60)
    print("TEST 1: Basic Cube Randomization (±2cm)")
    print("="*60)
    
    # Create environment with default ±2cm randomization
    env = create_randomized_env(cube_offset_cm=2.0, seed=42)
    
    # Test multiple resets
    print("\nTesting 5 consecutive resets:")
    print("-" * 40)
    
    positions = []
    offsets = []
    
    for i in range(5):
        obs, info = env.reset(seed=100 + i)  # Different seeds for variation
        
        cube_pos = info["cube_position"]
        cube_offset = info["cube_offset"]
        
        positions.append(cube_pos)
        offsets.append(cube_offset)
        
        print(f"Reset {i+1}:")
        print(f"  Cube position: ({cube_pos[0]:.4f}, {cube_pos[1]:.4f}, {cube_pos[2]:.4f})")
        print(f"  Offset from base: ({cube_offset[0]:.4f}, {cube_offset[1]:.4f}, {cube_offset[2]:.4f})")
        print(f"  Distance from base: {np.linalg.norm(cube_offset[:2]):.4f} m")
    
    # Calculate statistics
    positions = np.array(positions)
    offsets = np.array(offsets)
    
    print("\nStatistics:")
    print(f"  Mean X offset: {offsets[:, 0].mean():.4f} m")
    print(f"  Mean Y offset: {offsets[:, 1].mean():.4f} m")
    print(f"  Std X offset: {offsets[:, 0].std():.4f} m")
    print(f"  Std Y offset: {offsets[:, 1].std():.4f} m")
    print(f"  Max offset magnitude: {np.max(np.linalg.norm(offsets[:, :2], axis=1)):.4f} m")
    
    # Verify all offsets are within ±2cm
    assert np.all(np.abs(offsets[:, 0]) <= 0.02), "X offsets exceed ±2cm!"
    assert np.all(np.abs(offsets[:, 1]) <= 0.02), "Y offsets exceed ±2cm!"
    assert np.all(offsets[:, 2] == 0.0), "Z should not be randomized by default!"
    
    print("\n✅ Basic randomization test PASSED!")
    
    env.close()


def test_custom_randomization():
    """Test custom randomization ranges."""
    print("\n" + "="*60)
    print("TEST 2: Custom Randomization Ranges")
    print("="*60)
    
    # Test different randomization ranges
    test_ranges = [1.0, 3.0, 5.0]  # cm
    
    for range_cm in test_ranges:
        print(f"\nTesting ±{range_cm}cm randomization:")
        print("-" * 40)
        
        # Create environment with custom range
        config = RandomizedEnvConfig(
            cube_offset_range=range_cm / 100.0,  # Convert to meters
            randomize_xy_only=True
        )
        env = RandomizedPickPlaceEnv(config=config)
        
        # Collect offsets
        offsets = []
        for i in range(10):
            obs, info = env.reset(seed=i)
            offsets.append(info["cube_offset"])
        
        offsets = np.array(offsets)
        
        # Check bounds
        max_x = np.max(np.abs(offsets[:, 0]))
        max_y = np.max(np.abs(offsets[:, 1]))
        
        print(f"  Max X offset: {max_x*100:.2f} cm")
        print(f"  Max Y offset: {max_y*100:.2f} cm")
        
        # Verify within bounds
        assert max_x <= range_cm / 100.0, f"X offset exceeds ±{range_cm}cm!"
        assert max_y <= range_cm / 100.0, f"Y offset exceeds ±{range_cm}cm!"
        
        print(f"  ✓ Offsets within ±{range_cm}cm range")
        
        env.close()
    
    print("\n✅ Custom randomization test PASSED!")


def test_xyz_randomization():
    """Test randomization with Z-axis included."""
    print("\n" + "="*60)
    print("TEST 3: XYZ Randomization (including Z-axis)")
    print("="*60)
    
    # Create environment with XYZ randomization
    config = RandomizedEnvConfig(
        cube_offset_range=0.02,  # ±2cm
        randomize_xy_only=False  # Enable Z randomization
    )
    env = RandomizedPickPlaceEnv(config=config)
    
    print("\nTesting XYZ randomization (5 resets):")
    print("-" * 40)
    
    z_offsets = []
    
    for i in range(5):
        obs, info = env.reset(seed=200 + i)
        
        cube_pos = info["cube_position"]
        cube_offset = info["cube_offset"]
        
        z_offsets.append(cube_offset[2])
        
        print(f"Reset {i+1}:")
        print(f"  Position: ({cube_pos[0]:.4f}, {cube_pos[1]:.4f}, {cube_pos[2]:.4f})")
        print(f"  Z offset: {cube_offset[2]:.4f} m ({cube_offset[2]*100:.2f} cm)")
    
    # Check that Z varies
    z_offsets = np.array(z_offsets)
    z_variation = np.std(z_offsets)
    
    print(f"\nZ-axis statistics:")
    print(f"  Mean Z offset: {z_offsets.mean():.4f} m")
    print(f"  Std Z offset: {z_variation:.4f} m")
    print(f"  Range: [{z_offsets.min():.4f}, {z_offsets.max():.4f}] m")
    
    # Verify Z randomization is active
    assert z_variation > 0, "Z-axis should have variation when randomize_xy_only=False!"
    assert np.all(np.abs(z_offsets) <= 0.01), "Z offsets should be within ±1cm (half of XY range)!"
    
    print("\n✅ XYZ randomization test PASSED!")
    
    env.close()


def test_deterministic_reset():
    """Test that same seed produces same randomization."""
    print("\n" + "="*60)
    print("TEST 4: Deterministic Reset with Seeds")
    print("="*60)
    
    env = create_randomized_env(cube_offset_cm=2.0)
    
    print("\nTesting deterministic resets with same seed:")
    print("-" * 40)
    
    # Reset with seed 123 multiple times
    positions_1 = []
    positions_2 = []
    
    for _ in range(3):
        obs, info = env.reset(seed=123)
        positions_1.append(info["cube_position"])
    
    for _ in range(3):
        obs, info = env.reset(seed=123)
        positions_2.append(info["cube_position"])
    
    # Check all positions are identical for same seed
    for i in range(3):
        pos1 = positions_1[i]
        pos2 = positions_2[i]
        
        print(f"Reset {i+1} with seed=123:")
        print(f"  First run:  ({pos1[0]:.4f}, {pos1[1]:.4f}, {pos1[2]:.4f})")
        print(f"  Second run: ({pos2[0]:.4f}, {pos2[1]:.4f}, {pos2[2]:.4f})")
        
        # Verify positions match
        assert np.allclose(pos1, pos2), "Same seed should produce same position!"
        print("  ✓ Positions match!")
    
    print("\n✅ Deterministic reset test PASSED!")
    
    env.close()


def test_visual_distribution():
    """Generate visual representation of randomization distribution."""
    print("\n" + "="*60)
    print("TEST 5: Visual Distribution Analysis")
    print("="*60)
    
    env = create_randomized_env(cube_offset_cm=2.0)
    
    # Collect many samples
    n_samples = 100
    positions = []
    
    print(f"\nCollecting {n_samples} randomized positions...")
    
    for i in range(n_samples):
        obs, info = env.reset(seed=1000 + i)
        positions.append(info["cube_position"][:2])  # Just X,Y
    
    positions = np.array(positions)
    
    # Calculate distribution statistics
    mean_pos = positions.mean(axis=0)
    std_pos = positions.std(axis=0)
    
    print("\nDistribution Analysis:")
    print("-" * 40)
    print(f"Base position: (0.5, 0.0)")
    print(f"Mean position: ({mean_pos[0]:.4f}, {mean_pos[1]:.4f})")
    print(f"Std deviation: ({std_pos[0]:.4f}, {std_pos[1]:.4f})")
    
    # Check coverage of the randomization area
    x_range = positions[:, 0].max() - positions[:, 0].min()
    y_range = positions[:, 1].max() - positions[:, 1].min()
    
    print(f"X range covered: {x_range*100:.2f} cm")
    print(f"Y range covered: {y_range*100:.2f} cm")
    
    # Verify good coverage (should be close to 4cm for ±2cm range)
    assert x_range > 0.035, "X range coverage too small!"
    assert y_range > 0.035, "Y range coverage too small!"
    
    print(f"\n✓ Good distribution coverage achieved")
    
    # Create ASCII visualization
    print("\nASCII Visualization of Spawn Positions:")
    print("-" * 40)
    
    # Create grid
    grid_size = 21  # 21x21 grid for ±2cm range
    grid = np.zeros((grid_size, grid_size), dtype=int)
    
    # Map positions to grid
    for x, y in positions:
        # Convert to grid coordinates
        gx = int((x - 0.48) / 0.04 * (grid_size - 1))
        gy = int((y + 0.02) / 0.04 * (grid_size - 1))
        
        # Clip to grid bounds
        gx = np.clip(gx, 0, grid_size - 1)
        gy = np.clip(gy, 0, grid_size - 1)
        
        grid[gy, gx] += 1
    
    # Print grid (rotated for correct orientation)
    for row in grid:
        line = ""
        for count in row:
            if count == 0:
                line += "·"
            elif count == 1:
                line += "○"
            elif count < 3:
                line += "●"
            else:
                line += "■"
        print(f"  {line}")
    
    print("\nLegend: · = empty, ○ = 1 spawn, ● = 2 spawns, ■ = 3+ spawns")
    print("Grid represents ±2cm area around base position (0.5, 0.0)")
    
    print("\n✅ Visual distribution test PASSED!")
    
    env.close()


def main():
    """Run all randomization tests."""
    print("\n" + "#"*60)
    print("# CUBE RANDOMIZATION TEST SUITE")
    print("#"*60)
    print("\nThis test suite verifies that cube spawning is properly")
    print("randomized by ±2cm (or custom range) at each reset().")
    
    try:
        # Run all tests
        test_basic_randomization()
        test_custom_randomization()
        test_xyz_randomization()
        test_deterministic_reset()
        test_visual_distribution()
        
        print("\n" + "#"*60)
        print("# ALL TESTS PASSED! 🎉")
        print("#"*60)
        print("\nSummary:")
        print("  ✅ Basic ±2cm randomization works correctly")
        print("  ✅ Custom randomization ranges supported")
        print("  ✅ XYZ randomization option available")
        print("  ✅ Deterministic reset with seeds works")
        print("  ✅ Good distribution coverage achieved")
        print("\nThe cube initial position randomization is fully functional!")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())