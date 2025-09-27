"""
Test client for the CogniForge FastAPI execution endpoint.

Tests the complete pipeline: plan → expert → BC → optimize → vision → codegen
"""

import asyncio
import json
import time
from typing import Optional, Dict, Any

import aiohttp
import requests


class CogniForgeClient:
    """Client for interacting with CogniForge API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize client with API base URL."""
        self.base_url = base_url
    
    def execute_sync(
        self,
        task_description: str,
        object_name: Optional[str] = None,
        use_vision: bool = True,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Execute task synchronously and wait for result.
        
        Args:
            task_description: Natural language task description
            object_name: Name of object to manipulate
            use_vision: Whether to use vision refinement
            dry_run: Whether to simulate execution
            
        Returns:
            Execution result
        """
        # Prepare request
        request_data = {
            "task_type": "pick_and_place",
            "task_description": task_description,
            "use_vision": use_vision,
            "dry_run": dry_run,
            "num_bc_epochs": 5,  # Reduced for testing
            "num_optimization_steps": 20  # Reduced for testing
        }
        
        if object_name:
            request_data["object_name"] = object_name
        
        # Send request
        print(f"Sending execution request to {self.base_url}/execute")
        print(f"Task: {task_description}")
        print("-" * 60)
        
        response = requests.post(
            f"{self.base_url}/execute",
            json=request_data
        )
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    
    async def execute_async(
        self,
        task_description: str,
        object_name: Optional[str] = None,
        use_vision: bool = True,
        dry_run: bool = True,
        monitor_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Execute task asynchronously with progress monitoring.
        
        Args:
            task_description: Natural language task description
            object_name: Name of object to manipulate  
            use_vision: Whether to use vision refinement
            dry_run: Whether to simulate execution
            monitor_progress: Whether to poll for progress updates
            
        Returns:
            Execution result
        """
        request_data = {
            "task_type": "pick_and_place",
            "task_description": task_description,
            "use_vision": use_vision,
            "dry_run": dry_run,
            "num_bc_epochs": 5,
            "num_optimization_steps": 20
        }
        
        if object_name:
            request_data["object_name"] = object_name
        
        async with aiohttp.ClientSession() as session:
            # Send request
            print(f"Sending async execution request...")
            print(f"Task: {task_description}")
            print("-" * 60)
            
            async with session.post(
                f"{self.base_url}/execute",
                json=request_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    if monitor_progress and "request_id" in result:
                        # Poll for status updates
                        request_id = result["request_id"]
                        await self._monitor_progress(session, request_id)
                    
                    return result
                else:
                    print(f"Error: {response.status}")
                    text = await response.text()
                    print(text)
                    return None
    
    async def _monitor_progress(
        self,
        session: aiohttp.ClientSession,
        request_id: str
    ):
        """Monitor execution progress."""
        print(f"\nMonitoring progress for request: {request_id}")
        print("=" * 60)
        
        last_stage = None
        while True:
            async with session.get(
                f"{self.base_url}/status/{request_id}"
            ) as response:
                if response.status == 200:
                    status = await response.json()
                    
                    # Print stage changes
                    if status["stage"] != last_stage:
                        progress_bar = "█" * int(status["progress"] * 20)
                        empty_bar = "░" * (20 - int(status["progress"] * 20))
                        
                        print(f"\n[{progress_bar}{empty_bar}] {status['progress']*100:.0f}%")
                        print(f"Stage: {status['stage']}")
                        print(f"Status: {status['message']}")
                        
                        if status.get("details"):
                            print(f"Details: {json.dumps(status['details'], indent=2)}")
                        
                        last_stage = status["stage"]
                    
                    # Check if completed
                    if status["stage"] in ["completed", "failed"]:
                        print("\n" + "=" * 60)
                        if status["stage"] == "completed":
                            print("✅ Execution completed successfully!")
                        else:
                            print("❌ Execution failed!")
                            if status.get("errors"):
                                print("Errors:", status["errors"])
                        break
                
                else:
                    print(f"Failed to get status: {response.status}")
                    break
            
            await asyncio.sleep(0.5)
    
    async def execute_with_websocket(
        self,
        task_description: str,
        object_name: Optional[str] = None,
        use_vision: bool = True,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Execute task with WebSocket monitoring for real-time updates.
        """
        import websockets
        
        # First send the execution request
        request_data = {
            "task_type": "pick_and_place",
            "task_description": task_description,
            "use_vision": use_vision,
            "dry_run": dry_run,
            "num_bc_epochs": 5,
            "num_optimization_steps": 20
        }
        
        if object_name:
            request_data["object_name"] = object_name
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/execute",
                json=request_data
            ) as response:
                if response.status != 200:
                    print(f"Failed to start execution: {response.status}")
                    return None
                
                result = await response.json()
                request_id = result["request_id"]
                
                print(f"Execution started: {request_id}")
                print("Connecting to WebSocket for real-time updates...")
                print("=" * 60)
                
                # Connect WebSocket for real-time updates
                ws_url = f"ws://localhost:8000/ws/{request_id}"
                
                try:
                    async with websockets.connect(ws_url) as websocket:
                        while True:
                            message = await websocket.recv()
                            data = json.loads(message)
                            
                            if data["type"] == "status_update":
                                status = data["data"]
                                progress_bar = "█" * int(status["progress"] * 20)
                                empty_bar = "░" * (20 - int(status["progress"] * 20))
                                
                                print(f"\r[{progress_bar}{empty_bar}] {status['progress']*100:.0f}% - {status['stage']}: {status['message']}", end="")
                                
                            elif data["type"] == "execution_complete":
                                print("\n" + "=" * 60)
                                if data["data"]["success"]:
                                    print("✅ Execution completed successfully!")
                                else:
                                    print("❌ Execution failed!")
                                break
                
                except Exception as e:
                    print(f"WebSocket error: {e}")
                
                return result


def print_result(result: Dict[str, Any]):
    """Pretty print execution result."""
    if not result:
        return
    
    print("\n" + "=" * 70)
    print("EXECUTION RESULT")
    print("=" * 70)
    
    print(f"Request ID: {result.get('request_id', 'N/A')}")
    print(f"Success: {'✅' if result.get('success') else '❌'}")
    print(f"Total Time: {result.get('total_time_seconds', 0):.2f} seconds")
    
    # Stage timings
    if "stage_times" in result:
        print("\nStage Timings:")
        for stage, duration in result["stage_times"].items():
            print(f"  • {stage}: {duration:.2f}s")
    
    # Stages completed
    if "stages_completed" in result:
        print(f"\nStages Completed: {len(result['stages_completed'])}")
        for stage in result["stages_completed"]:
            print(f"  ✓ {stage}")
    
    # Key metrics
    print("\nKey Metrics:")
    if "expert_trajectories" in result:
        print(f"  • Expert Trajectories: {result['expert_trajectories']}")
    if "bc_loss" in result:
        print(f"  • BC Loss: {result['bc_loss']:.4f}")
    if "optimization_reward" in result:
        print(f"  • Optimization Reward: {result['optimization_reward']:.3f}")
    
    # Vision alignment
    if "vision_alignment" in result and result["vision_alignment"]:
        vision = result["vision_alignment"]
        alignment = vision.get("alignment", {})
        print(f"\nVision Alignment:")
        print(f"  • Status: {alignment.get('status', 'N/A')}")
        print(f"  • Aligned: {'✅' if alignment.get('is_aligned') else '❌'}")
        print(f"  • Error: {alignment.get('error_mm', 0):.1f}mm")
    
    # Generated code preview
    if "generated_code" in result and result["generated_code"]:
        code_lines = result["generated_code"].split("\n")[:10]
        print(f"\nGenerated Code (first 10 lines):")
        for line in code_lines:
            print(f"  {line}")
    
    # Execution result
    if "execution_result" in result and result["execution_result"]:
        exec_res = result["execution_result"]
        print(f"\nExecution Result:")
        print(f"  • Success: {'✅' if exec_res.get('success') else '❌'}")
        print(f"  • Execution Time: {exec_res.get('execution_time', 0):.2f}s")
        if "final_position" in exec_res:
            pos = exec_res["final_position"]
            print(f"  • Final Position: ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})")
        print(f"  • Gripper State: {exec_res.get('gripper_state', 'unknown')}")
        print(f"  • Object Grasped: {'✅' if exec_res.get('object_grasped') else '❌'}")
    
    print("=" * 70)


async def main():
    """Main test function."""
    client = CogniForgeClient()
    
    # Test cases
    test_cases = [
        {
            "description": "Pick up the blue cube and place it on the red platform",
            "object_name": "blue_cube",
            "use_vision": True
        },
        {
            "description": "Stack the green block on top of the yellow block",
            "object_name": "green_block",
            "use_vision": True
        },
        {
            "description": "Move the metal cylinder into the designated hole",
            "object_name": "cylinder",
            "use_vision": False
        }
    ]
    
    print("=" * 70)
    print("COGNIFORGE API TEST CLIENT")
    print("=" * 70)
    
    # Test 1: Synchronous execution
    print("\n📝 Test 1: Synchronous Execution")
    print("-" * 40)
    
    result = client.execute_sync(
        task_description=test_cases[0]["description"],
        object_name=test_cases[0]["object_name"],
        use_vision=test_cases[0]["use_vision"],
        dry_run=True
    )
    
    print_result(result)
    
    # Test 2: Asynchronous execution with progress monitoring
    print("\n📝 Test 2: Asynchronous Execution with Progress Monitoring")
    print("-" * 40)
    
    result = await client.execute_async(
        task_description=test_cases[1]["description"],
        object_name=test_cases[1]["object_name"],
        use_vision=test_cases[1]["use_vision"],
        dry_run=True,
        monitor_progress=True
    )
    
    print_result(result)
    
    # Test 3: WebSocket real-time monitoring
    print("\n📝 Test 3: WebSocket Real-time Monitoring")
    print("-" * 40)
    
    try:
        result = await client.execute_with_websocket(
            task_description=test_cases[2]["description"],
            object_name=test_cases[2]["object_name"],
            use_vision=test_cases[2]["use_vision"],
            dry_run=True
        )
        
        print_result(result)
    except Exception as e:
        print(f"WebSocket test failed: {e}")
        print("Make sure websockets library is installed: pip install websockets")
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    # Run async main
    asyncio.run(main())