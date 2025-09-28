"""
Test script for /healthz endpoint
Demonstrates comprehensive health monitoring capabilities
"""

import requests
import json
import time
import pybullet as p
from reset_system import ResetSystem
import threading

def pretty_print_health(health_data):
    """Pretty print health check data"""
    print("\n" + "="*80)
    print(" SYSTEM HEALTH CHECK REPORT")
    print("="*80)
    
    # Overall Status
    status = health_data.get('status', 'unknown')
    status_emoji = "✅" if status == 'healthy' else "⚠️" if status == 'degraded' else "❌"
    print(f"\n{status_emoji} Overall Status: {status.upper()}")
    
    if 'uptime' in health_data and health_data['uptime']:
        uptime_minutes = health_data['uptime'] / 60
        print(f"⏱️  Uptime: {uptime_minutes:.1f} minutes")
    
    # Warnings
    if 'warnings' in health_data:
        print("\n⚠️  Warnings:")
        for warning in health_data['warnings']:
            print(f"   - {warning}")
    
    # PyBullet Status
    print("\n" + "-"*40)
    print(" PyBullet Status")
    print("-"*40)
    pb_status = health_data.get('pybullet', {})
    if pb_status.get('connected'):
        print(f"✅ Connected via {pb_status.get('connection_method', 'N/A')}")
        print(f"   Bodies: {pb_status.get('num_bodies', 0)}")
        print(f"   Robot Joints: {pb_status.get('num_joints', 0)}")
        print(f"   Gravity: {pb_status.get('gravity', 'N/A')}")
        print(f"   Time Step: {pb_status.get('time_step', 'N/A')}")
        print(f"   Real-Time Sim: {pb_status.get('real_time_simulation', False)}")
    else:
        print(f"❌ Not Connected - {pb_status.get('error', 'Unknown error')}")
    
    # Model Status
    print("\n" + "-"*40)
    print(" Model Status")
    print("-"*40)
    models = health_data.get('models', {})
    bc_model = models.get('bc', {})
    if bc_model.get('loaded'):
        print(f"✅ BC Model Loaded")
        if bc_model.get('parameters'):
            print(f"   Parameters: {bc_model['parameters']:,}")
        if bc_model.get('device'):
            print(f"   Device: {bc_model['device']}")
        if bc_model.get('type'):
            print(f"   Type: {bc_model['type']}")
    else:
        print("⚠️  BC Model Not Loaded")
    
    # Device Information
    print("\n" + "-"*40)
    print(" Device Information")
    print("-"*40)
    device = health_data.get('device', {})
    print(f"Platform: {device.get('platform', 'N/A')} {device.get('platform_release', '')}")
    print(f"Architecture: {device.get('architecture', 'N/A')}")
    print(f"Python: {device.get('python_version', 'N/A')}")
    print(f"CPU: {device.get('cpu_count', 0)} cores @ {device.get('cpu_percent', 0):.1f}% usage")
    
    memory = device.get('memory', {})
    if memory:
        total_gb = memory.get('total', 0) / (1024**3)
        used_gb = memory.get('used', 0) / (1024**3)
        percent = memory.get('percent', 0)
        print(f"Memory: {used_gb:.1f}/{total_gb:.1f} GB ({percent:.1f}%)")
    
    # GPU Information
    if device.get('cuda_available'):
        print("\nGPU Information:")
        cuda_info = device.get('cuda', {})
        print(f"   Device: {cuda_info.get('device_name', 'N/A')}")
        print(f"   Capability: {cuda_info.get('capability', 'N/A')}")
        mem_alloc = cuda_info.get('memory_allocated', 0) / (1024**2)
        mem_reserved = cuda_info.get('memory_reserved', 0) / (1024**2)
        print(f"   Memory: {mem_alloc:.1f}/{mem_reserved:.1f} MB")
    else:
        print("GPU: Not Available")
    
    # System Resources
    print("\n" + "-"*40)
    print(" System Resources")
    print("-"*40)
    system = health_data.get('system', {})
    if not system.get('error'):
        print(f"Process ID: {system.get('process_id', 'N/A')}")
        print(f"Process Memory: {system.get('process_memory_mb', 0):.1f} MB")
        print(f"Process CPU: {system.get('process_cpu_percent', 0):.1f}%")
        print(f"Active Threads: {system.get('threads', 0)}")
        
        disk = system.get('disk_usage', {})
        if disk:
            disk_free_gb = disk.get('free', 0) / (1024**3)
            disk_total_gb = disk.get('total', 0) / (1024**3)
            disk_percent = disk.get('percent', 0)
            print(f"Disk: {disk_free_gb:.1f} GB free / {disk_total_gb:.1f} GB total ({disk_percent:.1f}% used)")
    
    # Scene Status
    print("\n" + "-"*40)
    print(" Scene Status")
    print("-"*40)
    scene = health_data.get('scene', {})
    print(f"Objects: {scene.get('objects_count', 0)}")
    print(f"Robot Present: {'Yes' if scene.get('robot_id') else 'No'}")
    print(f"Current Seed: {scene.get('current_seed', 'N/A')}")
    print(f"Reset Count: {scene.get('reset_count', 0)}")
    if scene.get('last_reset_time'):
        print(f"Last Reset Time: {scene['last_reset_time']:.3f}s")
    if scene.get('scene_objects'):
        print(f"Scene Objects: {', '.join(scene['scene_objects'])}")
    
    # Benchmark Status
    print("\n" + "-"*40)
    print(" Benchmark Status")
    print("-"*40)
    benchmark = health_data.get('benchmark', {})
    if benchmark.get('is_running'):
        print("🔄 Benchmark Currently Running")
    else:
        print("✅ Benchmark Idle")
    print(f"Total Benchmark Runs: {benchmark.get('total_runs', 0)}")
    print(f"Has Latest Results: {'Yes' if benchmark.get('last_run') else 'No'}")
    
    # API Endpoints
    print("\n" + "-"*40)
    print(" API Server")
    print("-"*40)
    api = health_data.get('api', {})
    print(f"Server Running: {'Yes' if api.get('running') else 'No'}")
    print(f"Address: http://{api.get('host', 'localhost')}:{api.get('port', 5000)}")
    
    endpoints = api.get('endpoints_available', [])
    if endpoints:
        print(f"\nAvailable Endpoints ({len(endpoints)}):")
        # Group by method
        get_endpoints = [e for e in endpoints if 'GET' in e.get('methods', [])]
        post_endpoints = [e for e in endpoints if 'POST' in e.get('methods', [])]
        
        if get_endpoints:
            print("  GET:")
            for endpoint in get_endpoints[:5]:  # Show first 5
                print(f"    - {endpoint['path']}")
            if len(get_endpoints) > 5:
                print(f"    ... and {len(get_endpoints) - 5} more")
                
        if post_endpoints:
            print("  POST:")
            for endpoint in post_endpoints[:5]:  # Show first 5
                print(f"    - {endpoint['path']}")
            if len(post_endpoints) > 5:
                print(f"    ... and {len(post_endpoints) - 5} more")
    
    print("\n" + "="*80)


def test_healthz_api():
    """Test the /healthz endpoint via API"""
    
    print("\n🔍 Testing /healthz Endpoint...")
    
    # Start PyBullet and reset system
    print("\n1. Starting PyBullet simulation...")
    client = p.connect(p.DIRECT)  # Use DIRECT for headless testing
    p.setGravity(0, 0, -9.81)
    
    print("2. Initializing reset system...")
    reset_system = ResetSystem(client)
    reset_system.initialize(enable_hotkeys=False, enable_api=True, api_port=5001)
    
    # Give the server time to start
    time.sleep(2)
    
    print("3. Checking health endpoint...")
    try:
        # Test regular health endpoint
        response = requests.get("http://localhost:5001/health")
        print(f"\n/health response: {response.json()}")
        
        # Test comprehensive healthz endpoint
        response = requests.get("http://localhost:5001/healthz")
        
        if response.status_code == 200:
            health_data = response.json()
            
            # Pretty print the health data
            pretty_print_health(health_data)
            
            # Save to file
            with open("health_report.json", "w") as f:
                json.dump(health_data, f, indent=2)
            print("\n📄 Full health report saved to: health_report.json")
            
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server")
    except Exception as e:
        print(f"❌ Error during health check: {e}")
    
    finally:
        # Cleanup
        print("\n4. Cleaning up...")
        reset_system.shutdown()
        p.disconnect()
        print("✅ Test complete!")


def test_health_monitoring():
    """Test health monitoring during operation"""
    
    print("\n🔍 Testing Health Monitoring During Operation...")
    
    # Start PyBullet GUI
    print("\n1. Starting PyBullet GUI simulation...")
    client = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    
    print("2. Initializing reset system...")
    reset_system = ResetSystem(client)
    reset_system.initialize(enable_hotkeys=True, enable_api=True, api_port=5002)
    
    # Give the server time to start
    time.sleep(2)
    
    def monitor_health():
        """Continuously monitor health"""
        print("\n3. Starting health monitoring (press Ctrl+C to stop)...")
        print("   Monitoring every 5 seconds...\n")
        
        try:
            while True:
                response = requests.get("http://localhost:5002/healthz")
                if response.status_code == 200:
                    health = response.json()
                    
                    # Print compact status
                    status_icon = "✅" if health['status'] == 'healthy' else "⚠️"
                    pb_icon = "🔗" if health['pybullet']['connected'] else "❌"
                    model_icon = "🤖" if health['models']['bc']['loaded'] else "⚪"
                    
                    cpu = health['device']['cpu_percent']
                    mem = health['device']['memory']['percent']
                    objects = health['scene']['objects_count']
                    resets = health['scene']['reset_count']
                    
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"[{timestamp}] {status_icon} Status | {pb_icon} PyBullet | {model_icon} Model | "
                          f"CPU: {cpu:.1f}% | MEM: {mem:.1f}% | Objects: {objects} | Resets: {resets}")
                    
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\n\n✋ Monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ Monitoring error: {e}")
    
    # Start monitoring in a thread
    monitor_thread = threading.Thread(target=monitor_health, daemon=True)
    monitor_thread.start()
    
    print("\n" + "="*60)
    print(" SIMULATION RUNNING")
    print("="*60)
    print("Press 'R' to soft reset")
    print("Press 'S' to reseed")
    print("Press Ctrl+C to exit")
    print("="*60)
    
    try:
        # Keep simulation running
        while True:
            p.stepSimulation()
            time.sleep(1/240)
            
    except KeyboardInterrupt:
        print("\n\n4. Shutting down...")
        
    finally:
        reset_system.shutdown()
        p.disconnect()
        print("✅ Test complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test /healthz endpoint")
    parser.add_argument('--mode', choices=['api', 'monitor', 'both'], 
                       default='api', help='Test mode')
    
    args = parser.parse_args()
    
    if args.mode == 'api' or args.mode == 'both':
        test_healthz_api()
        
    if args.mode == 'monitor' or args.mode == 'both':
        test_health_monitoring()
        
    if args.mode == 'both':
        print("\n" + "="*80)
        print(" ALL TESTS COMPLETE")
        print("="*80)