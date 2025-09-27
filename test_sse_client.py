"""
Test client for Server-Sent Events (SSE) streaming endpoint.

Demonstrates real-time event streaming with phase, message, and metrics.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Optional, Dict, Any
import threading

import requests
import sseclient  # pip install sseclient-py


class SSEClient:
    """Client for consuming Server-Sent Events from CogniForge API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize SSE client."""
        self.base_url = base_url
    
    def stream_events(
        self,
        request_id: str,
        callback: Optional[callable] = None,
        print_events: bool = True
    ):
        """
        Stream events from SSE endpoint.
        
        Args:
            request_id: ID of execution request to monitor
            callback: Optional callback function for each event
            print_events: Whether to print events to console
        """
        url = f"{self.base_url}/events/{request_id}"
        
        print(f"🔌 Connecting to SSE stream: {url}")
        print("=" * 70)
        
        headers = {
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache"
        }
        
        response = requests.get(url, stream=True, headers=headers)
        client = sseclient.SSEClient(response)
        
        for event in client.events():
            if event.data:
                try:
                    data = json.loads(event.data)
                    
                    if print_events:
                        self._print_event(data)
                    
                    if callback:
                        callback(data)
                    
                    # Check for end conditions
                    if data.get("phase") in ["completed", "failed", "stream_end", "timeout"]:
                        print("\n" + "=" * 70)
                        if data.get("phase") == "completed":
                            print("✅ Pipeline completed successfully!")
                        elif data.get("phase") == "failed":
                            print("❌ Pipeline failed!")
                        elif data.get("phase") == "timeout":
                            print("⏱️ Stream timeout!")
                        else:
                            print("📡 Stream ended")
                        break
                        
                except json.JSONDecodeError as e:
                    print(f"Failed to parse event: {e}")
                    print(f"Raw data: {event.data}")
    
    def _print_event(self, data: Dict[str, Any]):
        """Pretty print SSE event."""
        phase = data.get("phase", "unknown")
        message = data.get("message", "")
        progress = data.get("progress", 0)
        metrics = data.get("metrics", {})
        timestamp = data.get("timestamp", "")
        
        # Skip heartbeats in normal output
        if phase == "heartbeat":
            return
        
        # Format progress bar
        progress_pct = int(progress * 100)
        progress_bar = "█" * int(progress * 20)
        empty_bar = "░" * (20 - int(progress * 20))
        
        # Phase emoji mapping
        phase_icons = {
            "connected": "🔌",
            "planning": "📋",
            "expert_demonstration": "👨‍🏫",
            "behavior_cloning": "🧠",
            "optimization": "⚙️",
            "vision_refinement": "👁️",
            "code_generation": "💻",
            "execution": "🤖",
            "completed": "✅",
            "failed": "❌",
            "stream_end": "🏁"
        }
        
        icon = phase_icons.get(phase, "●")
        
        # Print formatted event
        print(f"\n[{timestamp.split('T')[1].split('.')[0] if 'T' in timestamp else timestamp[:8]}]")
        print(f"{icon} Phase: {phase.upper().replace('_', ' ')}")
        print(f"[{progress_bar}{empty_bar}] {progress_pct}%")
        print(f"📝 {message}")
        
        # Print key metrics
        if metrics:
            print("📊 Metrics:")
            for key, value in metrics.items():
                if key != "request_id":  # Skip request_id in metrics
                    # Format different types of values
                    if isinstance(value, float):
                        print(f"   • {key}: {value:.4f}")
                    elif isinstance(value, bool):
                        print(f"   • {key}: {'✓' if value else '✗'}")
                    else:
                        print(f"   • {key}: {value}")
    
    def create_demo_stream(self) -> str:
        """
        Create a demo event stream for testing.
        
        Returns:
            Request ID for the demo stream
        """
        response = requests.post(f"{self.base_url}/events/demo")
        if response.status_code == 200:
            data = response.json()
            print(f"✨ Created demo stream: {data['request_id']}")
            return data["request_id"]
        else:
            print(f"Failed to create demo stream: {response.status_code}")
            return None
    
    def list_active_streams(self):
        """List all active event streams."""
        response = requests.get(f"{self.base_url}/events")
        if response.status_code == 200:
            data = response.json()
            print("\n📊 Active Event Streams")
            print("=" * 70)
            print(f"Total active: {data['total_active']}")
            
            if data['active_streams']:
                for stream in data['active_streams']:
                    print(f"\n• Request ID: {stream['request_id']}")
                    print(f"  Events: {stream['num_events']}")
                    print(f"  Last phase: {stream['last_phase']}")
                    print(f"  Progress: {stream['progress']*100:.0f}%")
                    print(f"  Last update: {stream['last_update']}")
            else:
                print("No active streams")
            
            print("=" * 70)
        else:
            print(f"Failed to list streams: {response.status_code}")


class EventCollector:
    """Collects events for analysis."""
    
    def __init__(self):
        self.events = []
        self.phase_durations = {}
        self.last_phase_time = None
        self.last_phase = None
    
    def collect(self, event: Dict[str, Any]):
        """Collect an event."""
        self.events.append(event)
        
        # Track phase durations
        phase = event.get("phase")
        timestamp = event.get("timestamp")
        
        if phase and timestamp:
            current_time = datetime.fromisoformat(timestamp)
            
            if self.last_phase and self.last_phase != phase:
                # Calculate duration of last phase
                if self.last_phase not in self.phase_durations:
                    self.phase_durations[self.last_phase] = []
                
                duration = (current_time - self.last_phase_time).total_seconds()
                self.phase_durations[self.last_phase].append(duration)
            
            self.last_phase = phase
            self.last_phase_time = current_time
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of collected events."""
        if not self.events:
            return {"error": "No events collected"}
        
        # Calculate metrics
        total_events = len(self.events)
        unique_phases = len(set(e.get("phase") for e in self.events))
        
        # Get all metrics
        all_metrics = {}
        for event in self.events:
            if event.get("metrics"):
                all_metrics.update(event["metrics"])
        
        # Phase progression
        phases = [e.get("phase") for e in self.events if e.get("phase")]
        
        return {
            "total_events": total_events,
            "unique_phases": unique_phases,
            "phases_seen": list(dict.fromkeys(phases)),  # Preserve order, remove duplicates
            "phase_durations": self.phase_durations,
            "collected_metrics": all_metrics,
            "final_progress": self.events[-1].get("progress", 0) if self.events else 0,
            "success": any(e.get("phase") == "completed" for e in self.events)
        }


def test_sse_streaming():
    """Test SSE streaming functionality."""
    client = SSEClient()
    
    print("=" * 70)
    print("SERVER-SENT EVENTS (SSE) TEST")
    print("=" * 70)
    
    # Test 1: Create and stream demo events
    print("\n📝 Test 1: Demo Event Stream")
    print("-" * 40)
    
    demo_id = client.create_demo_stream()
    if demo_id:
        # Stream events
        client.stream_events(demo_id)
    
    # Test 2: Stream with event collection
    print("\n📝 Test 2: Event Collection and Analysis")
    print("-" * 40)
    
    demo_id2 = client.create_demo_stream()
    if demo_id2:
        collector = EventCollector()
        client.stream_events(
            demo_id2,
            callback=collector.collect,
            print_events=False  # Don't print, just collect
        )
        
        # Print summary
        summary = collector.get_summary()
        print("\n📊 Event Stream Summary:")
        print(f"• Total events: {summary['total_events']}")
        print(f"• Unique phases: {summary['unique_phases']}")
        print(f"• Success: {'✅' if summary['success'] else '❌'}")
        print(f"• Final progress: {summary['final_progress']*100:.0f}%")
        
        print("\n📈 Phases progression:")
        for phase in summary['phases_seen']:
            print(f"  → {phase}")
        
        print("\n🔑 Collected metrics:")
        for key, value in summary['collected_metrics'].items():
            if isinstance(value, float):
                print(f"  • {key}: {value:.4f}")
            else:
                print(f"  • {key}: {value}")
    
    # Test 3: List active streams
    print("\n📝 Test 3: List Active Streams")
    print("-" * 40)
    
    client.list_active_streams()
    
    print("\n" + "=" * 70)
    print("All SSE tests completed!")
    print("=" * 70)


def test_concurrent_streams():
    """Test multiple concurrent SSE streams."""
    client = SSEClient()
    
    print("=" * 70)
    print("CONCURRENT SSE STREAMS TEST")
    print("=" * 70)
    
    # Create multiple demo streams
    stream_ids = []
    for i in range(3):
        stream_id = client.create_demo_stream()
        if stream_id:
            stream_ids.append(stream_id)
            print(f"Created stream {i+1}: {stream_id}")
    
    # Monitor all streams concurrently
    threads = []
    collectors = []
    
    for i, stream_id in enumerate(stream_ids):
        collector = EventCollector()
        collectors.append(collector)
        
        def stream_worker(sid, idx, col):
            print(f"\n🎬 Starting stream {idx+1} monitoring...")
            client.stream_events(sid, callback=col.collect, print_events=False)
        
        thread = threading.Thread(
            target=stream_worker,
            args=(stream_id, i, collector)
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all threads
    for thread in threads:
        thread.join()
    
    # Print summaries
    print("\n" + "=" * 70)
    print("CONCURRENT STREAM RESULTS")
    print("=" * 70)
    
    for i, collector in enumerate(collectors):
        summary = collector.get_summary()
        print(f"\nStream {i+1} Summary:")
        print(f"  • Events: {summary['total_events']}")
        print(f"  • Success: {'✅' if summary['success'] else '❌'}")
        print(f"  • Final progress: {summary['final_progress']*100:.0f}%")


def test_real_pipeline_sse():
    """Test SSE with a real pipeline execution."""
    client = SSEClient()
    
    print("=" * 70)
    print("REAL PIPELINE SSE TEST")
    print("=" * 70)
    
    # First, trigger a real pipeline execution
    print("\n🚀 Starting pipeline execution...")
    
    pipeline_request = {
        "task_type": "pick_and_place",
        "task_description": "Pick up the blue cube and place it on the platform",
        "object_name": "blue_cube",
        "use_vision": True,
        "dry_run": True,
        "num_bc_epochs": 3,
        "num_optimization_steps": 10
    }
    
    response = requests.post(
        "http://localhost:8000/execute",
        json=pipeline_request
    )
    
    if response.status_code == 200:
        result = response.json()
        request_id = result["request_id"]
        
        print(f"✅ Pipeline started: {request_id}")
        print("\n📡 Streaming events...")
        print("-" * 70)
        
        # Stream events for the real execution
        collector = EventCollector()
        client.stream_events(
            request_id,
            callback=collector.collect,
            print_events=True
        )
        
        # Print final summary
        summary = collector.get_summary()
        print("\n" + "=" * 70)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 70)
        print(f"Request ID: {request_id}")
        print(f"Success: {'✅' if summary['success'] else '❌'}")
        print(f"Total events: {summary['total_events']}")
        print(f"Final progress: {summary['final_progress']*100:.0f}%")
        
        # Print execution result
        result_response = requests.get(f"http://localhost:8000/result/{request_id}")
        if result_response.status_code == 200:
            exec_result = result_response.json()
            print(f"\nExecution time: {exec_result.get('total_time_seconds', 0):.2f}s")
            
            if exec_result.get("stage_times"):
                print("\nStage timings:")
                for stage, duration in exec_result["stage_times"].items():
                    print(f"  • {stage}: {duration:.2f}s")
    else:
        print(f"❌ Failed to start pipeline: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        
        if test_name == "demo":
            test_sse_streaming()
        elif test_name == "concurrent":
            test_concurrent_streams()
        elif test_name == "real":
            test_real_pipeline_sse()
        else:
            print(f"Unknown test: {test_name}")
            print("Available tests: demo, concurrent, real")
    else:
        # Run all tests
        print("Running SSE tests...")
        print("\n1️⃣ DEMO TEST")
        test_sse_streaming()
        
        print("\n\n2️⃣ CONCURRENT TEST")
        test_concurrent_streams()
        
        print("\n\n3️⃣ REAL PIPELINE TEST")
        print("Note: Make sure the API server is running!")
        print("Start server with: python -m cogniforge.api.execute_endpoint")
        input("\nPress Enter to continue with real pipeline test...")
        test_real_pipeline_sse()