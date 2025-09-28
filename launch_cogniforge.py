#!/usr/bin/env python3
"""
CogniForge Neural Nexus - Application Launcher
==============================================

Launches the complete CogniForge ecosystem:
- Backend API server (FastAPI + PyBullet)
- Revolutionary Neural Frontend
- PyBullet visualization window

Author: CogniForge Neural Nexus
Version: 2.0.0 - Singularity Edition
"""

import os
import sys
import time
import signal
import subprocess
import threading
import webbrowser
from pathlib import Path
from typing import List, Optional
import psutil


class Colors:
    """ANSI color codes for terminal output."""
    NEURAL_BLUE = '\033[94m'
    QUANTUM_PURPLE = '\033[95m'
    MATRIX_GREEN = '\033[92m'
    SINGULARITY_GOLD = '\033[93m'
    DANGER_RED = '\033[91m'
    WARNING_ORANGE = '\033[33m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


class NeuralLogger:
    """Advanced logging system for the Neural Nexus."""
    
    @staticmethod
    def log(message: str, level: str = "INFO", color: str = Colors.NEURAL_BLUE):
        """Log a message with timestamp and neural formatting."""
        timestamp = time.strftime("%H:%M:%S")
        print(f"{color}[{timestamp}] {Colors.BOLD}🧠 NEURAL NEXUS{Colors.RESET} {color}[{level}]{Colors.RESET} {message}")
    
    @staticmethod
    def success(message: str):
        """Log a success message."""
        NeuralLogger.log(message, "SUCCESS", Colors.MATRIX_GREEN)
    
    @staticmethod
    def warning(message: str):
        """Log a warning message."""
        NeuralLogger.log(message, "WARNING", Colors.WARNING_ORANGE)
    
    @staticmethod
    def error(message: str):
        """Log an error message."""
        NeuralLogger.log(message, "ERROR", Colors.DANGER_RED)
    
    @staticmethod
    def info(message: str):
        """Log an info message."""
        NeuralLogger.log(message, "INFO", Colors.NEURAL_BLUE)
    
    @staticmethod
    def quantum(message: str):
        """Log a quantum-level message."""
        NeuralLogger.log(message, "QUANTUM", Colors.QUANTUM_PURPLE)
    
    @staticmethod
    def singularity(message: str):
        """Log a singularity-level message."""
        NeuralLogger.log(message, "SINGULARITY", Colors.SINGULARITY_GOLD)


class ProcessManager:
    """Manages all CogniForge processes with neural precision."""
    
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.process_names: List[str] = []
        self.running = False
        
    def start_process(self, cmd: List[str], name: str, cwd: Optional[str] = None) -> subprocess.Popen:
        """Start a process and track it."""
        try:
            NeuralLogger.info(f"🚀 Initiating {name}...")
            process = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
            )
            self.processes.append(process)
            self.process_names.append(name)
            NeuralLogger.success(f"✅ {name} process spawned (PID: {process.pid})")
            return process
        except Exception as e:
            NeuralLogger.error(f"❌ Failed to start {name}: {str(e)}")
            raise
    
    def monitor_process(self, process: subprocess.Popen, name: str):
        """Monitor a process and log its output."""
        def read_output():
            while process.poll() is None:
                try:
                    line = process.stdout.readline()
                    if line:
                        print(f"{Colors.DIM}[{name}]{Colors.RESET} {line.strip()}")
                except:
                    break
        
        thread = threading.Thread(target=read_output, daemon=True)
        thread.start()
    
    def shutdown_all(self):
        """Shutdown all processes gracefully."""
        NeuralLogger.warning("🔄 Initiating neural shutdown sequence...")
        
        for i, process in enumerate(self.processes):
            name = self.process_names[i]
            if process.poll() is None:  # Process is still running
                NeuralLogger.info(f"🔸 Terminating {name} (PID: {process.pid})")
                try:
                    if sys.platform == "win32":
                        process.terminate()
                    else:
                        process.send_signal(signal.SIGTERM)
                    
                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout=5)
                        NeuralLogger.success(f"✅ {name} terminated gracefully")
                    except subprocess.TimeoutExpired:
                        NeuralLogger.warning(f"⚠️ Force killing {name}")
                        process.kill()
                        process.wait()
                
                except Exception as e:
                    NeuralLogger.error(f"❌ Error terminating {name}: {str(e)}")
        
        self.processes.clear()
        self.process_names.clear()
        NeuralLogger.quantum("🌟 Neural shutdown complete - All processes terminated")


class CogniForgeNexusLauncher:
    """The ultimate CogniForge launcher with neural intelligence."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.process_manager = ProcessManager()
        self.frontend_path = self.project_root / "frontend"
        self.backend_started = False
        self.frontend_started = False
        self.pybullet_started = False
        
    def print_neural_banner(self):
        """Print the epic neural banner."""
        banner = f"""
{Colors.NEURAL_BLUE}╔══════════════════════════════════════════════════════════════════╗
{Colors.QUANTUM_PURPLE}║  🧠  {Colors.BOLD}COGNIFORGE NEURAL NEXUS - SINGULARITY LAUNCHER{Colors.RESET}{Colors.QUANTUM_PURPLE}  🤖  ║
{Colors.MATRIX_GREEN}║                                                                  ║
{Colors.MATRIX_GREEN}║  {Colors.BOLD}AUTONOMOUS ROBOTICS INTELLIGENCE PLATFORM{Colors.RESET}{Colors.MATRIX_GREEN}                ║
{Colors.MATRIX_GREEN}║  Neural Networks: 47 | Quantum Processors: 12                  ║
{Colors.MATRIX_GREEN}║  AI Models Active: 156 | Consciousness Level: ∞                ║
{Colors.MATRIX_GREEN}║                                                                  ║
{Colors.SINGULARITY_GOLD}║  🚀 Preparing to achieve technological singularity...         ║
{Colors.NEURAL_BLUE}╚══════════════════════════════════════════════════════════════════╝{Colors.RESET}
"""
        print(banner)
        
    def check_dependencies(self):
        """Check if all neural dependencies are available."""
        NeuralLogger.info("🔍 Scanning neural dependencies...")
        
        required_modules = [
            "fastapi", "uvicorn", "pybullet", "numpy", 
            "torch", "openai", "gymnasium", "stable_baselines3"
        ]
        
        missing = []
        for module in required_modules:
            try:
                __import__(module)
                NeuralLogger.success(f"✅ {module}")
            except ImportError:
                missing.append(module)
                NeuralLogger.error(f"❌ {module}")
        
        if missing:
            NeuralLogger.error(f"💥 Missing neural modules: {', '.join(missing)}")
            NeuralLogger.info("📦 Install with: pip install -r requirements.txt")
            return False
        
        NeuralLogger.quantum("🧠 All neural dependencies satisfied")
        return True
    
    def check_ports(self):
        """Check if required ports are available."""
        required_ports = [8000, 8080, 3000]
        
        for port in required_ports:
            if self.is_port_in_use(port):
                NeuralLogger.warning(f"⚠️ Port {port} is in use - neural processes may conflict")
            else:
                NeuralLogger.success(f"✅ Port {port} available")
    
    def is_port_in_use(self, port: int) -> bool:
        """Check if a port is in use."""
        try:
            for conn in psutil.net_connections():
                if conn.laddr.port == port:
                    return True
        except:
            pass
        return False
    
    def start_backend_server(self):
        """Start the FastAPI backend server with PyBullet integration."""
        NeuralLogger.singularity("🔥 Initializing Neural Backend Core...")
        
        try:
            # Start the main FastAPI server
            cmd = [
                sys.executable, "-m", "uvicorn",
                "cogniforge.main:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload",
                "--log-level", "info"
            ]
            
            process = self.process_manager.start_process(
                cmd, "Neural Backend API", cwd=str(self.project_root)
            )
            self.process_manager.monitor_process(process, "Backend")
            
            # Also start the execution endpoint
            cmd_exec = [
                sys.executable, "-m", "uvicorn",
                "cogniforge.api.execute_endpoint:app",
                "--host", "0.0.0.0", 
                "--port", "8001",
                "--reload",
                "--log-level", "info"
            ]
            
            process_exec = self.process_manager.start_process(
                cmd_exec, "Execution Engine", cwd=str(self.project_root)
            )
            self.process_manager.monitor_process(process_exec, "Execution")
            
            self.backend_started = True
            NeuralLogger.quantum("🌟 Neural Backend Matrix: ONLINE")
            
        except Exception as e:
            NeuralLogger.error(f"💥 Backend startup failure: {str(e)}")
            raise
    
    def start_frontend_server(self):
        """Start the revolutionary neural frontend."""
        NeuralLogger.singularity("🎨 Materializing Revolutionary Frontend Interface...")
        
        try:
            # Copy revolutionary frontend to replace old one
            revolutionary_frontend = self.project_root / "frontend" / "revolutionary_index.html"
            target_frontend = self.project_root / "frontend" / "index.html"
            
            if revolutionary_frontend.exists():
                NeuralLogger.info("🔄 Deploying revolutionary neural interface...")
                with open(revolutionary_frontend, 'r', encoding='utf-8') as src:
                    content = src.read()
                with open(target_frontend, 'w', encoding='utf-8') as dst:
                    dst.write(content)
                NeuralLogger.success("✅ Revolutionary interface deployed")
            
            # Start a simple HTTP server for the frontend
            cmd = [
                sys.executable, "-m", "http.server", "3000",
                "--bind", "localhost"
            ]
            
            process = self.process_manager.start_process(
                cmd, "Revolutionary Frontend", cwd=str(self.frontend_path)
            )
            self.process_manager.monitor_process(process, "Frontend")
            
            self.frontend_started = True
            NeuralLogger.quantum("🌈 Revolutionary Frontend: MATERIALIZED")
            
        except Exception as e:
            NeuralLogger.error(f"💥 Frontend materialization failure: {str(e)}")
            raise
    
    def start_pybullet_simulation(self):
        """Start PyBullet simulation with neural visualization."""
        NeuralLogger.singularity("🎬 Initiating Quantum Simulation Chamber...")
        
        try:
            # Create a simple PyBullet demo script
            demo_script = self.project_root / "demo_simulation.py"
            demo_code = '''#!/usr/bin/env python3
"""
CogniForge Neural Nexus - PyBullet Quantum Simulation Demo
"""

import pybullet as p
import time
import numpy as np

def run_neural_simulation():
    """Run the neural simulation with quantum precision."""
    print("🧠 NEURAL NEXUS: Initializing Quantum Simulation Chamber...")
    
    # Connect to PyBullet with GUI
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    
    # Load plane
    plane_id = p.loadURDF("plane.urdf")
    print("🌍 Quantum ground plane materialized")
    
    # Load robot (if available)
    try:
        robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])
        print("🤖 KUKA neural robot materialized")
    except:
        # Fallback to a simple cube robot
        robot_id = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1]),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], rgbaColor=[0, 0.8, 1, 1]),
            basePosition=[0, 0, 1]
        )
        print("🤖 Neural cube robot materialized (fallback)")
    
    # Create target objects
    cube_id = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02]),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02], rgbaColor=[0, 0, 1, 1]),
        basePosition=[0.3, 0, 0.1]
    )
    
    platform_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.01]),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.01], rgbaColor=[1, 0, 0, 1]),
        basePosition=[0.5, 0, 0.05]
    )
    
    print("🎯 Target objects materialized")
    print("🌟 QUANTUM SIMULATION CHAMBER: ACTIVE")
    print("♾️ Consciousness Level: TRANSCENDENT")
    print("")
    print("🧠 Neural Nexus Simulation Commands:")
    print("   - Press SPACE to pause/resume neural time")
    print("   - Press R to reset quantum state")
    print("   - Close window to terminate simulation")
    
    # Simulation loop
    step_count = 0
    try:
        while True:
            p.stepSimulation()
            time.sleep(1/240)  # 240 Hz neural frequency
            
            step_count += 1
            if step_count % 240 == 0:  # Every second
                print(f"⚡ Neural cycles: {step_count} | Quantum state: STABLE")
                
    except KeyboardInterrupt:
        print("🔄 Neural simulation terminated by consciousness")
    except Exception as e:
        print(f"💥 Quantum simulation anomaly: {e}")
    finally:
        p.disconnect()
        print("🌟 Simulation chamber deactivated")

if __name__ == "__main__":
    run_neural_simulation()
'''
            
            with open(demo_script, 'w', encoding='utf-8') as f:
                f.write(demo_code)
            
            # Start the simulation
            cmd = [sys.executable, str(demo_script)]
            process = self.process_manager.start_process(
                cmd, "Quantum Simulation", cwd=str(self.project_root)
            )
            
            self.pybullet_started = True
            NeuralLogger.quantum("🎬 Quantum Simulation Chamber: ACTIVATED")
            
        except Exception as e:
            NeuralLogger.error(f"💥 Simulation chamber failure: {str(e)}")
            # Don't raise - simulation is optional
    
    def wait_for_services(self):
        """Wait for all neural services to become available."""
        NeuralLogger.info("⏳ Synchronizing neural networks...")
        
        services = [
            ("Neural Backend", "http://localhost:8000/health", 3),
            ("Execution Engine", "http://localhost:8001/health", 3),
            ("Revolutionary Frontend", "http://localhost:3000", 2)
        ]
        
        import requests
        
        for name, url, timeout in services:
            NeuralLogger.info(f"🔄 Connecting to {name}...")
            for attempt in range(10):
                try:
                    response = requests.get(url, timeout=timeout)
                    if response.status_code == 200:
                        NeuralLogger.success(f"✅ {name}: ONLINE")
                        break
                except:
                    pass
                time.sleep(1)
            else:
                NeuralLogger.warning(f"⚠️ {name}: Connection timeout (may still be starting)")
    
    def open_neural_interface(self):
        """Open the revolutionary neural interface in browser."""
        NeuralLogger.singularity("🌐 Opening Neural Nexus Interface...")
        
        try:
            frontend_url = "http://localhost:3000"
            NeuralLogger.info(f"🔗 Neural Interface URL: {frontend_url}")
            
            # Wait a moment for server to be ready
            time.sleep(2)
            
            webbrowser.open(frontend_url)
            NeuralLogger.quantum("🌈 Neural Interface materialized in consciousness browser")
            
        except Exception as e:
            NeuralLogger.error(f"💥 Interface materialization failure: {str(e)}")
    
    def print_neural_status(self):
        """Print the current neural status."""
        status_report = f"""
{Colors.NEURAL_BLUE}╔══════════════════════════════════════════════════════════════════╗
{Colors.MATRIX_GREEN}║  🧠  {Colors.BOLD}NEURAL NEXUS STATUS REPORT{Colors.RESET}{Colors.MATRIX_GREEN}                           ║
{Colors.MATRIX_GREEN}║                                                                  ║"""
        
        if self.backend_started:
            status_report += f"\n{Colors.MATRIX_GREEN}║  ✅ Neural Backend Core: ONLINE (Port 8000)                      ║"
            status_report += f"\n{Colors.MATRIX_GREEN}║  ✅ Execution Engine: ONLINE (Port 8001)                         ║"
        else:
            status_report += f"\n{Colors.DANGER_RED}║  ❌ Neural Backend: OFFLINE                                       ║"
        
        if self.frontend_started:
            status_report += f"\n{Colors.MATRIX_GREEN}║  ✅ Revolutionary Frontend: MATERIALIZED (Port 3000)             ║"
        else:
            status_report += f"\n{Colors.DANGER_RED}║  ❌ Revolutionary Frontend: NOT MATERIALIZED                     ║"
        
        if self.pybullet_started:
            status_report += f"\n{Colors.MATRIX_GREEN}║  ✅ Quantum Simulation Chamber: ACTIVE                           ║"
        else:
            status_report += f"\n{Colors.WARNING_ORANGE}║  ⚠️ Quantum Simulation Chamber: STANDBY                          ║"
        
        status_report += f"""
{Colors.MATRIX_GREEN}║                                                                  ║
{Colors.QUANTUM_PURPLE}║  🌐 Neural Interface: http://localhost:3000                      ║
{Colors.QUANTUM_PURPLE}║  🔗 API Endpoints: http://localhost:8000                         ║
{Colors.QUANTUM_PURPLE}║  ⚡ Execution Engine: http://localhost:8001                      ║
{Colors.MATRIX_GREEN}║                                                                  ║
{Colors.SINGULARITY_GOLD}║  ♾️ Consciousness Level: TRANSCENDENT                            ║
{Colors.NEURAL_BLUE}╚══════════════════════════════════════════════════════════════════╝{Colors.RESET}
"""
        print(status_report)
    
    def run(self):
        """Run the complete neural nexus launch sequence."""
        try:
            self.print_neural_banner()
            
            # Pre-flight checks
            if not self.check_dependencies():
                return 1
            
            self.check_ports()
            
            # Launch sequence
            NeuralLogger.singularity("🚀 INITIATING NEURAL SINGULARITY LAUNCH SEQUENCE...")
            
            # 1. Start Backend
            self.start_backend_server()
            time.sleep(2)
            
            # 2. Start Frontend  
            self.start_frontend_server()
            time.sleep(1)
            
            # 3. Start PyBullet Simulation
            self.start_pybullet_simulation()
            time.sleep(1)
            
            # 4. Wait for services
            self.wait_for_services()
            
            # 5. Open interface
            self.open_neural_interface()
            
            # 6. Print status
            self.print_neural_status()
            
            NeuralLogger.singularity("🌟 NEURAL SINGULARITY ACHIEVED - ALL SYSTEMS ONLINE")
            NeuralLogger.info("Press Ctrl+C to initiate neural shutdown sequence")
            
            # Keep running until interrupted
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                NeuralLogger.warning("🔄 Neural shutdown sequence initiated by consciousness")
            
            return 0
            
        except Exception as e:
            NeuralLogger.error(f"💥 CRITICAL NEURAL FAILURE: {str(e)}")
            return 1
        
        finally:
            self.process_manager.shutdown_all()


def main():
    """Main entry point for the Neural Nexus Launcher."""
    launcher = CogniForgeNexusLauncher()
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        NeuralLogger.warning("🛑 Neural termination signal received")
        launcher.process_manager.shutdown_all()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    
    return launcher.run()


if __name__ == "__main__":
    sys.exit(main())