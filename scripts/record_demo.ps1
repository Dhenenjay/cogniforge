# CogniForge Demo Recording Script (PowerShell)
# Records a demonstration of CogniForge capabilities

param(
    [string]$OutputDir = "recordings",
    [int]$Width = 1280,
    [int]$Height = 720,
    [switch]$GifOnly,
    [switch]$VideoOnly,
    [switch]$SkipPyBullet
)

# Colors for output
$colors = @{
    Success = "Green"
    Error = "Red"
    Warning = "Yellow"
    Info = "Cyan"
    Header = "Magenta"
}

function Write-ColorOutput($Message, $Color = "White") {
    Write-Host $Message -ForegroundColor $Color
}

function Show-Header {
    Write-ColorOutput ("=" * 60) $colors.Header
    Write-ColorOutput "     COGNIFORGE DEMO RECORDER" $colors.Header
    Write-ColorOutput ("=" * 60) $colors.Header
    Write-Host ""
}

function Check-Dependencies {
    Write-ColorOutput "Checking dependencies..." $colors.Info
    
    $dependencies = @{
        "Python" = (Get-Command python -ErrorAction SilentlyContinue)
        "FFmpeg" = (Get-Command ffmpeg -ErrorAction SilentlyContinue)
        "PyBullet" = (python -c "import pybullet" 2>$null; $?)
        "OpenCV" = (python -c "import cv2" 2>$null; $?)
        "PIL" = (python -c "from PIL import Image" 2>$null; $?)
        "imageio" = (python -c "import imageio" 2>$null; $?)
    }
    
    $missing = @()
    
    foreach ($dep in $dependencies.Keys) {
        if ($dependencies[$dep]) {
            Write-ColorOutput "  ✓ $dep" $colors.Success
        } else {
            Write-ColorOutput "  ✗ $dep" $colors.Error
            $missing += $dep
        }
    }
    
    if ($missing.Count -gt 0) {
        Write-Host ""
        Write-ColorOutput "Missing dependencies detected!" $colors.Warning
        Write-ColorOutput "To install Python packages:" $colors.Info
        Write-Host "  pip install pybullet opencv-python pillow imageio[ffmpeg]"
        
        if ($missing -contains "FFmpeg") {
            Write-ColorOutput "To install FFmpeg:" $colors.Info
            Write-Host "  winget install ffmpeg"
            Write-Host "  or download from: https://ffmpeg.org/download.html"
        }
        
        $continue = Read-Host "Continue anyway? (y/n)"
        if ($continue -ne 'y') {
            exit 1
        }
    }
    
    Write-Host ""
}

function Start-ScreenRecording {
    param(
        [string]$OutputFile,
        [int]$Duration = 30
    )
    
    if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
        Write-ColorOutput "FFmpeg not found. Using Python recording instead." $colors.Warning
        return $null
    }
    
    Write-ColorOutput "Starting screen recording with FFmpeg..." $colors.Info
    
    # Get screen dimensions
    Add-Type @"
        using System;
        using System.Runtime.InteropServices;
        public class Screen {
            [DllImport("user32.dll")]
            public static extern int GetSystemMetrics(int nIndex);
        }
"@
    
    $screenWidth = [Screen]::GetSystemMetrics(0)
    $screenHeight = [Screen]::GetSystemMetrics(1)
    
    # Start FFmpeg recording in background
    $ffmpegArgs = @(
        "-f", "gdigrab",
        "-framerate", "30",
        "-i", "desktop",
        "-t", $Duration,
        "-vcodec", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-s", "${Width}x${Height}",
        "-y",
        $OutputFile
    )
    
    $process = Start-Process ffmpeg -ArgumentList $ffmpegArgs -PassThru -WindowStyle Hidden
    return $process
}

function Run-CogniForgeDemo {
    Write-ColorOutput "Running CogniForge demonstrations..." $colors.Info
    Write-Host ""
    
    # 1. Start API server in background
    Write-ColorOutput "Step 1: Starting API server..." $colors.Info
    $apiProcess = Start-Process python -ArgumentList "main.py" -PassThru -WindowStyle Hidden
    Start-Sleep -Seconds 3
    
    try {
        # 2. Run training with deterministic mode
        Write-ColorOutput "Step 2: Running training with deterministic mode..." $colors.Info
        $training = @"
import requests
import json

response = requests.post('http://localhost:8000/api/v1/train', 
    json={
        'epochs': 10,
        'deterministic': True,
        'seed': 42,
        'pipeline': 'train'
    })
print(f"Training started: {response.json()['task_id']}")
"@
        $training | python
        Start-Sleep -Seconds 2
        
        # 3. Generate code
        Write-ColorOutput "Step 3: Generating code from natural language..." $colors.Info
        $codegen = @"
import requests
import json

response = requests.post('http://localhost:8000/api/v1/generate',
    json={
        'description': 'pick up the red cube and place it on the platform',
        'framework': 'pybullet',
        'style': 'modular'
    })
print(f"Code generation started: {response.json()['task_id']}")
"@
        $codegen | python
        Start-Sleep -Seconds 2
        
        # 4. Run demo
        Write-ColorOutput "Step 4: Running robotic demo..." $colors.Info
        $demo = @"
import requests
import json

response = requests.post('http://localhost:8000/api/v1/demo',
    json={
        'demo_type': 'grasp',
        'object_name': 'red_cube',
        'robot': 'franka',
        'visualize': False
    })
print(f"Demo result: {json.dumps(response.json(), indent=2)}")
"@
        $demo | python
        Start-Sleep -Seconds 2
        
        # 5. Show task status
        Write-ColorOutput "Step 5: Checking task status..." $colors.Info
        $status = @"
import requests
import json

response = requests.get('http://localhost:8000/api/v1/tasks')
tasks = response.json()
print(f"Active tasks: {tasks['total']}")
for task in tasks['tasks'][:3]:
    print(f"  - {task['task_id']}: {task['status']} ({task['progress']*100:.0f}%)")
"@
        $status | python
        
    } finally {
        # Stop API server
        if ($apiProcess -and -not $apiProcess.HasExited) {
            Stop-Process -Id $apiProcess.Id -Force
        }
    }
}

function Run-PyBulletVisualization {
    if ($SkipPyBullet) {
        Write-ColorOutput "Skipping PyBullet visualization (--SkipPyBullet flag set)" $colors.Warning
        return
    }
    
    Write-ColorOutput "Running PyBullet visualization..." $colors.Info
    
    $pybullet_script = @"
import pybullet as p
import pybullet_data
import time
import numpy as np

# Connect to PyBullet
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Setup
p.setGravity(0, 0, -9.81)
planeId = p.loadURDF("plane.urdf")

# Load Kuka robot
kukaId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])

# Load cube
cubeStartPos = [0.5, 0, 0.1]
cubeId = p.loadURDF("cube.urdf", cubeStartPos, globalScaling=0.1)
p.changeVisualShape(cubeId, -1, rgbaColor=[1, 0, 0, 1])

# Camera setup
p.resetDebugVisualizerCamera(
    cameraDistance=1.5,
    cameraYaw=45,
    cameraPitch=-30,
    cameraTargetPosition=[0, 0, 0.5]
)

print("Running simulation for 10 seconds...")
for i in range(300):  # 10 seconds at 30 Hz
    # Move robot
    for j in range(p.getNumJoints(kukaId)):
        target = np.sin(i * 0.01 + j) * 0.5
        p.setJointMotorControl2(kukaId, j, p.POSITION_CONTROL, targetPosition=target)
    
    p.stepSimulation()
    time.sleep(1./30.)

p.disconnect()
print("Simulation complete!")
"@
    
    $pybullet_script | Out-File -Encoding UTF8 "temp_pybullet.py"
    python temp_pybullet.py
    Remove-Item "temp_pybullet.py" -Force
}

function Create-DemoGif {
    param(
        [string]$VideoFile,
        [string]$GifFile
    )
    
    if (-not (Test-Path $VideoFile)) {
        Write-ColorOutput "Video file not found: $VideoFile" $colors.Error
        return
    }
    
    Write-ColorOutput "Creating GIF from video..." $colors.Info
    
    if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
        # Use FFmpeg to create GIF
        $paletteFile = [System.IO.Path]::GetTempFileName() + ".png"
        
        # Generate palette
        ffmpeg -i $VideoFile -vf "fps=10,scale=640:-1:flags=lanczos,palettegen" -y $paletteFile 2>$null
        
        # Create GIF using palette
        ffmpeg -i $VideoFile -i $paletteFile -filter_complex "fps=10,scale=640:-1:flags=lanczos[x];[x][1:v]paletteuse" -y $GifFile 2>$null
        
        Remove-Item $paletteFile -Force
        
        if (Test-Path $GifFile) {
            Write-ColorOutput "✓ GIF created successfully" $colors.Success
        }
    } else {
        # Fallback to Python
        $python_gif = @"
import imageio
import cv2

video = cv2.VideoCapture('$VideoFile')
frames = []

while True:
    ret, frame = video.read()
    if not ret:
        break
    if len(frames) % 3 == 0:  # Skip frames for smaller GIF
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb_frame)

video.release()

if frames:
    imageio.mimsave('$GifFile', frames, fps=10)
    print('GIF created successfully')
"@
        $python_gif | python
    }
}

# Main execution
Show-Header
Check-Dependencies

# Create output directory
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$videoFile = Join-Path $OutputDir "cogniforge_demo_${timestamp}.mp4"
$gifFile = Join-Path $OutputDir "cogniforge_demo_${timestamp}.gif"

Write-ColorOutput "Output configuration:" $colors.Info
Write-Host "  Directory:  $OutputDir"
Write-Host "  Resolution: ${Width}x${Height}"
Write-Host "  Video:      $(if ($GifOnly) {'Skipped'} else {$videoFile})"
Write-Host "  GIF:        $(if ($VideoOnly) {'Skipped'} else {$gifFile})"
Write-Host ""

# Option 1: Try to use Python recorder
Write-ColorOutput "Starting demo recording..." $colors.Header
$use_python_recorder = Read-Host "Use Python recorder (recommended)? (y/n)"

if ($use_python_recorder -eq 'y') {
    # Use the Python script
    python (Join-Path $PSScriptRoot "record_demo.py")
} else {
    # Option 2: Manual recording with FFmpeg
    $recordingProcess = $null
    
    if (-not $GifOnly) {
        $recordingProcess = Start-ScreenRecording -OutputFile $videoFile -Duration 30
        Start-Sleep -Seconds 2
    }
    
    # Run the actual demo
    Run-CogniForgeDemo
    
    if (-not $SkipPyBullet) {
        Run-PyBulletVisualization
    }
    
    # Wait for recording to finish
    if ($recordingProcess) {
        Write-ColorOutput "Waiting for recording to complete..." $colors.Info
        $recordingProcess.WaitForExit()
    }
    
    # Create GIF if requested
    if (-not $VideoOnly -and (Test-Path $videoFile)) {
        Create-DemoGif -VideoFile $videoFile -GifFile $gifFile
    }
}

# Summary
Write-Host ""
Write-ColorOutput ("=" * 60) $colors.Success
Write-ColorOutput "RECORDING COMPLETE!" $colors.Success
Write-ColorOutput ("=" * 60) $colors.Success

if (Test-Path $videoFile) {
    $videoSize = (Get-Item $videoFile).Length / 1MB
    Write-ColorOutput "✓ Video saved: $videoFile" $colors.Success
    Write-Host "  Size: $([math]::Round($videoSize, 1)) MB"
}

if (Test-Path $gifFile) {
    $gifSize = (Get-Item $gifFile).Length / 1MB
    Write-ColorOutput "✓ GIF saved: $gifFile" $colors.Success
    Write-Host "  Size: $([math]::Round($gifSize, 1)) MB"
}

Write-Host ""
Write-ColorOutput "You can now share these files to showcase CogniForge!" $colors.Info
Write-Host ""

# Open output directory
$openFolder = Read-Host "Open output folder? (y/n)"
if ($openFolder -eq 'y') {
    explorer.exe $OutputDir
}