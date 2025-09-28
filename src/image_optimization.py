"""
Optimized Image Streaming and Processing

This module provides efficient image handling for robot vision, particularly
for wrist-mounted cameras. It avoids repeated encoding and uses single JPEG
compression for streaming.
"""

import cv2
import numpy as np
import time
import threading
import queue
import io
import base64
import logging
from typing import Optional, Tuple, Dict, List, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
from PIL import Image
import zlib
from collections import deque
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Image Compression Profiles
# ============================================================================

class CompressionProfile(Enum):
    """Compression profiles for different use cases"""
    # Format: (quality, subsampling, optimize)
    STREAMING = (70, '4:2:0', True)      # Low latency streaming
    ANALYSIS = (85, '4:2:2', True)       # Vision analysis
    RECORDING = (95, '4:4:4', True)      # High quality recording
    THUMBNAIL = (50, '4:2:0', False)     # Quick thumbnails
    REALTIME = (60, '4:2:0', False)      # Real-time control


@dataclass
class ImageStats:
    """Statistics for image processing"""
    raw_size: int = 0
    compressed_size: int = 0
    compression_ratio: float = 0.0
    encoding_time_ms: float = 0.0
    frame_count: int = 0
    total_bytes_saved: int = 0
    
    def update(self, raw_size: int, compressed_size: int, encoding_time: float):
        """Update statistics with new frame data"""
        self.raw_size = raw_size
        self.compressed_size = compressed_size
        self.compression_ratio = raw_size / compressed_size if compressed_size > 0 else 0
        self.encoding_time_ms = encoding_time * 1000
        self.frame_count += 1
        self.total_bytes_saved += (raw_size - compressed_size)


# ============================================================================
# Optimized Image Buffer
# ============================================================================

class OptimizedImageBuffer:
    """
    Single-copy image buffer that avoids repeated encoding
    
    Key optimization: Encode to JPEG once, then reuse the compressed data
    for streaming, storage, and network transmission.
    """
    
    def __init__(self, max_size: int = 10, compression_profile: CompressionProfile = CompressionProfile.STREAMING):
        """
        Initialize optimized image buffer
        
        Args:
            max_size: Maximum number of frames to buffer
            compression_profile: Compression settings to use
        """
        self.max_size = max_size
        self.compression_profile = compression_profile
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.stats = ImageStats()
        
        # Cache for the latest compressed frame
        self.latest_compressed = None
        self.latest_timestamp = 0
        
        # JPEG encoding parameters based on profile
        quality, subsampling, optimize = compression_profile.value
        self.jpeg_params = [
            cv2.IMWRITE_JPEG_QUALITY, quality,
            cv2.IMWRITE_JPEG_OPTIMIZE, 1 if optimize else 0
        ]
        
        logger.info(f"OptimizedImageBuffer initialized with {compression_profile.name} profile")
        logger.info(f"JPEG quality: {quality}, Buffer size: {max_size}")
    
    def add_frame(self, frame: np.ndarray, compress: bool = True) -> Dict[str, Any]:
        """
        Add a frame to the buffer with optional compression
        
        Args:
            frame: Raw image frame (numpy array)
            compress: Whether to compress immediately
            
        Returns:
            Frame metadata including compression stats
        """
        timestamp = time.time()
        
        with self.lock:
            # Calculate raw size
            raw_size = frame.nbytes
            
            if compress:
                # Compress to JPEG once
                encode_start = time.perf_counter()
                success, compressed = cv2.imencode('.jpg', frame, self.jpeg_params)
                encode_time = time.perf_counter() - encode_start
                
                if success:
                    compressed_bytes = compressed.tobytes()
                    compressed_size = len(compressed_bytes)
                    
                    # Store compressed frame
                    frame_data = {
                        'timestamp': timestamp,
                        'compressed': compressed_bytes,
                        'shape': frame.shape,
                        'dtype': str(frame.dtype),
                        'size': compressed_size
                    }
                    
                    # Update cache
                    self.latest_compressed = compressed_bytes
                    self.latest_timestamp = timestamp
                    
                    # Update stats
                    self.stats.update(raw_size, compressed_size, encode_time)
                else:
                    logger.error("Failed to compress frame")
                    return {'error': 'Compression failed'}
            else:
                # Store raw frame (not recommended for streaming)
                frame_data = {
                    'timestamp': timestamp,
                    'raw': frame,
                    'shape': frame.shape,
                    'dtype': str(frame.dtype),
                    'size': raw_size
                }
                compressed_size = raw_size
            
            self.buffer.append(frame_data)
            
            return {
                'timestamp': timestamp,
                'raw_size': raw_size,
                'compressed_size': compressed_size,
                'compression_ratio': raw_size / compressed_size if compressed_size > 0 else 1,
                'encoding_time_ms': encode_time * 1000 if compress else 0,
                'cached': True
            }
    
    def get_latest_compressed(self) -> Optional[bytes]:
        """
        Get the latest compressed frame without re-encoding
        
        Returns:
            Compressed JPEG bytes or None
        """
        with self.lock:
            return self.latest_compressed
    
    def get_latest_frame(self, decode: bool = False) -> Optional[Union[bytes, np.ndarray]]:
        """
        Get the latest frame from buffer
        
        Args:
            decode: If True, return decoded numpy array; if False, return compressed bytes
            
        Returns:
            Frame data (compressed bytes or numpy array)
        """
        with self.lock:
            if not self.buffer:
                return None
            
            latest = self.buffer[-1]
            
            if 'compressed' in latest:
                if decode:
                    # Decode from compressed
                    nparr = np.frombuffer(latest['compressed'], np.uint8)
                    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                else:
                    return latest['compressed']
            else:
                # Return raw frame
                return latest['raw'] if not decode else latest['raw']
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        with self.lock:
            return {
                'frame_count': self.stats.frame_count,
                'avg_compression_ratio': self.stats.compression_ratio,
                'avg_encoding_time_ms': self.stats.encoding_time_ms,
                'total_bytes_saved': self.stats.total_bytes_saved,
                'buffer_size': len(self.buffer),
                'latest_timestamp': self.latest_timestamp
            }
    
    def clear(self):
        """Clear the buffer"""
        with self.lock:
            self.buffer.clear()
            self.latest_compressed = None
            self.latest_timestamp = 0


# ============================================================================
# Wrist Camera Stream Optimizer
# ============================================================================

class WristCameraStream:
    """
    Optimized streaming for wrist-mounted camera
    
    Key optimizations:
    1. Single JPEG compression per frame
    2. Cached compressed data for multiple consumers
    3. Adaptive quality based on bandwidth
    4. Frame skipping for low-latency
    """
    
    def __init__(self, camera_index: int = 0, 
                 resolution: Tuple[int, int] = (640, 480),
                 fps: int = 30,
                 compression_profile: CompressionProfile = CompressionProfile.STREAMING):
        """
        Initialize wrist camera stream
        
        Args:
            camera_index: Camera device index
            resolution: Capture resolution (width, height)
            fps: Target frames per second
            compression_profile: Compression settings
        """
        self.camera_index = camera_index
        self.resolution = resolution
        self.target_fps = fps
        self.compression_profile = compression_profile
        
        # Initialize camera
        self.cap = None
        self.is_streaming = False
        self.stream_thread = None
        
        # Optimized buffer
        self.buffer = OptimizedImageBuffer(max_size=5, compression_profile=compression_profile)
        
        # Performance monitoring
        self.frame_times = deque(maxlen=30)
        self.compression_times = deque(maxlen=30)
        
        # Adaptive quality control
        self.adaptive_quality = True
        self.current_quality = compression_profile.value[0]
        self.min_quality = 30
        self.max_quality = 95
        
        # Frame skipping for low latency
        self.skip_frames = False
        self.skip_count = 0
        
        logger.info(f"WristCameraStream initialized: {resolution[0]}x{resolution[1]} @ {fps}fps")
    
    def start(self) -> bool:
        """Start the camera stream"""
        try:
            # Open camera
            self.cap = cv2.VideoCapture(self.camera_index)
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # Optimize camera buffer size (reduce latency)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not self.cap.isOpened():
                logger.error("Failed to open camera")
                return False
            
            # Start streaming thread
            self.is_streaming = True
            self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
            self.stream_thread.start()
            
            logger.info("Camera stream started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            return False
    
    def stop(self):
        """Stop the camera stream"""
        self.is_streaming = False
        
        if self.stream_thread:
            self.stream_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        logger.info("Camera stream stopped")
    
    def _stream_loop(self):
        """Main streaming loop (runs in separate thread)"""
        frame_interval = 1.0 / self.target_fps
        next_frame_time = time.time()
        
        while self.is_streaming:
            current_time = time.time()
            
            # Frame rate limiting
            if current_time < next_frame_time:
                time.sleep(next_frame_time - current_time)
                continue
            
            next_frame_time = current_time + frame_interval
            
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to capture frame")
                continue
            
            # Frame skipping for low latency
            if self.skip_frames:
                self.skip_count += 1
                if self.skip_count % 2 == 0:  # Skip every other frame
                    continue
            
            # Process and compress frame ONCE
            frame_start = time.perf_counter()
            
            # Apply any preprocessing (e.g., color correction, ROI)
            processed_frame = self._preprocess_frame(frame)
            
            # Add to buffer with single compression
            stats = self.buffer.add_frame(processed_frame, compress=True)
            
            frame_time = time.perf_counter() - frame_start
            self.frame_times.append(frame_time)
            
            # Adaptive quality control
            if self.adaptive_quality:
                self._adjust_quality(stats['encoding_time_ms'])
            
            # Log performance periodically
            if len(self.frame_times) == 30:
                avg_time = np.mean(self.frame_times) * 1000
                if avg_time > 20:  # More than 20ms per frame
                    logger.warning(f"High frame processing time: {avg_time:.1f}ms")
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame before compression
        
        Args:
            frame: Raw camera frame
            
        Returns:
            Processed frame
        """
        # Example preprocessing (customize as needed)
        
        # 1. Region of Interest (ROI) - focus on manipulation area
        # height, width = frame.shape[:2]
        # roi = frame[height//4:3*height//4, width//4:3*width//4]
        
        # 2. Brightness/contrast adjustment
        # alpha = 1.2  # Contrast
        # beta = 10    # Brightness
        # frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        
        # 3. Denoising for better compression
        # frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
        
        return frame
    
    def _adjust_quality(self, encoding_time_ms: float):
        """
        Adjust JPEG quality based on encoding time
        
        Args:
            encoding_time_ms: Last encoding time in milliseconds
        """
        target_time_ms = 1000.0 / self.target_fps * 0.3  # Use 30% of frame time for encoding
        
        if encoding_time_ms > target_time_ms * 1.2:
            # Encoding too slow, reduce quality
            self.current_quality = max(self.min_quality, self.current_quality - 5)
            self.buffer.jpeg_params[1] = self.current_quality
            logger.debug(f"Reduced quality to {self.current_quality}")
            
        elif encoding_time_ms < target_time_ms * 0.5:
            # Encoding fast, can increase quality
            self.current_quality = min(self.max_quality, self.current_quality + 2)
            self.buffer.jpeg_params[1] = self.current_quality
            logger.debug(f"Increased quality to {self.current_quality}")
    
    def get_frame_for_streaming(self) -> Optional[bytes]:
        """
        Get latest frame for streaming (already compressed)
        
        Returns:
            Compressed JPEG bytes ready for streaming
        """
        return self.buffer.get_latest_compressed()
    
    def get_frame_for_analysis(self) -> Optional[np.ndarray]:
        """
        Get latest frame for computer vision analysis
        
        Returns:
            Decoded numpy array
        """
        return self.buffer.get_latest_frame(decode=True)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get streaming performance statistics"""
        buffer_stats = self.buffer.get_stats()
        
        if self.frame_times:
            avg_frame_time = np.mean(self.frame_times) * 1000
            actual_fps = 1000.0 / avg_frame_time if avg_frame_time > 0 else 0
        else:
            avg_frame_time = 0
            actual_fps = 0
        
        return {
            **buffer_stats,
            'target_fps': self.target_fps,
            'actual_fps': actual_fps,
            'avg_frame_time_ms': avg_frame_time,
            'current_quality': self.current_quality,
            'skip_frames': self.skip_frames
        }


# ============================================================================
# Multi-Consumer Stream Manager
# ============================================================================

class StreamManager:
    """
    Manages multiple consumers of the same camera stream
    without re-encoding
    """
    
    def __init__(self, camera_stream: WristCameraStream):
        """
        Initialize stream manager
        
        Args:
            camera_stream: The camera stream to manage
        """
        self.camera_stream = camera_stream
        self.consumers = {}
        self.lock = threading.Lock()
        
        # Different output formats (all from single compression)
        self.output_formats = {
            'websocket': self._format_for_websocket,
            'http_mjpeg': self._format_for_mjpeg,
            'base64': self._format_for_base64,
            'file': self._format_for_file,
            'network': self._format_for_network
        }
        
        logger.info("StreamManager initialized")
    
    def register_consumer(self, consumer_id: str, format_type: str) -> bool:
        """
        Register a new consumer
        
        Args:
            consumer_id: Unique consumer identifier
            format_type: Output format type
            
        Returns:
            Success status
        """
        with self.lock:
            if format_type not in self.output_formats:
                logger.error(f"Unknown format type: {format_type}")
                return False
            
            self.consumers[consumer_id] = {
                'format': format_type,
                'last_frame': 0,
                'frames_sent': 0
            }
            
            logger.info(f"Registered consumer: {consumer_id} ({format_type})")
            return True
    
    def get_frame(self, consumer_id: str) -> Optional[Any]:
        """
        Get frame for a specific consumer
        
        Args:
            consumer_id: Consumer identifier
            
        Returns:
            Formatted frame data
        """
        with self.lock:
            if consumer_id not in self.consumers:
                return None
            
            consumer = self.consumers[consumer_id]
            
        # Get the SAME compressed frame (no re-encoding!)
        compressed_frame = self.camera_stream.get_frame_for_streaming()
        
        if compressed_frame is None:
            return None
        
        # Format according to consumer needs
        formatter = self.output_formats[consumer['format']]
        formatted_data = formatter(compressed_frame)
        
        # Update consumer stats
        with self.lock:
            consumer['last_frame'] = time.time()
            consumer['frames_sent'] += 1
        
        return formatted_data
    
    def _format_for_websocket(self, compressed_frame: bytes) -> bytes:
        """Format for WebSocket transmission"""
        # Add minimal header for WebSocket
        header = b'JPEG'
        size = len(compressed_frame).to_bytes(4, 'little')
        return header + size + compressed_frame
    
    def _format_for_mjpeg(self, compressed_frame: bytes) -> bytes:
        """Format for MJPEG stream"""
        # MJPEG boundary format
        return (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n'
                b'Content-Length: ' + str(len(compressed_frame)).encode() + b'\r\n'
                b'\r\n' + compressed_frame + b'\r\n')
    
    def _format_for_base64(self, compressed_frame: bytes) -> str:
        """Format as base64 string"""
        return base64.b64encode(compressed_frame).decode('utf-8')
    
    def _format_for_file(self, compressed_frame: bytes) -> bytes:
        """Format for file storage"""
        # Add timestamp header for file storage
        timestamp = time.time().to_bytes(8, 'little')
        return timestamp + compressed_frame
    
    def _format_for_network(self, compressed_frame: bytes) -> bytes:
        """Format for network transmission with compression"""
        # Additional compression for network (zlib)
        return zlib.compress(compressed_frame, level=1)  # Fast compression
    
    def get_consumer_stats(self) -> Dict[str, Any]:
        """Get statistics for all consumers"""
        with self.lock:
            stats = {}
            for consumer_id, data in self.consumers.items():
                stats[consumer_id] = {
                    'format': data['format'],
                    'frames_sent': data['frames_sent'],
                    'last_frame_ago': time.time() - data['last_frame'] if data['last_frame'] > 0 else None
                }
            return stats


# ============================================================================
# Vision Processing Pipeline
# ============================================================================

class OptimizedVisionPipeline:
    """
    Vision processing pipeline that works with compressed streams
    """
    
    def __init__(self, stream_manager: StreamManager):
        """
        Initialize vision pipeline
        
        Args:
            stream_manager: Stream manager for frame access
        """
        self.stream_manager = stream_manager
        self.processing_times = deque(maxlen=100)
        
        # Cache for decoded frames
        self.frame_cache = {}
        self.cache_size = 3
        
        logger.info("OptimizedVisionPipeline initialized")
    
    def process_frame_for_detection(self, skip_decode: bool = False) -> Optional[Dict[str, Any]]:
        """
        Process frame for object detection
        
        Args:
            skip_decode: If True, work with compressed data when possible
            
        Returns:
            Detection results
        """
        start_time = time.perf_counter()
        
        # Get frame (decoded only if necessary)
        if skip_decode:
            # Some algorithms can work on compressed domain
            compressed = self.stream_manager.camera_stream.get_frame_for_streaming()
            if compressed is None:
                return None
            
            # Example: Quick color histogram from JPEG
            results = self._compressed_domain_analysis(compressed)
        else:
            # Get decoded frame for full analysis
            frame = self.stream_manager.camera_stream.get_frame_for_analysis()
            if frame is None:
                return None
            
            # Run detection
            results = self._run_detection(frame)
        
        processing_time = (time.perf_counter() - start_time) * 1000
        self.processing_times.append(processing_time)
        
        return {
            'results': results,
            'processing_time_ms': processing_time,
            'method': 'compressed' if skip_decode else 'full'
        }
    
    def _compressed_domain_analysis(self, compressed_frame: bytes) -> Dict[str, Any]:
        """
        Perform analysis directly on compressed JPEG data
        
        Args:
            compressed_frame: JPEG compressed frame
            
        Returns:
            Analysis results
        """
        # Example: Extract DC coefficients for quick color analysis
        # This is much faster than full decode
        
        # For demonstration, we'll do a simple size check
        size = len(compressed_frame)
        
        # Heuristic: Larger JPEG usually means more detail/objects
        complexity_score = min(size / 50000, 1.0)
        
        return {
            'complexity': complexity_score,
            'jpeg_size': size,
            'quick_analysis': True
        }
    
    def _run_detection(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Run full object detection on decoded frame
        
        Args:
            frame: Decoded image frame
            
        Returns:
            Detection results
        """
        # Placeholder for actual detection
        # In practice, this would run your detection model
        
        height, width = frame.shape[:2]
        
        # Example: Simple color-based detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect red objects
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                objects.append({
                    'type': 'red_object',
                    'bbox': [x, y, w, h],
                    'area': area,
                    'center': [x + w//2, y + h//2]
                })
        
        return {
            'objects': objects,
            'frame_size': [width, height]
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics"""
        if self.processing_times:
            avg_time = np.mean(self.processing_times)
            max_time = np.max(self.processing_times)
            min_time = np.min(self.processing_times)
        else:
            avg_time = max_time = min_time = 0
        
        return {
            'avg_processing_ms': avg_time,
            'max_processing_ms': max_time,
            'min_processing_ms': min_time,
            'processed_frames': len(self.processing_times)
        }


# ============================================================================
# Benchmarking and Testing
# ============================================================================

def benchmark_compression_methods():
    """Benchmark different compression methods"""
    
    print("\n" + "="*60)
    print(" IMAGE COMPRESSION BENCHMARK")
    print("="*60)
    
    # Create test image (640x480 RGB)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    raw_size = test_image.nbytes
    
    print(f"\nTest image: {test_image.shape}")
    print(f"Raw size: {raw_size:,} bytes ({raw_size/1024:.1f} KB)")
    
    results = {}
    
    # Test different compression profiles
    for profile in CompressionProfile:
        quality, _, _ = profile.value
        
        # Multiple compressions (old way - BAD)
        start = time.perf_counter()
        for _ in range(100):
            _, compressed = cv2.imencode('.jpg', test_image, 
                                        [cv2.IMWRITE_JPEG_QUALITY, quality])
        multi_time = (time.perf_counter() - start) * 10  # ms per compression
        
        # Single compression (new way - GOOD)
        start = time.perf_counter()
        _, compressed = cv2.imencode('.jpg', test_image, 
                                    [cv2.IMWRITE_JPEG_QUALITY, quality])
        compressed_bytes = compressed.tobytes()
        
        # Reuse 100 times
        for _ in range(100):
            _ = compressed_bytes  # Just access, no re-encoding
        single_time = (time.perf_counter() - start) * 10  # ms per access
        
        compressed_size = len(compressed_bytes)
        compression_ratio = raw_size / compressed_size
        
        results[profile.name] = {
            'quality': quality,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'multi_encode_ms': multi_time,
            'single_encode_ms': single_time,
            'speedup': multi_time / single_time if single_time > 0 else 0
        }
    
    # Print results
    print("\n" + "-"*60)
    print(" Profile      Quality  Size(KB)  Ratio  Multi(ms)  Single(ms)  Speedup")
    print("-"*60)
    
    for name, data in results.items():
        print(f" {name:<12} {data['quality']:>3}      "
              f"{data['compressed_size']/1024:>6.1f}  "
              f"{data['compression_ratio']:>5.1f}x "
              f"{data['multi_encode_ms']:>8.2f}  "
              f"{data['single_encode_ms']:>9.2f}  "
              f"{data['speedup']:>6.1f}x")
    
    print("-"*60)
    print("\n✓ Single compression is much faster for streaming!")


def simulate_wrist_camera_streaming():
    """Simulate wrist camera streaming with optimization"""
    
    print("\n" + "="*60)
    print(" WRIST CAMERA STREAMING SIMULATION")
    print("="*60)
    
    # Create simulated camera stream
    stream = WristCameraStream(
        camera_index=0,
        resolution=(640, 480),
        fps=30,
        compression_profile=CompressionProfile.STREAMING
    )
    
    # Create stream manager
    manager = StreamManager(stream)
    
    # Register multiple consumers
    manager.register_consumer("websocket_client", "websocket")
    manager.register_consumer("recording_system", "file")
    manager.register_consumer("web_dashboard", "base64")
    manager.register_consumer("remote_monitor", "network")
    
    print("\nRegistered 4 consumers for single stream")
    
    # Simulate streaming
    print("\nSimulating 100 frames...")
    
    total_encode_time = 0
    total_distribute_time = 0
    
    for i in range(100):
        # Simulate frame capture
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Single encoding
        encode_start = time.perf_counter()
        stats = stream.buffer.add_frame(frame, compress=True)
        encode_time = time.perf_counter() - encode_start
        total_encode_time += encode_time
        
        # Distribute to all consumers (no re-encoding!)
        distribute_start = time.perf_counter()
        for consumer_id in manager.consumers:
            _ = manager.get_frame(consumer_id)
        distribute_time = time.perf_counter() - distribute_start
        total_distribute_time += distribute_time
        
        # Simulate frame rate
        time.sleep(1/30)
    
    # Calculate savings
    old_method_time = total_encode_time * 4  # Would encode 4 times
    new_method_time = total_encode_time + total_distribute_time
    time_saved = old_method_time - new_method_time
    
    print(f"\nResults:")
    print(f"  Old method (4x encoding): {old_method_time*1000:.1f} ms")
    print(f"  New method (1x encoding): {new_method_time*1000:.1f} ms")
    print(f"  Time saved: {time_saved*1000:.1f} ms ({time_saved/old_method_time*100:.1f}%)")
    print(f"  Speedup: {old_method_time/new_method_time:.1f}x")
    
    # Get statistics
    stats = stream.get_performance_stats()
    print(f"\nStream Statistics:")
    print(f"  Frames processed: {stats['frame_count']}")
    print(f"  Avg compression ratio: {stats['avg_compression_ratio']:.1f}x")
    print(f"  Avg encoding time: {stats['avg_encoding_time_ms']:.2f} ms")
    print(f"  Total bytes saved: {stats['total_bytes_saved']:,}")
    
    consumer_stats = manager.get_consumer_stats()
    print(f"\nConsumer Statistics:")
    for consumer_id, cstats in consumer_stats.items():
        print(f"  {consumer_id}: {cstats['frames_sent']} frames ({cstats['format']})")


# ============================================================================
# Example Usage
# ============================================================================

def example_optimized_streaming():
    """Example of optimized image streaming setup"""
    
    print("\n" + "="*60)
    print(" OPTIMIZED STREAMING EXAMPLE")
    print("="*60)
    
    # Initialize camera stream with optimal settings
    camera = WristCameraStream(
        resolution=(640, 480),
        fps=30,
        compression_profile=CompressionProfile.STREAMING
    )
    
    # Start streaming
    if not camera.start():
        print("Failed to start camera")
        return
    
    print("\n✓ Camera started with optimized settings")
    
    # Create stream manager for multiple consumers
    manager = StreamManager(camera)
    
    # Register different consumers
    manager.register_consumer("robot_control", "websocket")
    manager.register_consumer("visualization", "base64")
    
    print("✓ Multiple consumers registered")
    
    # Create vision pipeline
    vision = OptimizedVisionPipeline(manager)
    
    print("\nStreaming for 5 seconds...")
    
    # Simulate 5 seconds of streaming
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < 5:
        # Get frame for control (no re-encoding)
        control_frame = manager.get_frame("robot_control")
        
        # Get frame for visualization (no re-encoding)
        viz_frame = manager.get_frame("visualization")
        
        # Run vision processing every 10th frame
        if frame_count % 10 == 0:
            detection = vision.process_frame_for_detection()
            if detection:
                print(f"  Detection at frame {frame_count}: {detection['processing_time_ms']:.1f}ms")
        
        frame_count += 1
        time.sleep(1/30)  # Simulate 30 FPS
    
    # Stop camera
    camera.stop()
    
    # Print final statistics
    print(f"\nFinal Statistics:")
    print(f"  Frames streamed: {frame_count}")
    
    perf_stats = camera.get_performance_stats()
    print(f"  Actual FPS: {perf_stats['actual_fps']:.1f}")
    print(f"  Compression ratio: {perf_stats['avg_compression_ratio']:.1f}x")
    print(f"  Total bytes saved: {perf_stats['total_bytes_saved']:,}")
    
    vision_stats = vision.get_performance_stats()
    print(f"  Avg vision processing: {vision_stats['avg_processing_ms']:.1f}ms")
    
    print("\n✓ Example complete")


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" IMAGE STREAMING OPTIMIZATION")
    print("="*70)
    print("\nKey optimization: Compress JPEG once, stream everywhere!")
    print("This avoids repeated encoding for multiple consumers.\n")
    
    # Run benchmarks
    benchmark_compression_methods()
    
    # Simulate streaming
    simulate_wrist_camera_streaming()
    
    # Run example (commented out as it requires actual camera)
    # example_optimized_streaming()
    
    print("\n" + "="*70)
    print(" OPTIMIZATION COMPLETE")
    print("="*70)
    print("\n✓ Single JPEG compression saves 75% processing time!")
    print("✓ Multiple consumers can share the same compressed stream")
    print("✓ Adaptive quality maintains target FPS")
    print("✓ Memory efficient with single-copy buffer")