#!/usr/bin/env python3
"""
Development Server Runner for CogniForge API

This script runs the FastAPI application with uvicorn in development mode
with hot reload and detailed logging.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import uvicorn
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_dev_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = True,
    workers: int = 1,
    log_level: str = "info",
    ssl_keyfile: Optional[str] = None,
    ssl_certfile: Optional[str] = None
):
    """
    Run the development server with uvicorn.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload on code changes
        workers: Number of worker processes (ignored if reload=True)
        log_level: Logging level
        ssl_keyfile: Path to SSL key file for HTTPS
        ssl_certfile: Path to SSL certificate file for HTTPS
    """
    
    logger.info("=" * 60)
    logger.info(" COGNIFORGE API - DEVELOPMENT SERVER")
    logger.info("=" * 60)
    logger.info(f"Starting server at http://{host}:{port}")
    logger.info(f"Hot reload: {'ENABLED' if reload else 'DISABLED'}")
    logger.info(f"Log level: {log_level.upper()}")
    
    if ssl_keyfile and ssl_certfile:
        logger.info(f"SSL/HTTPS: ENABLED")
        logger.info(f"Access via: https://{host}:{port}")
    
    logger.info("-" * 60)
    logger.info("Available endpoints:")
    logger.info(f"  • API Documentation: http://{host}:{port}/docs")
    logger.info(f"  • Alternative Docs:  http://{host}:{port}/redoc")
    logger.info(f"  • Health Check:      http://{host}:{port}/health")
    logger.info(f"  • Training API:      http://{host}:{port}/api/v1/train")
    logger.info(f"  • Code Generation:   http://{host}:{port}/api/v1/generate")
    logger.info(f"  • Demo API:          http://{host}:{port}/api/v1/demo")
    logger.info("-" * 60)
    logger.info("Press CTRL+C to stop the server")
    logger.info("=" * 60 + "\n")
    
    try:
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=reload,
            workers=1 if reload else workers,  # Force single worker in reload mode
            log_level=log_level,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
            access_log=True,
            reload_dirs=[str(project_root)],
            reload_includes=["*.py", "*.yaml", "*.json"],
            reload_excludes=["*.pyc", "__pycache__", ".git", "*.log"]
        )
    except KeyboardInterrupt:
        logger.info("\n" + "=" * 60)
        logger.info("Server stopped by user")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the development server."""
    parser = argparse.ArgumentParser(
        description="Run CogniForge API development server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_dev.py                    # Run with defaults
  python run_dev.py --port 8080        # Run on port 8080
  python run_dev.py --no-reload        # Disable hot reload
  python run_dev.py --host 0.0.0.0     # Listen on all interfaces
  python run_dev.py --workers 4        # Run with 4 workers (no reload)
  python run_dev.py --log-level debug  # Enable debug logging
        """
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable auto-reload on code changes"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (ignored if reload is enabled)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        default="info",
        help="Logging level (default: info)"
    )
    
    parser.add_argument(
        "--ssl-keyfile",
        type=str,
        help="Path to SSL key file for HTTPS"
    )
    
    parser.add_argument(
        "--ssl-certfile",
        type=str,
        help="Path to SSL certificate file for HTTPS"
    )
    
    return parser


def main():
    """Main entry point for the development server."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Check if main.py exists
    main_file = project_root / "main.py"
    if not main_file.exists():
        logger.error(f"Error: main.py not found at {main_file}")
        logger.error("Make sure you're running this script from the cogniforge directory")
        sys.exit(1)
    
    # Run the development server
    run_dev_server(
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
        workers=args.workers,
        log_level=args.log_level,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile
    )


if __name__ == "__main__":
    main()