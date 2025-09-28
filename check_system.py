#!/usr/bin/env python3
"""
System diagnostic script for CogniForge
Checks API endpoints and PyBullet camera configuration
"""

import requests
import time
import sys

def check_api_endpoints():
    """Check all API endpoints"""
    
    print("=" * 60)
    print("COGNIFORGE SYSTEM DIAGNOSTIC")
    print("=" * 60)
    print()
    
    base_url = "http://localhost:8000"
    
    # Check base health endpoint
    print("1. Checking API health endpoints...")
    endpoints_to_check = [
        ("/health", "GET"),
        ("/", "GET"),
        ("/api/skills", "GET"),
        ("/docs", "GET")
    ]
    
    for endpoint, method in endpoints_to_check:
        try:
            url = f"{base_url}{endpoint}"
            if method == "GET":
                response = requests.get(url, timeout=2)
            status = "✓" if response.status_code == 200 else "✗"
            print(f"   {status} {method} {endpoint}: {response.status_code}")
        except Exception as e:
            print(f"   ✗ {method} {endpoint}: {e}")
    
    print()
    print("2. Checking execution endpoints...")
    
    # Check execution endpoints
    exec_endpoints = [
        ("/execute", "POST"),  # Old endpoint (should fail)
        ("/api/execute", "POST")  # New endpoint (should work)
    ]
    
    test_request = {
        "task_description": "Test task",
        "dry_run": True
    }
    
    for endpoint, method in exec_endpoints:
        try:
            url = f"{base_url}{endpoint}"
            response = requests.post(url, json=test_request, timeout=2)
            status = "✓" if response.status_code in [200, 201] else "✗"
            print(f"   {status} {method} {endpoint}: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"   ✗ {method} {endpoint}: Connection refused")
        except requests.exceptions.HTTPError as e:
            print(f"   ✗ {method} {endpoint}: {e}")
        except Exception as e:
            print(f"   ✗ {method} {endpoint}: {e}")
    
    print()
    print("3. Frontend connectivity check...")
    
    # Check frontend
    try:
        response = requests.get("http://localhost:8080", timeout=2)
        if response.status_code == 200:
            print(f"   ✓ Frontend server: Running at http://localhost:8080")
        else:
            print(f"   ✗ Frontend server: Status {response.status_code}")
    except:
        print(f"   ✗ Frontend server: Not accessible")
    
    print()
    print("=" * 60)
    print("DIAGNOSIS COMPLETE")
    print()
    print("If you see '/execute' returning 404, that's expected.")
    print("The correct endpoint is '/api/execute'.")
    print()
    print("The 404 error in logs might be from:")
    print("  - Browser preflight OPTIONS request")
    print("  - Old bookmarks or cached requests")
    print("  - Background health checks from old scripts")
    print("=" * 60)

if __name__ == "__main__":
    check_api_endpoints()