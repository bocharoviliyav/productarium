#!/usr/bin/env python3
"""
Test runner for DeepWiki project.

This script provides a unified way to run all tests or specific test categories.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_test_file(test_file):
    """Run a single test file and return success status."""
    print(f"\n🧪 Running {test_file}...")
    try:
        result = subprocess.run([sys.executable, str(test_file)], 
                              capture_output=True, text=True, cwd=project_root)
        
        if result.returncode == 0:
            print(f"✅ {test_file.name} - PASSED")
            if result.stdout:
                print(f"📄 Output:\n{result.stdout}")
            return True
        else:
            print(f"❌ {test_file.name} - FAILED")
            if result.stderr:
                print(f"💥 Error:\n{result.stderr}")
            if result.stdout:
                print(f"📄 Output:\n{result.stdout}")
            return False
    except Exception as e:
        print(f"💥 {test_file.name} - ERROR: {e}")
        return False

def run_tests(test_dirs):
    """Run all tests in the specified directories using pytest."""
    tests_path = Path(__file__).parent
    cmd = [sys.executable, "-m", "pytest"]

    for d in test_dirs:
        dir_path = tests_path / d
        if dir_path.exists():
            cmd.append(str(dir_path))

    if len(cmd) == 3 and not (tests_path / test_dirs[0]).exists():
        cmd.append(str(tests_path))

    print(f"\n🚀 Running pytest command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode == 0

def check_environment():
    """Check if required environment variables and dependencies are available."""
    print("🔧 Checking test environment...")
    
    # Check for .env file
    env_file = project_root / ".env"
    if env_file.exists():
        print("✅ .env file found")
        from dotenv import load_dotenv
        load_dotenv(env_file)
    else:
        print("⚠️  No .env file found - some tests may fail without local endpoints")
    
    # Check for local endpoints (no cloud API keys required).
    local_endpoints = {
        "OLLAMA_HOST": "Local Ollama LLM server",
        "LOCAL_OPENAI_BASE_URL": "Local OpenAI-compatible API server",
    }
    
    for key, purpose in local_endpoints.items():
        if os.getenv(key):
            print(f"✅ {key} is set ({purpose})")
        else:
            print(f"⚠️  {key} not set - {purpose} may fail")
    
    # Check Python dependencies
    try:
        import adalflow
        print("✅ adalflow available")
    except ImportError:
        print("❌ adalflow not available - install with: pip install adalflow")
    
    try:
        import requests
        print("✅ requests available")
    except ImportError:
        print("❌ requests not available - install with: pip install requests")

def main():
    parser = argparse.ArgumentParser(description="Run DeepWiki tests")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--api", action="store_true", help="Run only API tests")
    parser.add_argument("--check-env", action="store_true", help="Only check environment setup")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Check environment first
    check_environment()
    
    if args.check_env:
        return
    
    # Determine which tests to run
    test_dirs = []
    if args.unit:
        test_dirs.append("unit")
    if args.integration or args.api:
        test_dirs.append("integration")
    
    # If no specific category selected, run all
    if not test_dirs:
        test_dirs = ["unit", "integration"]
    
    print(f"\n🚀 Starting test run for: {', '.join(test_dirs)}")
    
    success = run_tests(test_dirs)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()