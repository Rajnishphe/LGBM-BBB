#!/usr/bin/env python3
"""
Test script for BBB Prediction FastAPI
Run this after starting your API server to verify everything works
"""

import requests
import json
import time
import sys
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8000"
TEST_SMILES = [
    {"smiles": "CCO", "name": "Ethanol"},
    {"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "name": "Aspirin"},
    {"smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "name": "Ibuprofen"},
    {"smiles": "CN1CCC[C@H]1C2=CN=CC=C2", "name": "Nicotine"},
    {"smiles": "C1=CC=C(C=C1)C(=O)O", "name": "Benzoic acid"}
]

def test_endpoint(endpoint: str, method: str = "GET", data: Dict = None, files: Dict = None) -> Dict[str, Any]:
    """Test a single endpoint and return results"""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            if files:
                response = requests.post(url, data=data, files=files, timeout=30)
            else:
                response = requests.post(url, json=data, timeout=30)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        return {
            "success": True,
            "status_code": response.status_code,
            "response": response.json() if response.status_code == 200 else response.text,
            "response_time": response.elapsed.total_seconds()
        }
    
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Connection failed - is the server running?",
            "response_time": None
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Request timeout",
            "response_time": None
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "response_time": None
        }

def print_test_result(test_name: str, result: Dict[str, Any]):
    """Print formatted test result"""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")
    
    if result["success"]:
        print(f"✅ Status: SUCCESS (HTTP {result['status_code']})")
        print(f"⏱️  Response Time: {result['response_time']:.3f}s")
        
        if result['status_code'] == 200:
            response = result['response']
            if isinstance(response, dict):
                print(f"📝 Response Preview:")
                print(json.dumps(response, indent=2)[:500] + ("..." if len(str(response)) > 500 else ""))
            else:
                print(f"📝 Response: {response}")
        else:
            print(f"⚠️  Response: {result['response']}")
    else:
        print(f"❌ Status: FAILED")
        print(f"💥 Error: {result['error']}")

def main():
    """Run all API tests"""
    print("🧪 BBB Prediction API Test Suite")
    print(f"🌐 Base URL: {BASE_URL}")
    print(f"⏰ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test 1: Health Check
    result = test_endpoint("/health")
    print_test_result("Health Check", result)
    
    if not result["success"]:
        print("\n❌ Server is not responding. Please check:")
        print("1. Is the server running? (python main.py)")
        print("2. Is it listening on the correct port?")
        print("3. Are there any firewall issues?")
        sys.exit(1)
    
    # Test 2: Model Info
    result = test_endpoint("/model/info")
    print_test_result("Model Information", result)
    
    # Test 3: Model Features
    result = test_endpoint("/model/features")
    print_test_result("Model Features", result)
    
    # Test 4: SMILES Validation
    result = test_endpoint("/validate/smiles", "POST", data={"smiles": "CCO"})
    print_test_result("SMILES Validation", result)
    
    # Test 5: Single Molecule Prediction
    single_data = {
        "smiles": "CCO",
        "name": "Ethanol",
        "threshold": 0.5228
    }
    result = test_endpoint("/predict/single", "POST", single_data)
    print_test_result("Single Molecule Prediction", result)
    
    # Test 6: Batch Prediction
    batch_data = {
        "molecules": TEST_SMILES[:3],  # Test with first 3 molecules
        "threshold": 0.5228
    }
    result = test_endpoint("/predict/batch", "POST", batch_data)
    print_test_result("Batch Prediction (3 molecules)", result)
    
    # Test 7: Model Test
    result = test_endpoint("/test/model", "POST", {"threshold": 0.5228})
    print_test_result("Model Test", result)
    
    # Test 8: Large Batch Prediction
    large_batch_data = {
        "molecules": TEST_SMILES,  # All test molecules
        "threshold": 0.5228
    }
    result = test_endpoint("/predict/batch", "POST", large_batch_data)
    print_test_result("Large Batch Prediction (5 molecules)", result)
    
    # Test 9: Invalid SMILES
    invalid_data = {
        "smiles": "INVALID_SMILES",
        "name": "Invalid Molecule",
        "threshold": 0.5228
    }
    result = test_endpoint("/predict/single", "POST", invalid_data)
    print_test_result("Invalid SMILES Test (Expected to Fail)", result)
    
    # Summary
    print(f"\n{'='*60}")
    print("🏁 TEST SUMMARY")
    print(f"{'='*60}")
    print("✅ Core functionality tests completed!")
    print("📊 Check individual test results above for details.")
    print("\n💡 Next steps:")
    print("1. Try the interactive docs at: http://localhost:8000/docs")
    print("2. Test file uploads with your own CSV/SDF files")
    print("3. Integrate with your application")
    
    print(f"\n⏰ Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()