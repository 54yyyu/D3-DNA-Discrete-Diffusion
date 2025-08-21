#!/usr/bin/env python3
"""
Test script for D3-DNA API

Simple test to verify the API is working correctly
"""

import requests
import json
import time
from typing import Dict, Any


def test_health_endpoint(base_url: str) -> bool:
    """Test the health endpoint"""
    try:
        print("🔍 Testing health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Status: {data['status']}")
            print(f"   GPU Available: {data['gpu_available']}")
            print(f"   Loaded Models: {len(data['loaded_models'])}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


def test_datasets_endpoint(base_url: str) -> bool:
    """Test the datasets info endpoint"""
    try:
        print("🔍 Testing datasets endpoint...")
        response = requests.get(f"{base_url}/datasets/info", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Datasets endpoint working")
            print(f"   Available datasets: {len(data['datasets'])}")
            for dataset in data['datasets']:
                print(f"   - {dataset['name']}: {dataset['description']}")
            return True
        else:
            print(f"❌ Datasets endpoint failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Datasets endpoint error: {e}")
        return False


def test_generation_endpoint(base_url: str) -> bool:
    """Test the generation endpoint with a small sample"""
    try:
        print("🔍 Testing generation endpoint...")
        
        # Small test request
        test_request = {
            "num_samples": 2,
            "steps": 10,
            "architecture": "transformer",
            "dataset": "deepstarr",
            "dev_activity": 1.0,
            "hk_activity": 1.5,
            "include_oracle": True,
            "include_sp_mse": False,  # Skip SP-MSE for quick test
            "include_visualization": True,
            "save_intermediate_steps": True
        }
        
        print(f"   Request: {json.dumps(test_request, indent=2)}")
        print("   Generating sequences...")
        
        start_time = time.time()
        response = requests.post(
            f"{base_url}/generate", 
            json=test_request,
            timeout=120  # 2 minute timeout
        )
        generation_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Generation successful!")
            print(f"   Generation time: {generation_time:.2f}s")
            print(f"   Server reported time: {data.get('generation_time', 'N/A'):.2f}s")
            print(f"   Generated sequences: {len(data.get('final_sequences', []))}")
            print(f"   Visualization steps: {len(data.get('steps', []))}")
            
            # Show first sequence preview
            if data.get('final_sequences'):
                seq = data['final_sequences'][0]
                preview = seq[:50] + "..." if len(seq) > 50 else seq
                print(f"   First sequence: {preview}")
            
            # Show metadata
            if data.get('metadata'):
                meta = data['metadata']
                print(f"   Metadata: {meta.get('dataset')} | {meta.get('num_samples')} samples | {meta.get('total_steps')} steps")
            
            return True
        else:
            print(f"❌ Generation failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('detail', 'Unknown error')}")
            except:
                print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return False


def main():
    """Run all API tests"""
    base_url = "http://localhost:2025"
    
    print("🧬 D3-DNA API Test Suite")
    print("=" * 50)
    print(f"Testing API at: {base_url}")
    print()
    
    # Run tests
    tests = [
        ("Health Check", lambda: test_health_endpoint(base_url)),
        ("Datasets Info", lambda: test_datasets_endpoint(base_url)),
        ("Generation Test", lambda: test_generation_endpoint(base_url))
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}")
        print("-" * 30)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the API server and configuration.")


if __name__ == "__main__":
    main()