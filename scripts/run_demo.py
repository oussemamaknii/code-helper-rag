#!/usr/bin/env python3
"""
Python Code Helper RAG System - Demo Runner

This script demonstrates different ways to run the project:
1. API Demo with mocked data
2. Individual component tests
3. Evaluation system demo
4. Production deployment demo

Run this script to see all available options and how to use them.
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header(title):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"🚀 {title}")
    print(f"{'='*60}")

def print_section(title):
    """Print a formatted section."""
    print(f"\n{'─'*40}")
    print(f"📋 {title}")
    print(f"{'─'*40}")

def main():
    """Main demonstration runner."""
    print_header("Python Code Helper RAG System - Demo Runner")
    
    print("This project has multiple demo modes:")
    print("• 🎯 Mocked data demos (for quick testing)")
    print("• 🔧 Real data integration (requires API keys)")
    print("• 🚀 Production deployment (full infrastructure)")
    
    print_section("Available Demo Scripts")
    
    demos = [
        ("simple_demo.py", "FastAPI server with mocked responses", "MOCKED DATA"),
        ("simple_evaluation_demo.py", "Evaluation system with mock metrics", "MOCKED DATA"),
        ("simple_deploy_demo.py", "Production deployment simulation", "MOCKED DATA"),
        ("test_endpoints.py", "API endpoint testing", "TESTS MOCKED API"),
        ("test_generation.py", "LLM integration testing", "REQUIRES API KEYS"),
        ("test_vector.py", "Vector database testing", "REQUIRES API KEYS"),
        ("test_processing.py", "Data processing testing", "USES SAMPLE DATA"),
        ("deploy.py", "Production deployment", "FULL PRODUCTION")
    ]
    
    for script, description, data_type in demos:
        status = "🟢" if "MOCKED" in data_type else "🟡" if "SAMPLE" in data_type else "🔴"
        print(f"  {status} python scripts/{script}")
        print(f"     └─ {description} ({data_type})")
    
    print_section("Data Usage Explanation")
    
    print("🟢 MOCKED DATA DEMOS (No API keys required):")
    print("   • Use simulated responses and data")
    print("   • Perfect for understanding system architecture")
    print("   • Fast execution, no external dependencies")
    print("   • Examples: simple_demo.py, simple_evaluation_demo.py")
    
    print("\n🟡 SAMPLE DATA TESTS (Minimal requirements):")
    print("   • Use small sample datasets for testing")
    print("   • Test core functionality without full setup")
    print("   • Examples: test_processing.py with small code samples")
    
    print("\n🔴 REAL DATA INTEGRATION (Requires API keys):")
    print("   • Connects to real services (OpenAI, Pinecone, GitHub)")
    print("   • Requires API keys in environment variables")
    print("   • Full production functionality")
    print("   • Examples: Full system with real LLM and vector DB")
    
    print_section("Quick Start Options")
    
    print("1. 🎯 Try the API Demo (Mocked Data):")
    print("   python scripts/simple_demo.py")
    print("   # Then in another terminal:")
    print("   python scripts/test_endpoints.py")
    
    print("\n2. 📊 See Evaluation System:")  
    print("   python scripts/simple_evaluation_demo.py")
    
    print("\n3. 🚀 View Deployment Process:")
    print("   python scripts/simple_deploy_demo.py")
    
    print("\n4. 🔧 Test Individual Components:")
    print("   python scripts/test_processing.py")
    print("   python scripts/test_vector.py  # Requires API keys")
    
    print_section("Setting Up Real Data (Optional)")
    
    print("To use REAL data instead of mocked data:")
    print("1. Copy env.template to .env:")
    print("   cp env.template .env")
    
    print("\n2. Add your API keys to .env:")
    print("   OPENAI_API_KEY=your_openai_api_key")
    print("   PINECONE_API_KEY=your_pinecone_key")
    print("   GITHUB_TOKEN=your_github_token")
    print("   ANTHROPIC_API_KEY=your_anthropic_key")
    
    print("\n3. Run the full system:")
    print("   python -m uvicorn src.api.app:app --reload")
    print("   # Then test with real data:")
    print("   python scripts/test_api.py")
    
    print_section("Production Deployment")
    
    print("For production deployment with Docker/Kubernetes:")
    print("1. Local development with Docker:")
    print("   docker-compose -f docker/docker-compose.yml up")
    
    print("\n2. Full production deployment:")
    print("   python scripts/deploy.py prod --dry-run  # Test first")
    print("   python scripts/deploy.py prod            # Deploy")
    
    print_section("Current Project Status")
    
    print("✅ All 8 phases completed")
    print("✅ Production-ready codebase")
    print("✅ Comprehensive testing suite")
    print("✅ Docker and Kubernetes deployment")
    print("✅ CI/CD pipeline with GitHub Actions")
    print("✅ Monitoring with Prometheus + Grafana")
    
    print(f"\n🎉 The system is ready! Choose a demo option above to get started.")
    
    print_section("Recommended First Steps")
    
    print("1. Start with mocked demos to understand the architecture")
    print("2. Try individual component tests")
    print("3. Set up API keys for real data integration")
    print("4. Deploy to production when ready")
    
    print(f"\n{'='*60}")
    print("Happy coding! 🐍✨")

if __name__ == "__main__":
    main() 