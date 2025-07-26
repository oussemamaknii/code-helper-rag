#!/usr/bin/env python3
"""
Test script for the data ingestion pipeline.

This script demonstrates how to use the ingestion pipeline to collect
data from GitHub and Stack Overflow with proper configuration.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pipeline import DataIngestionPipeline, PipelineConfig
from src.ingestion.github_crawler import GitHubCrawler
from src.ingestion.stackoverflow_collector import StackOverflowCollector
from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


async def test_github_crawler():
    """Test GitHub crawler functionality."""
    print("\n🔍 Testing GitHub Crawler...")
    
    try:
        # Create crawler with limited scope for testing
        crawler = GitHubCrawler(
            github_token=settings.github_token,
            max_file_size=50000,  # 50KB limit for testing
            min_stars=1000,       # High stars for quality repos
            exclude_forks=True,
            languages=["python"]
        )
        
        # Test health check
        health_status = await crawler.health_check()
        print(f"✅ GitHub API Health: {health_status['status']}")
        
        # Collect a few items for testing
        print("📥 Collecting sample GitHub files...")
        item_count = 0
        async for item in crawler.run_collection(max_repos=2, max_files_per_repo=3):
            item_count += 1
            print(f"  📄 {item.metadata['repository_name']}/{item.metadata['file_path']}")
            print(f"     Size: {item.metadata['file_size']} bytes, "
                  f"Functions: {item.metadata['contains_functions']}, "
                  f"Classes: {item.metadata['contains_classes']}")
            
            if item_count >= 5:  # Limit for testing
                break
        
        # Print metrics
        metrics = crawler.get_metrics()
        print(f"✅ GitHub Collection completed:")
        print(f"   Processed: {metrics['processed_items']} items")
        print(f"   Success rate: {metrics['success_rate']:.1f}%")
        
    except Exception as e:
        print(f"❌ GitHub crawler test failed: {e}")
        return False
    
    return True


async def test_stackoverflow_collector():
    """Test Stack Overflow collector functionality."""
    print("\n🔍 Testing Stack Overflow Collector...")
    
    try:
        # Create collector with limited scope for testing
        collector = StackOverflowCollector(
            api_key=settings.stackoverflow_api_key,
            min_question_score=10,  # Higher score for quality
            min_answer_score=5,
            tags=["python"],
            include_accepted_only=False
        )
        
        # Test with context manager
        async with collector:
            # Test health check
            health_status = await collector.health_check()
            print(f"✅ Stack Overflow API Health: {health_status['status']}")
            
            # Collect a few items for testing
            print("📥 Collecting sample Q&A pairs...")
            item_count = 0
            async for item in collector.run_collection(
                max_questions=3, 
                max_answers_per_question=2
            ):
                item_count += 1
                print(f"  ❓ Q{item.metadata['question_id']}: {item.metadata['question_title'][:60]}...")
                print(f"     Tags: {', '.join(item.metadata['tags'][:3])}")
                print(f"     Scores: Q{item.metadata['question_score']}/A{item.metadata['answer_score']}")
                print(f"     Accepted: {item.metadata['is_accepted']}")
                
                if item_count >= 5:  # Limit for testing
                    break
        
        # Print metrics
        metrics = collector.get_metrics()
        print(f"✅ Stack Overflow Collection completed:")
        print(f"   Processed: {metrics['processed_items']} items")
        print(f"   Success rate: {metrics['success_rate']:.1f}%")
        
    except Exception as e:
        print(f"❌ Stack Overflow collector test failed: {e}")
        return False
    
    return True


async def test_full_pipeline():
    """Test the complete ingestion pipeline."""
    print("\n🔍 Testing Complete Ingestion Pipeline...")
    
    try:
        # Create pipeline configuration for testing
        config = PipelineConfig(
            enable_github=True,
            enable_stackoverflow=True,
            github_max_repos=2,
            github_max_files_per_repo=2,
            stackoverflow_max_questions=3,
            stackoverflow_max_answers_per_question=2,
            max_concurrent_collectors=2,
            batch_size=10,
            save_to_file=False,  # Don't save files during testing
            continue_on_error=True
        )
        
        # Create and run pipeline
        pipeline = DataIngestionPipeline(config)
        
        print("🚀 Starting pipeline...")
        item_count = 0
        source_counts = {}
        
        async for item in pipeline.run_pipeline():
            item_count += 1
            source_type = item.source_type
            source_counts[source_type] = source_counts.get(source_type, 0) + 1
            
            print(f"  📦 Item {item_count}: {source_type} - {len(item.content[:100])} chars")
            
            if item_count >= 10:  # Limit for testing
                break
        
        # Get pipeline status
        status = pipeline.get_pipeline_status()
        print(f"✅ Pipeline completed:")
        print(f"   Status: {status['status']}")
        print(f"   Total items: {status['metrics']['total_items_collected']}")
        print(f"   Items by source: {status['metrics']['items_by_source']}")
        print(f"   Processing time: {status['metrics']['processing_time']:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False


def check_environment():
    """Check if required environment variables are set."""
    print("🔧 Checking environment configuration...")
    
    required_vars = {
        'OPENAI_API_KEY': 'OpenAI API key',
        'GITHUB_TOKEN': 'GitHub personal access token',
        'PINECONE_API_KEY': 'Pinecone API key'
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"  ❌ {var}: {description}")
        else:
            # Show only first and last 4 characters for security
            value = os.getenv(var)
            masked_value = f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "***"
            print(f"  ✅ {var}: {masked_value}")
    
    if missing_vars:
        print("\n❌ Missing required environment variables:")
        for var in missing_vars:
            print(var)
        print("\nPlease set these variables in your .env file or environment.")
        return False
    
    # Check optional variables
    optional_vars = {
        'STACKOVERFLOW_API_KEY': 'Stack Overflow API key (optional, but recommended)'
    }
    
    for var, description in optional_vars.items():
        if os.getenv(var):
            value = os.getenv(var)
            masked_value = f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "***"
            print(f"  ✅ {var}: {masked_value}")
        else:
            print(f"  ⚠️  {var}: Not set ({description})")
    
    return True


async def main():
    """Main test function."""
    print("🚀 Data Ingestion Pipeline Test")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Run tests
    tests = [
        ("GitHub Crawler", test_github_crawler),
        ("Stack Overflow Collector", test_stackoverflow_collector),
        ("Full Pipeline", test_full_pipeline)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = await test_func()
            results[test_name] = success
        except KeyboardInterrupt:
            print(f"\n⏹️  Test interrupted by user")
            break
        except Exception as e:
            print(f"❌ Unexpected error in {test_name}: {e}")
            results[test_name] = False
    
    # Print summary
    print(f"\n{'='*50}")
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The ingestion pipeline is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1) 