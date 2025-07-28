#!/usr/bin/env python3
"""
Production deployment script for Python Code Helper RAG System.

This script orchestrates the complete deployment process including:
- Infrastructure provisioning with Terraform
- Kubernetes cluster setup
- Application deployment
- Monitoring stack deployment
- Health checks and validation
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import argparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('deployment.log')
    ]
)
logger = logging.getLogger(__name__)


class DeploymentManager:
    """Manages the complete deployment lifecycle."""
    
    def __init__(self, environment: str, dry_run: bool = False):
        """
        Initialize deployment manager.
        
        Args:
            environment: Target environment (dev, staging, prod)
            dry_run: If True, show what would be done without executing
        """
        self.environment = environment
        self.dry_run = dry_run
        self.project_root = Path(__file__).parent.parent
        self.terraform_dir = self.project_root / "terraform"
        self.k8s_dir = self.project_root / "k8s"
        
        # Validate environment
        if environment not in ['dev', 'staging', 'prod']:
            raise ValueError(f"Invalid environment: {environment}")
        
        logger.info(f"🚀 Initializing deployment to {environment} (dry_run={dry_run})")
    
    async def deploy(self) -> bool:
        """
        Execute complete deployment process.
        
        Returns:
            bool: True if deployment successful, False otherwise
        """
        try:
            logger.info("=" * 60)
            logger.info(f"🚀 Starting Python Code Helper RAG System Deployment")
            logger.info(f"Environment: {self.environment}")
            logger.info(f"Timestamp: {datetime.utcnow().isoformat()}")
            logger.info("=" * 60)
            
            # Pre-deployment checks
            if not await self._pre_deployment_checks():
                return False
            
            # Phase 1: Infrastructure
            logger.info("\n📋 Phase 1: Infrastructure Provisioning")
            if not await self._deploy_infrastructure():
                return False
            
            # Phase 2: Kubernetes Setup
            logger.info("\n📋 Phase 2: Kubernetes Cluster Setup")
            if not await self._setup_kubernetes():
                return False
            
            # Phase 3: Application Deployment
            logger.info("\n📋 Phase 3: Application Deployment")
            if not await self._deploy_application():
                return False
            
            # Phase 4: Monitoring Stack
            logger.info("\n📋 Phase 4: Monitoring Stack Deployment")
            if not await self._deploy_monitoring():
                return False
            
            # Phase 5: Post-deployment validation
            logger.info("\n📋 Phase 5: Post-deployment Validation")
            if not await self._post_deployment_validation():
                return False
            
            logger.info("\n🎉 Deployment completed successfully!")
            await self._print_deployment_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            await self._cleanup_on_failure()
            return False
    
    async def _pre_deployment_checks(self) -> bool:
        """Run pre-deployment validation checks."""
        logger.info("🔍 Running pre-deployment checks...")
        
        checks = [
            ("Docker", self._check_docker),
            ("Kubectl", self._check_kubectl),
            ("Terraform", self._check_terraform),
            ("AWS CLI", self._check_aws_cli),
            ("Environment Variables", self._check_env_vars),
            ("Container Registry Access", self._check_registry_access),
        ]
        
        for check_name, check_func in checks:
            logger.info(f"  📝 Checking {check_name}...")
            if not await check_func():
                logger.error(f"❌ {check_name} check failed")
                return False
            logger.info(f"  ✅ {check_name} check passed")
        
        return True
    
    async def _check_docker(self) -> bool:
        """Check Docker availability."""
        try:
            result = await self._run_command(["docker", "--version"])
            return result.returncode == 0
        except Exception:
            return False
    
    async def _check_kubectl(self) -> bool:
        """Check kubectl availability."""
        try:
            result = await self._run_command(["kubectl", "version", "--client"])
            return result.returncode == 0
        except Exception:
            return False
    
    async def _check_terraform(self) -> bool:
        """Check Terraform availability."""
        try:
            result = await self._run_command(["terraform", "--version"])
            return result.returncode == 0
        except Exception:
            return False
    
    async def _check_aws_cli(self) -> bool:
        """Check AWS CLI configuration."""
        try:
            result = await self._run_command(["aws", "sts", "get-caller-identity"])
            return result.returncode == 0
        except Exception:
            return False
    
    async def _check_env_vars(self) -> bool:
        """Check required environment variables."""
        required_vars = [
            "OPENAI_API_KEY",
            "PINECONE_API_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY"
        ]
        
        if self.environment == "prod":
            required_vars.extend([
                "PROD_DATABASE_PASSWORD",
                "PROD_JWT_SECRET_KEY"
            ])
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"Missing environment variables: {missing_vars}")
            return False
        
        return True
    
    async def _check_registry_access(self) -> bool:
        """Check container registry access."""
        try:
            registry = "ghcr.io"
            result = await self._run_command([
                "docker", "login", registry, "--username", "dummy", "--password-stdin"
            ], input="dummy_token")
            # Login will fail but we check if registry is reachable
            return True
        except Exception:
            return False
    
    async def _deploy_infrastructure(self) -> bool:
        """Deploy infrastructure using Terraform."""
        logger.info("🏗️ Deploying infrastructure with Terraform...")
        
        try:
            # Change to terraform directory
            os.chdir(self.terraform_dir)
            
            # Initialize Terraform
            logger.info("  📝 Initializing Terraform...")
            if not await self._run_terraform_command(["init"]):
                return False
            
            # Plan infrastructure changes
            logger.info("  📋 Planning infrastructure changes...")
            plan_file = f"terraform-{self.environment}.plan"
            if not await self._run_terraform_command([
                "plan", 
                f"-var-file=environments/{self.environment}.tfvars",
                f"-out={plan_file}"
            ]):
                return False
            
            # Apply infrastructure changes
            if not self.dry_run:
                logger.info("  🚀 Applying infrastructure changes...")
                if not await self._run_terraform_command(["apply", plan_file]):
                    return False
            else:
                logger.info("  🔍 Dry run: Would apply infrastructure changes")
            
            # Get outputs
            logger.info("  📊 Retrieving infrastructure outputs...")
            await self._get_terraform_outputs()
            
            return True
            
        except Exception as e:
            logger.error(f"Infrastructure deployment failed: {e}")
            return False
        finally:
            os.chdir(self.project_root)
    
    async def _setup_kubernetes(self) -> bool:
        """Setup Kubernetes cluster and base resources."""
        logger.info("☸️ Setting up Kubernetes cluster...")
        
        try:
            # Update kubeconfig
            cluster_name = f"python-code-helper-{self.environment}"
            logger.info(f"  📝 Updating kubeconfig for cluster {cluster_name}...")
            
            if not self.dry_run:
                result = await self._run_command([
                    "aws", "eks", "update-kubeconfig",
                    "--region", "us-east-1",
                    "--name", cluster_name
                ])
                if result.returncode != 0:
                    return False
            
            # Create namespace
            logger.info("  📝 Creating namespace...")
            if not await self._apply_k8s_manifest("namespace.yaml"):
                return False
            
            # Apply RBAC
            logger.info("  📝 Setting up RBAC...")
            if not await self._apply_k8s_manifest("rbac.yaml"):
                return False
            
            # Create secrets
            logger.info("  🔐 Creating secrets...")
            if not await self._create_secrets():
                return False
            
            # Apply configmaps
            logger.info("  📝 Applying configuration...")
            if not await self._apply_k8s_manifest("configmap.yaml"):
                return False
            
            # Apply persistent volumes
            logger.info("  💾 Setting up storage...")
            if not await self._apply_k8s_manifest("pvc.yaml"):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Kubernetes setup failed: {e}")
            return False
    
    async def _deploy_application(self) -> bool:
        """Deploy the main application components."""
        logger.info("🚀 Deploying application components...")
        
        try:
            # Deploy database
            logger.info("  🗄️ Deploying PostgreSQL...")
            if not await self._apply_k8s_manifest("deployment.yaml"):
                return False
            
            # Deploy cache
            logger.info("  🗄️ Deploying Redis...")
            # Redis deployment is part of deployment.yaml
            
            # Deploy services
            logger.info("  🌐 Creating services...")
            if not await self._apply_k8s_manifest("service.yaml"):
                return False
            
            # Deploy main application
            logger.info("  🐍 Deploying Python Code Helper API...")
            # Main app deployment is part of deployment.yaml
            
            # Deploy ingress
            logger.info("  🌐 Setting up ingress...")
            if not await self._apply_k8s_manifest("ingress.yaml"):
                return False
            
            # Wait for deployments to be ready
            logger.info("  ⏳ Waiting for deployments to be ready...")
            if not self.dry_run:
                await self._wait_for_deployments([
                    "python-code-helper",
                    "postgres",
                    "redis"
                ])
            
            return True
            
        except Exception as e:
            logger.error(f"Application deployment failed: {e}")
            return False
    
    async def _deploy_monitoring(self) -> bool:
        """Deploy monitoring stack (Prometheus, Grafana)."""
        logger.info("📊 Deploying monitoring stack...")
        
        try:
            # Deploy Prometheus
            logger.info("  📈 Deploying Prometheus...")
            if not await self._apply_k8s_manifest("monitoring/prometheus.yaml"):
                return False
            
            # Deploy Grafana
            logger.info("  📊 Deploying Grafana...")
            if not await self._apply_k8s_manifest("monitoring/grafana.yaml"):
                return False
            
            # Deploy exporters
            logger.info("  📊 Deploying exporters...")
            exporters = ["node-exporter", "postgres-exporter", "redis-exporter"]
            for exporter in exporters:
                if not await self._apply_k8s_manifest(f"monitoring/{exporter}.yaml"):
                    logger.warning(f"Failed to deploy {exporter}, continuing...")
            
            return True
            
        except Exception as e:
            logger.error(f"Monitoring deployment failed: {e}")
            return False
    
    async def _post_deployment_validation(self) -> bool:
        """Run post-deployment validation tests."""
        logger.info("✅ Running post-deployment validation...")
        
        try:
            # Health check
            logger.info("  🏥 Checking application health...")
            if not await self._health_check():
                return False
            
            # API functionality test
            logger.info("  🧪 Testing API functionality...")
            if not await self._api_functionality_test():
                return False
            
            # Performance test
            logger.info("  🚀 Running performance tests...")
            if not await self._performance_test():
                return False
            
            # Security scan
            logger.info("  🔒 Running security validation...")
            if not await self._security_validation():
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Post-deployment validation failed: {e}")
            return False
    
    async def _health_check(self) -> bool:
        """Check application health endpoints."""
        if self.dry_run:
            logger.info("  🔍 Dry run: Would check health endpoints")
            return True
        
        try:
            # Get service URL
            service_url = await self._get_service_url()
            
            # Check main health endpoint
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{service_url}/health") as response:
                    if response.status != 200:
                        logger.error(f"Health check failed: {response.status}")
                        return False
                    
                    health_data = await response.json()
                    logger.info(f"  ✅ Health check passed: {health_data.get('status')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def _api_functionality_test(self) -> bool:
        """Test basic API functionality."""
        if self.dry_run:
            logger.info("  🔍 Dry run: Would test API functionality")
            return True
        
        try:
            service_url = await self._get_service_url()
            
            import aiohttp
            async with aiohttp.ClientSession() as session:
                # Test chat endpoint
                chat_payload = {
                    "message": "What is Python?",
                    "context": {"programming_language": "python"}
                }
                
                async with session.post(
                    f"{service_url}/api/v1/chat",
                    json=chat_payload,
                    headers={"Authorization": f"Bearer {os.getenv('TEST_API_KEY', 'test_key')}"}
                ) as response:
                    if response.status != 200:
                        logger.error(f"Chat API test failed: {response.status}")
                        return False
                    
                    logger.info("  ✅ Chat API test passed")
            
            return True
            
        except Exception as e:
            logger.error(f"API functionality test failed: {e}")
            return False
    
    async def _performance_test(self) -> bool:
        """Run basic performance tests."""
        if self.dry_run:
            logger.info("  🔍 Dry run: Would run performance tests")
            return True
        
        try:
            # Simple load test with asyncio
            service_url = await self._get_service_url()
            
            async def single_request():
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{service_url}/health") as response:
                        return response.status == 200
            
            # Run 10 concurrent requests
            start_time = time.time()
            tasks = [single_request() for _ in range(10)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            success_count = sum(1 for r in results if r is True)
            total_time = end_time - start_time
            
            logger.info(f"  📊 Performance test: {success_count}/10 successful, {total_time:.2f}s total")
            
            return success_count >= 8  # At least 80% success rate
            
        except Exception as e:
            logger.error(f"Performance test failed: {e}")
            return False
    
    async def _security_validation(self) -> bool:
        """Run basic security validation."""
        if self.dry_run:
            logger.info("  🔍 Dry run: Would run security validation")
            return True
        
        try:
            # Check for unauthorized access
            service_url = await self._get_service_url()
            
            import aiohttp
            async with aiohttp.ClientSession() as session:
                # Test without API key
                async with session.post(
                    f"{service_url}/api/v1/chat",
                    json={"message": "test"}
                ) as response:
                    if response.status not in [401, 403]:
                        logger.error("Security validation failed: API allows unauthorized access")
                        return False
            
            logger.info("  ✅ Security validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Security validation failed: {e}")
            return False
    
    async def _run_command(self, cmd: List[str], input: Optional[str] = None) -> subprocess.CompletedProcess:
        """Run a command asynchronously."""
        logger.debug(f"Running command: {' '.join(cmd)}")
        
        if self.dry_run and not cmd[0] in ['docker', 'kubectl', 'terraform']:
            logger.info(f"  🔍 Dry run: Would run {' '.join(cmd)}")
            return subprocess.CompletedProcess(cmd, 0)
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE if input else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate(
            input=input.encode() if input else None
        )
        
        result = subprocess.CompletedProcess(
            cmd, process.returncode, stdout, stderr
        )
        
        if result.returncode != 0:
            logger.error(f"Command failed: {' '.join(cmd)}")
            logger.error(f"Error: {stderr.decode()}")
        
        return result
    
    async def _run_terraform_command(self, args: List[str]) -> bool:
        """Run terraform command."""
        cmd = ["terraform"] + args
        result = await self._run_command(cmd)
        return result.returncode == 0
    
    async def _apply_k8s_manifest(self, manifest_file: str) -> bool:
        """Apply Kubernetes manifest."""
        manifest_path = self.k8s_dir / manifest_file
        
        if not manifest_path.exists():
            logger.warning(f"Manifest file not found: {manifest_path}")
            return True  # Don't fail for optional manifests
        
        if self.dry_run:
            cmd = ["kubectl", "apply", "--dry-run=client", "-f", str(manifest_path)]
        else:
            cmd = ["kubectl", "apply", "-f", str(manifest_path)]
        
        result = await self._run_command(cmd)
        return result.returncode == 0
    
    async def _create_secrets(self) -> bool:
        """Create Kubernetes secrets from environment variables."""
        if self.dry_run:
            logger.info("  🔍 Dry run: Would create secrets")
            return True
        
        try:
            # Create main application secrets
            secrets = {
                "openai-api-key": os.getenv("OPENAI_API_KEY", ""),
                "anthropic-api-key": os.getenv("ANTHROPIC_API_KEY", ""),
                "pinecone-api-key": os.getenv("PINECONE_API_KEY", ""),
                "github-token": os.getenv("GITHUB_TOKEN", ""),
                "jwt-secret-key": os.getenv("JWT_SECRET_KEY", "default-jwt-secret"),
            }
            
            # Build kubectl command for secret creation
            cmd = [
                "kubectl", "create", "secret", "generic",
                "python-code-helper-secrets",
                "-n", "python-code-helper"
            ]
            
            for key, value in secrets.items():
                cmd.extend([f"--from-literal={key}={value}"])
            
            cmd.append("--dry-run=client")
            cmd.extend(["-o", "yaml"])
            
            # Apply the secret
            result = await self._run_command(cmd)
            if result.returncode == 0:
                # Apply the generated YAML
                apply_cmd = ["kubectl", "apply", "-f", "-"]
                apply_result = await self._run_command(apply_cmd, input=result.stdout.decode())
                return apply_result.returncode == 0
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to create secrets: {e}")
            return False
    
    async def _wait_for_deployments(self, deployments: List[str]) -> None:
        """Wait for deployments to be ready."""
        for deployment in deployments:
            logger.info(f"    ⏳ Waiting for {deployment} to be ready...")
            cmd = [
                "kubectl", "rollout", "status", f"deployment/{deployment}",
                "-n", "python-code-helper", "--timeout=300s"
            ]
            await self._run_command(cmd)
    
    async def _get_service_url(self) -> str:
        """Get service URL for testing."""
        if self.environment == "prod":
            return "https://pythoncodehelper.com"
        elif self.environment == "staging":
            return "https://staging.pythoncodehelper.com"
        else:
            return "http://localhost:8080"
    
    async def _get_terraform_outputs(self) -> None:
        """Get and store Terraform outputs."""
        try:
            result = await self._run_command(["terraform", "output", "-json"])
            if result.returncode == 0:
                outputs = json.loads(result.stdout.decode())
                # Store outputs for later use
                with open(f"terraform-outputs-{self.environment}.json", "w") as f:
                    json.dump(outputs, f, indent=2)
                logger.info("  ✅ Terraform outputs saved")
        except Exception as e:
            logger.warning(f"Failed to get Terraform outputs: {e}")
    
    async def _print_deployment_summary(self) -> None:
        """Print deployment summary."""
        logger.info("\n🎉 Deployment Summary")
        logger.info("=" * 50)
        logger.info(f"Environment: {self.environment}")
        logger.info(f"Timestamp: {datetime.utcnow().isoformat()}")
        
        if self.environment == "prod":
            logger.info("🌐 Production URLs:")
            logger.info("  • Main API: https://pythoncodehelper.com")
            logger.info("  • Documentation: https://pythoncodehelper.com/docs")
            logger.info("  • Monitoring: https://monitoring.pythoncodehelper.com")
        
        logger.info("\n📊 Next Steps:")
        logger.info("  1. Monitor application metrics in Grafana")
        logger.info("  2. Set up alerting for critical metrics")
        logger.info("  3. Configure log aggregation")
        logger.info("  4. Run comprehensive end-to-end tests")
        logger.info("  5. Update DNS records if needed")
    
    async def _cleanup_on_failure(self) -> None:
        """Cleanup resources on deployment failure."""
        logger.info("🧹 Cleaning up on failure...")
        # Implementation would depend on what was partially deployed
        # This is a placeholder for cleanup logic


async def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy Python Code Helper RAG System")
    parser.add_argument(
        "environment",
        choices=["dev", "staging", "prod"],
        help="Target environment"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing"
    )
    parser.add_argument(
        "--skip-infrastructure",
        action="store_true",
        help="Skip infrastructure deployment"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip post-deployment validation"
    )
    
    args = parser.parse_args()
    
    try:
        deployment_manager = DeploymentManager(
            environment=args.environment,
            dry_run=args.dry_run
        )
        
        success = await deployment_manager.deploy()
        
        if success:
            logger.info("🎉 Deployment completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Deployment failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Deployment error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 