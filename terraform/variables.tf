variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
}

variable "environment" {
  description = "Deployment environment (dev|staging|prod)"
  type        = string
}

variable "project_name" {
  description = "Project name prefix"
  type        = string
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
}

variable "private_subnet_cidrs" {
  description = "List of CIDR blocks for private subnets"
  type        = list(string)
}

variable "public_subnet_cidrs" {
  description = "List of CIDR blocks for public subnets"
  type        = list(string)
}

variable "database_subnet_cidrs" {
  description = "List of CIDR blocks for database subnets"
  type        = list(string)
}

variable "kubernetes_version" {
  description = "EKS Kubernetes version"
  type        = string
}

variable "node_instance_types" {
  description = "List of EC2 instance types for EKS nodes"
  type        = list(string)
}

variable "node_group_min_size" {
  description = "Minimum number of nodes in EKS managed node group"
  type        = number
}

variable "node_group_max_size" {
  description = "Maximum number of nodes in EKS managed node group"
  type        = number
}

variable "node_group_desired_size" {
  description = "Desired number of nodes in EKS managed node group"
  type        = number
}

variable "spot_instance_types" {
  description = "List of EC2 spot instance types for EKS"
  type        = list(string)
}

variable "postgres_version" {
  description = "PostgreSQL engine version"
  type        = string
}

variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
}

variable "rds_allocated_storage" {
  description = "Initial allocated storage for RDS (GiB)"
  type        = number
}

variable "rds_max_allocated_storage" {
  description = "Maximum allocated storage for RDS (GiB)"
  type        = number
}

variable "database_name" {
  description = "Name of the application database"
  type        = string
}

variable "database_username" {
  description = "Master username for RDS"
  type        = string
}

variable "database_password" {
  description = "Master password for RDS"
  type        = string
  sensitive   = true
}

variable "redis_node_type" {
  description = "ElastiCache node type"
  type        = string
}

variable "redis_num_cache_nodes" {
  description = "Number of Redis cache nodes"
  type        = number
}

variable "redis_auth_token" {
  description = "Redis AUTH token"
  type        = string
  sensitive   = true
}

variable "log_retention_days" {
  description = "Log retention in CloudWatch (days)"
  type        = number
}

variable "domain_name" {
  description = "Optional Route53 domain name"
  type        = string
  default     = ""
} 