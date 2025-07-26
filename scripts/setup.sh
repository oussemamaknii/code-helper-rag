#!/bin/bash

# Python Code Helper RAG System - Development Setup Script
# This script sets up the development environment for the project

set -e  # Exit on any error

echo "🚀 Setting up Python Code Helper RAG System development environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.11+ is installed
check_python() {
    print_status "Checking Python version..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 11 ]; then
            print_success "Python $PYTHON_VERSION found"
            PYTHON_CMD="python3"
        else
            print_error "Python 3.11+ required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.11+"
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    
    if [ ! -d "venv" ]; then
        $PYTHON_CMD -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
}

# Activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        print_success "Virtual environment activated"
    elif [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
        print_success "Virtual environment activated (Windows)"
    else
        print_error "Could not find virtual environment activation script"
        exit 1
    fi
}

# Install dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install main dependencies
    pip install -r requirements.txt
    
    # Install development dependencies
    pip install -e ".[dev,evaluation]"
    
    print_success "Dependencies installed"
}

# Create environment file
setup_env_file() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        cp env.template .env
        print_success "Environment file created from template"
        print_warning "Please edit .env file with your API keys and configuration"
        print_warning "Required: OPENAI_API_KEY, GITHUB_TOKEN, PINECONE_API_KEY"
    else
        print_warning ".env file already exists"
    fi
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    directories=("logs" "data" "cache" "temp")
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_success "Created directory: $dir"
        fi
    done
}

# Run basic tests to verify setup
run_tests() {
    print_status "Running basic tests to verify setup..."
    
    # Create a minimal .env for testing if it doesn't exist
    if [ ! -f ".env" ]; then
        cat > .env << EOL
OPENAI_API_KEY=test_key
GITHUB_TOKEN=test_token
PINECONE_API_KEY=test_pinecone_key
EOL
        print_status "Created minimal .env for testing"
    fi
    
    # Run basic import test
    if python -c "from src.config.settings import settings; print('✓ Configuration loading works')"; then
        print_success "Basic configuration test passed"
    else
        print_error "Configuration test failed"
        exit 1
    fi
    
    # Run unit tests if pytest is available
    if command -v pytest &> /dev/null; then
        print_status "Running unit tests..."
        if pytest tests/unit/ -v --tb=short; then
            print_success "Unit tests passed"
        else
            print_warning "Some unit tests failed, but setup is complete"
        fi
    else
        print_warning "pytest not available, skipping unit tests"
    fi
}

# Pre-commit hooks setup
setup_pre_commit() {
    print_status "Setting up code quality tools..."
    
    # Install pre-commit if available
    if command -v pre-commit &> /dev/null; then
        pre-commit install
        print_success "Pre-commit hooks installed"
    else
        print_warning "pre-commit not available, skipping hooks setup"
    fi
}

# Main setup process
main() {
    echo "=========================================="
    echo "Python Code Helper RAG System Setup"
    echo "=========================================="
    
    check_python
    create_venv
    activate_venv
    install_dependencies
    setup_env_file
    create_directories
    setup_pre_commit
    run_tests
    
    echo ""
    echo "=========================================="
    print_success "Setup completed successfully! 🎉"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Edit .env file with your API keys"
    echo "2. Activate virtual environment: source venv/bin/activate"
    echo "3. Run the application: uvicorn src.api.main:app --reload"
    echo ""
    echo "For more information, see README.md"
}

# Run main function
main 