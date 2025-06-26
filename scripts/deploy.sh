#!/bin/bash

# SELM Deployment Script
# This script helps deploy SELM to different environments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="local"
BUILD_IMAGE=true
PUSH_IMAGE=false
REGISTRY=""
TAG="latest"

# Print colored output
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

# Show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -e, --environment    Deployment environment (local, docker, k8s)"
    echo "  -b, --build          Build Docker image (default: true)"
    echo "  -p, --push           Push image to registry (default: false)"
    echo "  -r, --registry       Docker registry URL"
    echo "  -t, --tag            Image tag (default: latest)"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --environment local"
    echo "  $0 --environment docker --build --tag v1.0.0"
    echo "  $0 --environment k8s --push --registry myregistry.com --tag v1.0.0"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -b|--build)
            BUILD_IMAGE=true
            shift
            ;;
        --no-build)
            BUILD_IMAGE=false
            shift
            ;;
        -p|--push)
            PUSH_IMAGE=true
            shift
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(local|docker|k8s)$ ]]; then
    print_error "Invalid environment. Must be one of: local, docker, k8s"
    exit 1
fi

print_status "Starting SELM deployment for environment: $ENVIRONMENT"

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check Docker for docker/k8s environments
    if [[ "$ENVIRONMENT" == "docker" || "$ENVIRONMENT" == "k8s" ]]; then
        if ! command -v docker &> /dev/null; then
            print_error "Docker is required for $ENVIRONMENT deployment"
            exit 1
        fi
    fi
    
    # Check kubectl for k8s environment
    if [[ "$ENVIRONMENT" == "k8s" ]]; then
        if ! command -v kubectl &> /dev/null; then
            print_error "kubectl is required for Kubernetes deployment"
            exit 1
        fi
    fi
    
    print_success "Prerequisites check passed"
}

# Deploy to local environment
deploy_local() {
    print_status "Deploying to local environment..."
    
    # Check if virtual environment exists
    if [[ ! -d "venv" ]]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate || source venv/Scripts/activate
    
    # Install dependencies
    print_status "Installing dependencies..."
    pip install -r requirements.txt
    
    # Create .env file if it doesn't exist
    if [[ ! -f ".env" ]]; then
        print_status "Creating .env file..."
        cp .env.example .env
        print_warning "Please update .env file with your configuration"
    fi
    
    # Initialize system
    print_status "Initializing system..."
    python scripts/initialize_system.py
    
    print_success "Local deployment completed!"
    print_status "To start the API server, run: python -m src.api.main"
}

# Deploy with Docker
deploy_docker() {
    print_status "Deploying with Docker..."
    
    # Build image if requested
    if [[ "$BUILD_IMAGE" == true ]]; then
        print_status "Building Docker image..."
        
        IMAGE_NAME="selm"
        if [[ -n "$REGISTRY" ]]; then
            IMAGE_NAME="$REGISTRY/selm"
        fi
        
        docker build -t "$IMAGE_NAME:$TAG" .
        
        # Push image if requested
        if [[ "$PUSH_IMAGE" == true ]]; then
            print_status "Pushing image to registry..."
            docker push "$IMAGE_NAME:$TAG"
        fi
    fi
    
    # Create .env file if it doesn't exist
    if [[ ! -f ".env" ]]; then
        print_status "Creating .env file..."
        cp .env.example .env
        print_warning "Please update .env file with your configuration"
    fi
    
    # Start services with Docker Compose
    print_status "Starting services with Docker Compose..."
    docker-compose up -d
    
    # Wait for services to be ready
    print_status "Waiting for services to be ready..."
    sleep 30
    
    # Check service health
    if curl -f http://localhost:8000/health &> /dev/null; then
        print_success "Docker deployment completed! API is available at http://localhost:8000"
    else
        print_warning "Services started but health check failed. Check logs with: docker-compose logs"
    fi
}

# Deploy to Kubernetes
deploy_k8s() {
    print_status "Deploying to Kubernetes..."
    
    # Check if image needs to be built and pushed
    if [[ "$BUILD_IMAGE" == true && -z "$REGISTRY" ]]; then
        print_error "Registry URL is required for Kubernetes deployment"
        exit 1
    fi
    
    # Build and push image if requested
    if [[ "$BUILD_IMAGE" == true ]]; then
        print_status "Building and pushing Docker image..."
        
        IMAGE_NAME="$REGISTRY/selm"
        docker build -t "$IMAGE_NAME:$TAG" .
        docker push "$IMAGE_NAME:$TAG"
        
        # Update image tag in Kubernetes manifests
        sed -i.bak "s|image: selm:latest|image: $IMAGE_NAME:$TAG|g" kubernetes/03-api.yaml
        sed -i.bak "s|image: selm:latest|image: $IMAGE_NAME:$TAG|g" kubernetes/04-training-worker.yaml
    fi
    
    # Apply Kubernetes manifests
    print_status "Applying Kubernetes manifests..."
    kubectl apply -f kubernetes/
    
    # Wait for deployments to be ready
    print_status "Waiting for deployments to be ready..."
    kubectl rollout status deployment/selm-api -n selm-system --timeout=300s
    kubectl rollout status deployment/selm-redis -n selm-system --timeout=300s
    
    # Get service information
    print_status "Getting service information..."
    kubectl get services -n selm-system
    
    print_success "Kubernetes deployment completed!"
    print_status "Use 'kubectl port-forward svc/selm-api-service 8000:8000 -n selm-system' to access the API locally"
}

# Main deployment logic
main() {
    check_prerequisites
    
    case $ENVIRONMENT in
        "local")
            deploy_local
            ;;
        "docker")
            deploy_docker
            ;;
        "k8s")
            deploy_k8s
            ;;
    esac
    
    print_success "SELM deployment completed for environment: $ENVIRONMENT"
}

# Run main function
main
