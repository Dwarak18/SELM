# SELM Deployment Script for Windows PowerShell

param(
    [string]$Environment = "local",
    [switch]$Build = $true,
    [switch]$Push = $false,
    [string]$Registry = "",
    [string]$Tag = "latest",
    [switch]$Help
)

# Colors for output
$Colors = @{
    Info = "Cyan"
    Success = "Green"
    Warning = "Yellow"
    Error = "Red"
}

function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor $Colors.Info
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor $Colors.Success
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor $Colors.Warning
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $Colors.Error
}

function Show-Usage {
    Write-Host "Usage: .\deploy.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Environment    Deployment environment (local, docker, k8s)"
    Write-Host "  -Build          Build Docker image (default: true)"
    Write-Host "  -Push           Push image to registry (default: false)"
    Write-Host "  -Registry       Docker registry URL"
    Write-Host "  -Tag            Image tag (default: latest)"
    Write-Host "  -Help           Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\deploy.ps1 -Environment local"
    Write-Host "  .\deploy.ps1 -Environment docker -Build -Tag v1.0.0"
    Write-Host "  .\deploy.ps1 -Environment k8s -Push -Registry myregistry.com -Tag v1.0.0"
}

function Test-Prerequisites {
    Write-Status "Checking prerequisites..."
    
    # Check Python
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        Write-Error "Python is required but not installed"
        exit 1
    }
    
    # Check Docker for docker/k8s environments
    if ($Environment -eq "docker" -or $Environment -eq "k8s") {
        if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
            Write-Error "Docker is required for $Environment deployment"
            exit 1
        }
    }
    
    # Check kubectl for k8s environment
    if ($Environment -eq "k8s") {
        if (-not (Get-Command kubectl -ErrorAction SilentlyContinue)) {
            Write-Error "kubectl is required for Kubernetes deployment"
            exit 1
        }
    }
    
    Write-Success "Prerequisites check passed"
}

function Deploy-Local {
    Write-Status "Deploying to local environment..."
    
    # Check if virtual environment exists
    if (-not (Test-Path "venv")) {
        Write-Status "Creating virtual environment..."
        python -m venv venv
    }
    
    # Activate virtual environment
    if (Test-Path "venv\Scripts\Activate.ps1") {
        & "venv\Scripts\Activate.ps1"
    } else {
        Write-Error "Failed to activate virtual environment"
        exit 1
    }
    
    # Install dependencies
    Write-Status "Installing dependencies..."
    pip install -r requirements.txt
    
    # Create .env file if it doesn't exist
    if (-not (Test-Path ".env")) {
        Write-Status "Creating .env file..."
        Copy-Item ".env.example" ".env"
        Write-Warning "Please update .env file with your configuration"
    }
    
    # Initialize system
    Write-Status "Initializing system..."
    python scripts\initialize_system.py
    
    Write-Success "Local deployment completed!"
    Write-Status "To start the API server, run: python -m src.api.main"
}

function Deploy-Docker {
    Write-Status "Deploying with Docker..."
    
    # Build image if requested
    if ($Build) {
        Write-Status "Building Docker image..."
        
        $ImageName = "selm"
        if ($Registry) {
            $ImageName = "$Registry/selm"
        }
        
        docker build -t "${ImageName}:$Tag" .
        
        # Push image if requested
        if ($Push) {
            Write-Status "Pushing image to registry..."
            docker push "${ImageName}:$Tag"
        }
    }
    
    # Create .env file if it doesn't exist
    if (-not (Test-Path ".env")) {
        Write-Status "Creating .env file..."
        Copy-Item ".env.example" ".env"
        Write-Warning "Please update .env file with your configuration"
    }
    
    # Start services with Docker Compose
    Write-Status "Starting services with Docker Compose..."
    docker-compose up -d
    
    # Wait for services to be ready
    Write-Status "Waiting for services to be ready..."
    Start-Sleep -Seconds 30
    
    # Check service health
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 10
        if ($response.StatusCode -eq 200) {
            Write-Success "Docker deployment completed! API is available at http://localhost:8000"
        }
    } catch {
        Write-Warning "Services started but health check failed. Check logs with: docker-compose logs"
    }
}

function Deploy-K8s {
    Write-Status "Deploying to Kubernetes..."
    
    # Check if image needs to be built and pushed
    if ($Build -and -not $Registry) {
        Write-Error "Registry URL is required for Kubernetes deployment"
        exit 1
    }
    
    # Build and push image if requested
    if ($Build) {
        Write-Status "Building and pushing Docker image..."
        
        $ImageName = "$Registry/selm"
        docker build -t "${ImageName}:$Tag" .
        docker push "${ImageName}:$Tag"
        
        # Update image tag in Kubernetes manifests
        (Get-Content "kubernetes\03-api.yaml") -replace "image: selm:latest", "image: ${ImageName}:$Tag" | Set-Content "kubernetes\03-api.yaml"
        (Get-Content "kubernetes\04-training-worker.yaml") -replace "image: selm:latest", "image: ${ImageName}:$Tag" | Set-Content "kubernetes\04-training-worker.yaml"
    }
    
    # Apply Kubernetes manifests
    Write-Status "Applying Kubernetes manifests..."
    kubectl apply -f kubernetes\
    
    # Wait for deployments to be ready
    Write-Status "Waiting for deployments to be ready..."
    kubectl rollout status deployment/selm-api -n selm-system --timeout=300s
    kubectl rollout status deployment/selm-redis -n selm-system --timeout=300s
    
    # Get service information
    Write-Status "Getting service information..."
    kubectl get services -n selm-system
    
    Write-Success "Kubernetes deployment completed!"
    Write-Status "Use 'kubectl port-forward svc/selm-api-service 8000:8000 -n selm-system' to access the API locally"
}

# Main script logic
if ($Help) {
    Show-Usage
    exit 0
}

# Validate environment
if ($Environment -notin @("local", "docker", "k8s")) {
    Write-Error "Invalid environment. Must be one of: local, docker, k8s"
    exit 1
}

Write-Status "Starting SELM deployment for environment: $Environment"

Test-Prerequisites

switch ($Environment) {
    "local" { Deploy-Local }
    "docker" { Deploy-Docker }
    "k8s" { Deploy-K8s }
}

Write-Success "SELM deployment completed for environment: $Environment"
