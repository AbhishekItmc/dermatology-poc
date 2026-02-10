#!/bin/bash

# Dermatological Analysis PoC - Setup Script

set -e

echo "========================================="
echo "Dermatological Analysis PoC Setup"
echo "========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✓ Docker and Docker Compose are installed"
echo ""

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "✓ .env file created"
    echo "⚠ Please update .env with your configuration"
else
    echo "✓ .env file already exists"
fi
echo ""

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p storage models
echo "✓ Directories created"
echo ""

# Build Docker images
echo "Building Docker images..."
docker-compose build
echo "✓ Docker images built"
echo ""

# Start services
echo "Starting services..."
docker-compose up -d
echo "✓ Services started"
echo ""

# Wait for services to be healthy
echo "Waiting for services to be ready..."
sleep 10

# Check service health
echo "Checking service health..."
if docker-compose ps | grep -q "Up"; then
    echo "✓ Services are running"
else
    echo "⚠ Some services may not be running properly"
    docker-compose ps
fi
echo ""

echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Access the application:"
echo "  Frontend:      http://localhost:3000"
echo "  Backend API:   http://localhost:8000"
echo "  API Docs:      http://localhost:8000/api/v1/docs"
echo "  Celery Flower: http://localhost:5555"
echo ""
echo "Useful commands:"
echo "  make logs          - View logs"
echo "  make down          - Stop services"
echo "  make test          - Run tests"
echo "  make help          - Show all commands"
echo ""
