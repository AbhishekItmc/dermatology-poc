.PHONY: help build up down logs test clean

help:
	@echo "Dermatological Analysis PoC - Development Commands"
	@echo ""
	@echo "Available commands:"
	@echo "  make build       - Build all Docker containers"
	@echo "  make up          - Start all services"
	@echo "  make down        - Stop all services"
	@echo "  make logs        - View logs from all services"
	@echo "  make test        - Run all tests"
	@echo "  make test-backend - Run backend tests"
	@echo "  make test-frontend - Run frontend tests"
	@echo "  make clean       - Remove all containers and volumes"
	@echo "  make shell-backend - Open shell in backend container"
	@echo "  make shell-frontend - Open shell in frontend container"

build:
	docker-compose build

up:
	docker-compose up -d
	@echo "Services started!"
	@echo "Frontend: http://localhost:3000"
	@echo "Backend API: http://localhost:8000"
	@echo "API Docs: http://localhost:8000/api/v1/docs"
	@echo "Celery Flower: http://localhost:5555"

down:
	docker-compose down

logs:
	docker-compose logs -f

test: test-backend test-frontend

test-backend:
	docker-compose exec backend pytest tests/ -v

test-frontend:
	docker-compose exec frontend npm test -- --watchAll=false

clean:
	docker-compose down -v
	rm -rf backend/__pycache__
	rm -rf backend/.pytest_cache
	rm -rf frontend/node_modules
	rm -rf frontend/build

shell-backend:
	docker-compose exec backend /bin/bash

shell-frontend:
	docker-compose exec frontend /bin/sh

restart:
	docker-compose restart

ps:
	docker-compose ps
