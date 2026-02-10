# Setup Guide - Dermatological Analysis PoC

This guide provides detailed instructions for setting up the development environment.

## Prerequisites

### Required Software

1. **Docker Desktop** (v20.10+)
   - Download: https://www.docker.com/products/docker-desktop
   - Includes Docker and Docker Compose

2. **Git**
   - Download: https://git-scm.com/downloads

### Optional (for local development without Docker)

3. **Python 3.10+**
   - Download: https://www.python.org/downloads/

4. **Node.js 20+**
   - Download: https://nodejs.org/

## Quick Setup (Recommended)

### 1. Clone the Repository

```bash
git clone <repository-url>
cd dermatological-analysis-poc
```

### 2. Run Setup Script

On Linux/Mac:
```bash
chmod +x setup.sh
./setup.sh
```

On Windows (PowerShell):
```powershell
# Run commands manually from setup.sh
```

### 3. Verify Installation

Check that all services are running:
```bash
docker-compose ps
```

You should see:
- dermatology-postgres (healthy)
- dermatology-redis (healthy)
- dermatology-backend (running)
- dermatology-frontend (running)
- dermatology-celery-worker (running)
- dermatology-celery-flower (running)

### 4. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/api/v1/docs
- **Celery Flower**: http://localhost:5555

## Manual Setup

### 1. Environment Configuration

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` and update the following:
- `SECRET_KEY`: Generate a secure random key
- `ENCRYPTION_KEY`: Generate a 32-byte encryption key
- Other settings as needed

### 2. Create Required Directories

```bash
mkdir -p storage models
```

### 3. Build Docker Images

```bash
docker-compose build
```

This will:
- Build the backend Python image
- Build the frontend Node.js image
- Pull PostgreSQL and Redis images

### 4. Start Services

```bash
docker-compose up -d
```

### 5. Initialize Database

```bash
# Run database migrations (when implemented)
docker-compose exec backend alembic upgrade head
```

## Local Development Setup (Without Docker)

### Backend Setup

1. **Create Virtual Environment**
```bash
cd backend
python -m venv venv
```

2. **Activate Virtual Environment**

Linux/Mac:
```bash
source venv/bin/activate
```

Windows:
```bash
venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Start PostgreSQL and Redis**
```bash
docker-compose up -d postgres redis
```

5. **Run Backend Server**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

1. **Install Dependencies**
```bash
cd frontend
npm install
```

2. **Start Development Server**
```bash
npm start
```

The frontend will be available at http://localhost:3000

## Verification Steps

### 1. Check Backend Health

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "environment": "development"
}
```

### 2. Check API Documentation

Visit http://localhost:8000/api/v1/docs

You should see the Swagger UI with all API endpoints.

### 3. Check Frontend

Visit http://localhost:3000

You should see the application homepage.

### 4. Check Database Connection

```bash
docker-compose exec postgres psql -U postgres -d dermatology_poc -c "SELECT version();"
```

### 5. Check Redis Connection

```bash
docker-compose exec redis redis-cli ping
```

Expected response: `PONG`

## Common Issues and Solutions

### Issue: Port Already in Use

**Error**: `Bind for 0.0.0.0:8000 failed: port is already allocated`

**Solution**:
```bash
# Find process using the port
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows

# Kill the process or change the port in docker-compose.yml
```

### Issue: Docker Build Fails

**Error**: Various build errors

**Solution**:
```bash
# Clean Docker cache and rebuild
docker-compose down -v
docker system prune -a
docker-compose build --no-cache
```

### Issue: Permission Denied on setup.sh

**Error**: `Permission denied: ./setup.sh`

**Solution**:
```bash
chmod +x setup.sh
```

### Issue: Database Connection Failed

**Error**: `could not connect to server`

**Solution**:
```bash
# Restart PostgreSQL
docker-compose restart postgres

# Check logs
docker-compose logs postgres
```

## Development Workflow

### Starting Work

```bash
# Start all services
make up

# Or manually
docker-compose up -d
```

### Viewing Logs

```bash
# All services
make logs

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Running Tests

```bash
# All tests
make test

# Backend only
make test-backend

# Frontend only
make test-frontend
```

### Stopping Services

```bash
# Stop all services
make down

# Or manually
docker-compose down
```

### Cleaning Up

```bash
# Remove all containers and volumes
make clean

# Or manually
docker-compose down -v
```

## IDE Setup

### VS Code

Recommended extensions:
- Python
- Pylance
- ESLint
- Prettier
- Docker
- GitLens

### PyCharm

1. Configure Python interpreter to use the virtual environment
2. Enable Docker integration
3. Configure code style to match project settings

## Next Steps

After successful setup:

1. **Review the Requirements**: Read `.kiro/specs/dermatological-analysis-poc/requirements.md`
2. **Review the Design**: Read `.kiro/specs/dermatological-analysis-poc/design.md`
3. **Review the Tasks**: Read `.kiro/specs/dermatological-analysis-poc/tasks.md`
4. **Start Development**: Begin with task 2 (Image Preprocessing Module)

## Getting Help

- Check the main README.md for project overview
- Review API documentation at http://localhost:8000/api/v1/docs
- Check Docker logs for error messages
- Ensure all prerequisites are installed correctly

## Security Notes

⚠️ **Important**: 
- Never commit `.env` file to version control
- Change default passwords in production
- Use strong encryption keys
- Enable HTTPS in production
- Follow security best practices for healthcare data

## Performance Notes

- GPU acceleration requires NVIDIA Docker runtime
- Adjust `BATCH_SIZE` in `.env` based on available GPU memory
- Monitor resource usage with `docker stats`
- Scale services as needed for production load
