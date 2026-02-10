# Project Structure

This document provides an overview of the project structure and organization.

## Directory Structure

```
dermatological-analysis-poc/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI/CD pipeline
├── .kiro/
│   └── specs/
│       └── dermatological-analysis-poc/
│           ├── requirements.md    # Project requirements
│           ├── design.md          # Design document
│           └── tasks.md           # Implementation tasks
├── backend/                       # Python FastAPI backend
│   ├── app/
│   │   ├── api/
│   │   │   └── v1/
│   │   │       ├── endpoints/
│   │   │       │   ├── patients.py      # Patient management endpoints
│   │   │       │   ├── analyses.py      # Analysis endpoints
│   │   │       │   └── simulations.py   # Treatment simulation endpoints
│   │   │       └── api.py               # API router aggregation
│   │   ├── core/
│   │   │   └── config.py                # Application configuration
│   │   ├── models/                      # Database models (to be implemented)
│   │   ├── services/                    # Business logic services (to be implemented)
│   │   ├── __init__.py
│   │   ├── main.py                      # FastAPI application entry point
│   │   └── worker.py                    # Celery worker configuration
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py                  # Pytest fixtures
│   │   └── test_main.py                 # Main application tests
│   ├── Dockerfile                       # Backend Docker image
│   ├── pytest.ini                       # Pytest configuration
│   └── requirements.txt                 # Python dependencies
├── frontend/                      # React TypeScript frontend
│   ├── public/
│   │   └── index.html                   # HTML template
│   ├── src/
│   │   ├── services/
│   │   │   └── api.ts                   # API service for backend communication
│   │   ├── types/
│   │   │   └── index.ts                 # TypeScript type definitions
│   │   ├── App.css                      # Application styles
│   │   ├── App.tsx                      # Main application component
│   │   ├── index.css                    # Global styles
│   │   └── index.tsx                    # Application entry point
│   ├── Dockerfile                       # Frontend Docker image
│   ├── package.json                     # Node.js dependencies
│   └── tsconfig.json                    # TypeScript configuration
├── models/                        # AI model storage
│   └── .gitkeep
├── storage/                       # Patient data storage
│   └── .gitkeep
├── .env.example                   # Environment variables template
├── .gitignore                     # Git ignore rules
├── CONTRIBUTING.md                # Contributing guidelines
├── docker-compose.yml             # Docker orchestration
├── Makefile                       # Development commands
├── PROJECT_STRUCTURE.md           # This file
├── README.md                      # Project overview
├── SETUP.md                       # Setup instructions
└── setup.sh                       # Setup script
```

## Component Overview

### Backend (Python/FastAPI)

**Purpose**: AI detection engine, 3D reconstruction, and API services

**Key Technologies**:
- FastAPI: Web framework
- PyTorch: Deep learning
- OpenCV: Computer vision
- PostgreSQL: Database
- Redis: Caching
- Celery: Task queue

**Main Components**:
- `app/main.py`: Application entry point with middleware and routing
- `app/core/config.py`: Configuration management with environment variables
- `app/api/v1/`: API endpoints organized by resource
- `app/models/`: SQLAlchemy database models (to be implemented)
- `app/services/`: Business logic and AI services (to be implemented)
- `app/worker.py`: Celery worker for async processing

**API Endpoints**:
- `/health`: Health check
- `/api/v1/patients/{id}/images`: Patient image management
- `/api/v1/analyses`: Analysis creation and retrieval
- `/api/v1/simulations`: Treatment simulations

### Frontend (React/TypeScript)

**Purpose**: Clinical dashboard and 3D visualization

**Key Technologies**:
- React: UI framework
- TypeScript: Type safety
- Three.js: 3D rendering
- Axios: HTTP client

**Main Components**:
- `src/App.tsx`: Main application component
- `src/services/api.ts`: Backend API client
- `src/types/index.ts`: TypeScript type definitions
- Components (to be implemented):
  - Dashboard
  - 3D Viewer
  - Analysis Results
  - Treatment Controls

### Infrastructure

**Docker Services**:
1. **postgres**: PostgreSQL database
2. **redis**: Redis cache
3. **backend**: FastAPI application
4. **celery-worker**: Background task processor
5. **celery-flower**: Task monitoring UI
6. **frontend**: React development server

**Volumes**:
- `postgres_data`: Database persistence
- `redis_data`: Cache persistence
- `./storage`: Patient data storage
- `./models`: AI model storage

### CI/CD Pipeline

**GitHub Actions Workflow** (`.github/workflows/ci.yml`):

1. **Backend Tests**:
   - Python linting (flake8, black)
   - Unit tests with pytest
   - Code coverage reporting

2. **Frontend Tests**:
   - TypeScript linting
   - Unit tests with Jest
   - Build verification

3. **Docker Build**:
   - Build backend image
   - Build frontend image
   - Cache optimization

4. **Property-Based Tests** (nightly):
   - Run Hypothesis tests
   - Extended test coverage

## Development Workflow

### Local Development

1. **Start Services**:
   ```bash
   make up
   ```

2. **View Logs**:
   ```bash
   make logs
   ```

3. **Run Tests**:
   ```bash
   make test
   ```

4. **Stop Services**:
   ```bash
   make down
   ```

### Adding New Features

1. Create feature branch
2. Implement changes
3. Add tests
4. Update documentation
5. Create pull request
6. Pass CI checks
7. Get code review
8. Merge to develop

## Configuration

### Environment Variables

Key configuration in `.env`:
- Database credentials
- Redis connection
- Storage configuration
- AI model settings
- Security keys
- Performance limits

See `.env.example` for all available options.

### Docker Compose

Services configured in `docker-compose.yml`:
- Port mappings
- Volume mounts
- Environment variables
- Health checks
- Dependencies

## Testing Strategy

### Unit Tests
- Test individual functions/components
- Mock external dependencies
- Fast execution
- Run on every commit

### Property-Based Tests
- Test universal properties
- Generate random inputs
- Comprehensive coverage
- Run nightly

### Integration Tests
- Test component interactions
- Use real dependencies
- End-to-end workflows
- Run on pull requests

## Security

### Data Protection
- Encryption at rest (AES-256)
- TLS for transmission
- Secure key management
- Audit logging

### Access Control
- JWT authentication
- Role-based authorization
- Session management
- Rate limiting

### Compliance
- HIPAA requirements
- GDPR requirements
- Regular security audits
- Vulnerability scanning

## Performance

### Backend
- Async I/O operations
- Redis caching
- Database query optimization
- GPU acceleration

### Frontend
- Code splitting
- Lazy loading
- Bundle optimization
- WebGL rendering

### Targets
- Analysis: < 60 seconds
- Rendering: 30+ FPS
- Concurrent users: 10+
- API response: < 2 seconds

## Future Enhancements

### Planned Features
- Additional detection models
- Advanced visualization modes
- Enhanced simulation capabilities
- Clinical integration

### Infrastructure
- Kubernetes deployment
- Auto-scaling
- Multi-region support
- CDN integration

## Resources

### Documentation
- [README.md](README.md): Project overview
- [SETUP.md](SETUP.md): Setup instructions
- [CONTRIBUTING.md](CONTRIBUTING.md): Contributing guidelines
- [Requirements](..kiro/specs/dermatological-analysis-poc/requirements.md): Detailed requirements
- [Design](..kiro/specs/dermatological-analysis-poc/design.md): System design
- [Tasks](..kiro/specs/dermatological-analysis-poc/tasks.md): Implementation tasks

### External Links
- FastAPI: https://fastapi.tiangolo.com/
- React: https://react.dev/
- Three.js: https://threejs.org/
- PyTorch: https://pytorch.org/
- Docker: https://docs.docker.com/

## Maintenance

### Regular Tasks
- Update dependencies
- Review security advisories
- Monitor performance metrics
- Backup database
- Clean up old data

### Monitoring
- Prometheus metrics
- Grafana dashboards
- Sentry error tracking
- Log aggregation (ELK)

## Support

For questions or issues:
1. Check documentation
2. Search existing issues
3. Create new issue
4. Contact team

---

Last Updated: 2024
Version: 0.1.0
