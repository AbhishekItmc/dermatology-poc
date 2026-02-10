# Dermatological Analysis PoC

AI-driven Dermatological Analysis Platform for detecting and analyzing skin conditions with 3D visualization and treatment simulation.

## Features

- **AI Detection Engine**: Deep learning models for pigmentation and wrinkle detection
- **3D Reconstruction**: Multi-view facial reconstruction from 180-degree image sets
- **Interactive 3D Viewer**: Real-time visualization with anomaly highlighting
- **Treatment Simulation**: Predictive modeling for aesthetic outcome visualization
- **Clinical Dashboard**: Comprehensive reporting and analysis tools

## Technology Stack

### Backend
- Python 3.10+
- FastAPI
- PyTorch
- OpenCV
- PostgreSQL
- Redis
- Celery

### Frontend
- TypeScript
- React
- Three.js
- WebGL

### Infrastructure
- Docker
- Docker Compose
- GitHub Actions

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.10+ (for local development)
- Node.js 20+ (for local development)

### Quick Start with Docker

1. Clone the repository:
```bash
git clone <repository-url>
cd dermatological-analysis-poc
```

2. Copy environment variables:
```bash
cp .env.example .env
```

3. Start all services:
```bash
docker-compose up -d
```

4. Access the application:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/api/v1/docs
- Celery Flower: http://localhost:5555

### Local Development

#### Backend Setup

1. Navigate to backend directory:
```bash
cd backend
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start PostgreSQL and Redis (via Docker):
```bash
docker-compose up -d postgres redis
```

5. Run the development server:
```bash
uvicorn app.main:app --reload
```

#### Frontend Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

## Project Structure

```
.
├── backend/                 # Python FastAPI backend
│   ├── app/
│   │   ├── api/            # API endpoints
│   │   ├── core/           # Core configuration
│   │   ├── models/         # Database models
│   │   ├── services/       # Business logic
│   │   └── main.py         # Application entry point
│   ├── tests/              # Backend tests
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/               # React TypeScript frontend
│   ├── public/
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── services/      # API services
│   │   ├── types/         # TypeScript types
│   │   └── App.tsx        # Main application
│   ├── Dockerfile
│   └── package.json
├── .github/
│   └── workflows/         # CI/CD pipelines
├── docker-compose.yml     # Docker orchestration
├── .env.example          # Environment variables template
└── README.md             # This file
```

## Testing

### Backend Tests

Run unit tests:
```bash
cd backend
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ -v --cov=app --cov-report=html
```

Run property-based tests:
```bash
pytest tests/ -v -m property
```

### Frontend Tests

Run tests:
```bash
cd frontend
npm test
```

Run with coverage:
```bash
npm test -- --coverage
```

## API Documentation

Once the backend is running, visit:
- Swagger UI: http://localhost:8000/api/v1/docs
- ReDoc: http://localhost:8000/api/v1/redoc

## Requirements

This project implements the requirements specified in:
- `.kiro/specs/dermatological-analysis-poc/requirements.md`
- `.kiro/specs/dermatological-analysis-poc/design.md`
- `.kiro/specs/dermatological-analysis-poc/tasks.md`

## Security

- All patient data is encrypted at rest (AES-256)
- TLS 1.2+ for data transmission
- Role-based access control (RBAC)
- Comprehensive audit logging
- HIPAA and GDPR compliant

## Performance

- Analysis processing: < 60 seconds
- 3D rendering: 30+ FPS
- Concurrent sessions: 10+ users
- GPU-accelerated inference

## License

[License information]

## Contributing

[Contributing guidelines]

## Support

For issues and questions, please contact [support contact].
