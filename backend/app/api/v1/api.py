"""
API v1 router aggregation
"""
from fastapi import APIRouter

from app.api.v1.endpoints import patients, analyses, simulations, auth

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(patients.router, prefix="/patients", tags=["patients"])
api_router.include_router(analyses.router, prefix="/analyses", tags=["analyses"])
api_router.include_router(simulations.router, prefix="/simulations", tags=["simulations"])
