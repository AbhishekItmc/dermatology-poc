/**
 * API service for backend communication
 */
import axios, { AxiosInstance } from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

class ApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 60000, // 60 seconds
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add request interceptor for authentication
    this.client.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('access_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Add response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          // Handle unauthorized access
          localStorage.removeItem('access_token');
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );
  }

  // Patient endpoints
  async uploadPatientImages(patientId: string, images: File[]) {
    const formData = new FormData();
    images.forEach((image) => {
      formData.append('images', image);
    });

    return this.client.post(`/patients/${patientId}/images`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  }

  async listPatientImages(patientId: string) {
    return this.client.get(`/patients/${patientId}/images`);
  }

  // Analysis endpoints
  async createAnalysis(patientId: string, imageSetId: string) {
    return this.client.post('/analyses', { patient_id: patientId, image_set_id: imageSetId });
  }

  async getAnalysis(analysisId: string) {
    return this.client.get(`/analyses/${analysisId}`);
  }

  async getAnalysisStatus(analysisId: string) {
    return this.client.get(`/analyses/${analysisId}/status`);
  }

  async getAnalysisMesh(analysisId: string) {
    return this.client.get(`/analyses/${analysisId}/mesh`);
  }

  async getAnalysisTexture(analysisId: string) {
    return this.client.get(`/analyses/${analysisId}/texture`);
  }

  async getAnalysisAnomalies(analysisId: string) {
    return this.client.get(`/analyses/${analysisId}/anomalies`);
  }

  async getRecommendations(analysisId: string) {
    return this.client.get(`/analyses/${analysisId}/recommendations`);
  }

  // Simulation endpoints
  async createSimulation(analysisId: string, treatmentType: string, parameters: any) {
    return this.client.post('/simulations', {
      analysis_id: analysisId,
      treatment_type: treatmentType,
      parameters,
    });
  }

  async getSimulation(simulationId: string) {
    return this.client.get(`/simulations/${simulationId}`);
  }

  async getSimulationTimeline(simulationId: string) {
    return this.client.get(`/simulations/${simulationId}/timeline`);
  }
}

export default new ApiService();
