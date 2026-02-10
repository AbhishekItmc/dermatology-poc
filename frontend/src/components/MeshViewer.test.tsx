import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import MeshViewer from './MeshViewer';
import apiService from '../services/api';

// Mock the API service
jest.mock('../services/api');
const mockedApiService = apiService as jest.Mocked<typeof apiService>;

// Mock Viewer3D component
jest.mock('./Viewer3D', () => {
  return function MockViewer3D({ mesh }: any) {
    return (
      <div data-testid="viewer3d">
        {mesh && <div data-testid="mesh-loaded">Mesh Loaded</div>}
      </div>
    );
  };
});

describe('MeshViewer', () => {
  const mockAnalysisId = 'test-analysis-123';

  const mockMeshData = {
    vertices: [
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0]
    ],
    faces: [[0, 1, 2]],
    normals: [
      [0, 0, 1],
      [0, 0, 1],
      [0, 0, 1]
    ],
    uv_coords: [
      [0, 0],
      [1, 0],
      [0, 1]
    ],
    vertex_colors: [
      [255, 0, 0],
      [0, 255, 0],
      [0, 0, 255]
    ],
    vertex_labels: [0, 0, 0],
    texture_map: ''
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders loading state initially', () => {
    mockedApiService.getAnalysisStatus.mockReturnValue(
      new Promise(() => {}) // Never resolves to keep loading state
    );

    render(<MeshViewer analysisId={mockAnalysisId} />);

    expect(screen.getByText(/Loading 3D mesh/i)).toBeInTheDocument();
  });

  test('loads and displays mesh data successfully', async () => {
    mockedApiService.getAnalysisStatus.mockResolvedValue({
      data: {
        analysis_id: mockAnalysisId,
        status: 'completed',
        progress: 1.0,
        message: 'Analysis completed'
      }
    } as any);

    mockedApiService.getAnalysisMesh.mockResolvedValue({
      data: {
        analysis_id: mockAnalysisId,
        mesh: mockMeshData,
        confidence: 0.95
      }
    } as any);

    mockedApiService.getAnalysisTexture.mockResolvedValue({
      data: {
        analysis_id: mockAnalysisId,
        texture_available: true,
        message: 'Texture data embedded in mesh vertex colors'
      }
    } as any);

    render(<MeshViewer analysisId={mockAnalysisId} />);

    await waitFor(() => {
      expect(screen.getByTestId('viewer3d')).toBeInTheDocument();
    });

    expect(screen.getByTestId('mesh-loaded')).toBeInTheDocument();
  });

  test('displays error when analysis is not completed', async () => {
    mockedApiService.getAnalysisStatus.mockResolvedValue({
      data: {
        analysis_id: mockAnalysisId,
        status: 'processing',
        progress: 0.5,
        message: 'Analysis in progress'
      }
    } as any);

    render(<MeshViewer analysisId={mockAnalysisId} />);

    await waitFor(() => {
      expect(screen.getByText(/Error Loading Mesh/i)).toBeInTheDocument();
    });

    expect(screen.getByText(/Analysis not completed/i)).toBeInTheDocument();
  });

  test('displays error when mesh data is missing', async () => {
    mockedApiService.getAnalysisStatus.mockResolvedValue({
      data: {
        analysis_id: mockAnalysisId,
        status: 'completed',
        progress: 1.0,
        message: 'Analysis completed'
      }
    } as any);

    mockedApiService.getAnalysisMesh.mockResolvedValue({
      data: {
        analysis_id: mockAnalysisId,
        mesh: null
      }
    } as any);

    render(<MeshViewer analysisId={mockAnalysisId} />);

    await waitFor(() => {
      expect(screen.getByText(/Error Loading Mesh/i)).toBeInTheDocument();
    });

    expect(screen.getByText(/No mesh data available/i)).toBeInTheDocument();
  });

  test('displays error when API call fails', async () => {
    mockedApiService.getAnalysisStatus.mockRejectedValue(
      new Error('Network error')
    );

    render(<MeshViewer analysisId={mockAnalysisId} />);

    await waitFor(() => {
      expect(screen.getByText(/Error Loading Mesh/i)).toBeInTheDocument();
    });

    expect(screen.getByText(/Network error/i)).toBeInTheDocument();
  });

  test('converts vertex colors from 0-255 to 0-1 range', async () => {
    mockedApiService.getAnalysisStatus.mockResolvedValue({
      data: {
        analysis_id: mockAnalysisId,
        status: 'completed',
        progress: 1.0
      }
    } as any);

    mockedApiService.getAnalysisMesh.mockResolvedValue({
      data: {
        analysis_id: mockAnalysisId,
        mesh: mockMeshData
      }
    } as any);

    mockedApiService.getAnalysisTexture.mockResolvedValue({
      data: { texture_available: true }
    } as any);

    render(<MeshViewer analysisId={mockAnalysisId} />);

    await waitFor(() => {
      expect(screen.getByTestId('mesh-loaded')).toBeInTheDocument();
    });

    // Verify API was called
    expect(mockedApiService.getAnalysisMesh).toHaveBeenCalledWith(mockAnalysisId);
  });

  test('handles missing normals gracefully', async () => {
    const meshWithoutNormals = {
      ...mockMeshData,
      normals: []
    };

    mockedApiService.getAnalysisStatus.mockResolvedValue({
      data: {
        analysis_id: mockAnalysisId,
        status: 'completed',
        progress: 1.0
      }
    } as any);

    mockedApiService.getAnalysisMesh.mockResolvedValue({
      data: {
        analysis_id: mockAnalysisId,
        mesh: meshWithoutNormals
      }
    } as any);

    mockedApiService.getAnalysisTexture.mockResolvedValue({
      data: { texture_available: false }
    } as any);

    render(<MeshViewer analysisId={mockAnalysisId} />);

    await waitFor(() => {
      expect(screen.getByTestId('mesh-loaded')).toBeInTheDocument();
    });
  });

  test('handles missing UV coordinates gracefully', async () => {
    const meshWithoutUVs = {
      ...mockMeshData,
      uv_coords: []
    };

    mockedApiService.getAnalysisStatus.mockResolvedValue({
      data: {
        analysis_id: mockAnalysisId,
        status: 'completed',
        progress: 1.0
      }
    } as any);

    mockedApiService.getAnalysisMesh.mockResolvedValue({
      data: {
        analysis_id: mockAnalysisId,
        mesh: meshWithoutUVs
      }
    } as any);

    mockedApiService.getAnalysisTexture.mockResolvedValue({
      data: { texture_available: false }
    } as any);

    render(<MeshViewer analysisId={mockAnalysisId} />);

    await waitFor(() => {
      expect(screen.getByTestId('mesh-loaded')).toBeInTheDocument();
    });
  });

  test('handles texture loading failure gracefully', async () => {
    mockedApiService.getAnalysisStatus.mockResolvedValue({
      data: {
        analysis_id: mockAnalysisId,
        status: 'completed',
        progress: 1.0
      }
    } as any);

    mockedApiService.getAnalysisMesh.mockResolvedValue({
      data: {
        analysis_id: mockAnalysisId,
        mesh: mockMeshData
      }
    } as any);

    mockedApiService.getAnalysisTexture.mockRejectedValue(
      new Error('Texture not found')
    );

    render(<MeshViewer analysisId={mockAnalysisId} />);

    await waitFor(() => {
      expect(screen.getByTestId('mesh-loaded')).toBeInTheDocument();
    });

    // Should still load mesh even if texture fails
    expect(screen.queryByText(/Error Loading Mesh/i)).not.toBeInTheDocument();
  });

  test('validates mesh data structure', async () => {
    const invalidMesh = {
      vertices: [], // Missing vertices
      faces: []
    };

    mockedApiService.getAnalysisStatus.mockResolvedValue({
      data: {
        analysis_id: mockAnalysisId,
        status: 'completed',
        progress: 1.0
      }
    } as any);

    mockedApiService.getAnalysisMesh.mockResolvedValue({
      data: {
        analysis_id: mockAnalysisId,
        mesh: invalidMesh
      }
    } as any);

    render(<MeshViewer analysisId={mockAnalysisId} />);

    await waitFor(() => {
      expect(screen.getByText(/Error Loading Mesh/i)).toBeInTheDocument();
    });

    expect(screen.getByText(/Invalid mesh data/i)).toBeInTheDocument();
  });

  test('shows progress during loading', async () => {
    let resolveStatus: any;
    const statusPromise = new Promise((resolve) => {
      resolveStatus = resolve;
    });

    mockedApiService.getAnalysisStatus.mockReturnValue(statusPromise as any);

    render(<MeshViewer analysisId={mockAnalysisId} />);

    // Should show loading with progress
    expect(screen.getByText(/Loading 3D mesh/i)).toBeInTheDocument();
    expect(screen.getByText(/0%/i)).toBeInTheDocument();

    // Resolve the promise
    resolveStatus({
      data: {
        analysis_id: mockAnalysisId,
        status: 'completed',
        progress: 1.0
      }
    });

    mockedApiService.getAnalysisMesh.mockResolvedValue({
      data: {
        analysis_id: mockAnalysisId,
        mesh: mockMeshData
      }
    } as any);

    mockedApiService.getAnalysisTexture.mockResolvedValue({
      data: { texture_available: true }
    } as any);

    await waitFor(() => {
      expect(screen.getByTestId('mesh-loaded')).toBeInTheDocument();
    });
  });
});
