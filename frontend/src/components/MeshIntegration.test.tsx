/**
 * Integration tests for mesh loading and rendering
 * 
 * Tests the complete workflow:
 * 1. Load mesh data from API
 * 2. Convert to Three.js format
 * 3. Render with proper shading and textures
 * 
 * Requirements: 3.1, 3.3
 */

import { render, waitFor, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import MeshViewer from './MeshViewer';
import apiService from '../services/api';

jest.mock('../services/api');
const mockedApiService = apiService as jest.Mocked<typeof apiService>;

jest.mock('./Viewer3D', () => {
  return function MockViewer3D({ mesh }: any) {
    if (!mesh) return null;
    
    // Validate mesh has data
    if (!mesh.vertices || mesh.vertices.length === 0 || 
        !mesh.faces || mesh.faces.length === 0) {
      return null;
    }
    
    return (
      <div data-testid="viewer3d">
        <div data-testid="vertex-count">{mesh.vertices.length}</div>
        <div data-testid="face-count">{mesh.faces.length}</div>
        <div data-testid="has-normals">{mesh.normals.length > 0 ? 'yes' : 'no'}</div>
        <div data-testid="has-colors">{mesh.vertexColors.length > 0 ? 'yes' : 'no'}</div>
      </div>
    );
  };
});

describe('Mesh Loading and Rendering Integration', () => {
  describe('Complete workflow from API to rendering', () => {
    test('loads mesh with all attributes and renders correctly', async () => {
      const mockMeshData = {
        vertices: Array.from({ length: 468 }, (_, i) => [i, i * 2, i * 3]),
        faces: Array.from({ length: 800 }, (_, i) => [i, i + 1, i + 2]),
        normals: Array.from({ length: 468 }, (_, i) => [0, 0, 1]),
        uv_coords: Array.from({ length: 468 }, (_, i) => [i / 468, i / 468]),
        vertex_colors: Array.from({ length: 468 }, () => [200, 150, 120]),
        vertex_labels: Array.from({ length: 468 }, () => 0),
        texture_map: ''
      };

      mockedApiService.getAnalysisStatus.mockResolvedValue({
        data: { status: 'completed', progress: 1.0 }
      } as any);

      mockedApiService.getAnalysisMesh.mockResolvedValue({
        data: { mesh: mockMeshData }
      } as any);

      mockedApiService.getAnalysisTexture.mockResolvedValue({
        data: { texture_available: true }
      } as any);

      const { getByTestId } = render(<MeshViewer analysisId="test-123" />);

      await waitFor(() => {
        expect(getByTestId('viewer3d')).toBeInTheDocument();
      });

      expect(getByTestId('vertex-count')).toHaveTextContent('468');
      expect(getByTestId('face-count')).toHaveTextContent('800');
      expect(getByTestId('has-normals')).toHaveTextContent('yes');
      expect(getByTestId('has-colors')).toHaveTextContent('yes');
    });

    test('handles mesh with vertex colors in 0-255 range', async () => {
      const mockMeshData = {
        vertices: [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces: [[0, 1, 2]],
        normals: [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
        uv_coords: [[0, 0], [1, 0], [0, 1]],
        vertex_colors: [[255, 128, 64], [200, 150, 100], [180, 140, 110]],
        vertex_labels: [0, 0, 0]
      };

      mockedApiService.getAnalysisStatus.mockResolvedValue({
        data: { status: 'completed', progress: 1.0 }
      } as any);

      mockedApiService.getAnalysisMesh.mockResolvedValue({
        data: { mesh: mockMeshData }
      } as any);

      mockedApiService.getAnalysisTexture.mockResolvedValue({
        data: { texture_available: true }
      } as any);

      const { getByTestId } = render(<MeshViewer analysisId="test-123" />);

      await waitFor(() => {
        expect(getByTestId('has-colors')).toHaveTextContent('yes');
      });
    });

    test('handles mesh without optional attributes', async () => {
      const mockMeshData = {
        vertices: [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces: [[0, 1, 2]],
        normals: [],
        uv_coords: [],
        vertex_colors: [],
        vertex_labels: []
      };

      mockedApiService.getAnalysisStatus.mockResolvedValue({
        data: { status: 'completed', progress: 1.0 }
      } as any);

      mockedApiService.getAnalysisMesh.mockResolvedValue({
        data: { mesh: mockMeshData }
      } as any);

      mockedApiService.getAnalysisTexture.mockResolvedValue({
        data: { texture_available: false }
      } as any);

      const { getByTestId } = render(<MeshViewer analysisId="test-123" />);

      await waitFor(() => {
        expect(getByTestId('viewer3d')).toBeInTheDocument();
      });

      // Should still render even without optional attributes
      expect(getByTestId('vertex-count')).toHaveTextContent('3');
      expect(getByTestId('face-count')).toHaveTextContent('1');
    });

    test('handles large mesh efficiently', async () => {
      // Create a large mesh (10,000 vertices, 18,000 faces)
      const vertexCount = 10000;
      const faceCount = 18000;

      const mockMeshData = {
        vertices: Array.from({ length: vertexCount }, (_, i) => [
          Math.cos(i) * 100,
          Math.sin(i) * 100,
          i * 0.1
        ]),
        faces: Array.from({ length: faceCount }, (_, i) => [
          i % vertexCount,
          (i + 1) % vertexCount,
          (i + 2) % vertexCount
        ]),
        normals: Array.from({ length: vertexCount }, (_, i) => [0, 0, 1]),
        uv_coords: Array.from({ length: vertexCount }, (_, i) => [
          i / vertexCount,
          i / vertexCount
        ]),
        vertex_colors: Array.from({ length: vertexCount }, () => [200, 150, 120]),
        vertex_labels: Array.from({ length: vertexCount }, () => 0)
      };

      mockedApiService.getAnalysisStatus.mockResolvedValue({
        data: { status: 'completed', progress: 1.0 }
      } as any);

      mockedApiService.getAnalysisMesh.mockResolvedValue({
        data: { mesh: mockMeshData }
      } as any);

      mockedApiService.getAnalysisTexture.mockResolvedValue({
        data: { texture_available: true }
      } as any);

      const startTime = Date.now();
      const { getByTestId } = render(<MeshViewer analysisId="test-123" />);

      await waitFor(() => {
        expect(getByTestId('viewer3d')).toBeInTheDocument();
      });

      const loadTime = Date.now() - startTime;

      // Should load within reasonable time (< 2 seconds)
      expect(loadTime).toBeLessThan(2000);
      expect(getByTestId('vertex-count')).toHaveTextContent(vertexCount.toString());
      expect(getByTestId('face-count')).toHaveTextContent(faceCount.toString());
    });
  });

  describe('Error handling and recovery', () => {
    test('recovers from transient network errors', async () => {
      let callCount = 0;
      mockedApiService.getAnalysisStatus.mockImplementation(() => {
        callCount++;
        if (callCount === 1) {
          return Promise.reject(new Error('Network timeout'));
        }
        return Promise.resolve({
          data: { status: 'completed', progress: 1.0 }
        } as any);
      });

      mockedApiService.getAnalysisMesh.mockResolvedValue({
        data: {
          mesh: {
            vertices: [[0, 0, 0]],
            faces: [[0, 0, 0]],
            normals: [],
            uv_coords: [],
            vertex_colors: [],
            vertex_labels: []
          }
        }
      } as any);

      mockedApiService.getAnalysisTexture.mockResolvedValue({
        data: { texture_available: false }
      } as any);

      const { getByText, queryByTestId } = render(
        <MeshViewer analysisId="test-123" />
      );

      // Should show error initially
      await waitFor(() => {
        expect(getByText(/Error Loading Mesh/i)).toBeInTheDocument();
      });

      // Click retry button
      const retryButton = getByText(/Retry/i);
      
      await act(async () => {
        retryButton.click();
      });

      // Should succeed on retry
      await waitFor(() => {
        expect(queryByTestId('viewer3d')).toBeInTheDocument();
      }, { timeout: 3000 });
    });

    test('validates mesh data structure before rendering', async () => {
      // Test with null vertices
      mockedApiService.getAnalysisStatus.mockResolvedValue({
        data: { status: 'completed', progress: 1.0 }
      } as any);

      mockedApiService.getAnalysisMesh.mockResolvedValue({
        data: { mesh: { vertices: null, faces: [[0, 1, 2]] } }
      } as any);

      const { getByText } = render(
        <MeshViewer analysisId="test-123" />
      );

      await waitFor(() => {
        expect(getByText(/Error Loading Mesh/i)).toBeInTheDocument();
        expect(getByText(/Invalid mesh data/i)).toBeInTheDocument();
      }, { timeout: 3000 });
    });
  });

  describe('Smooth shading and normals', () => {
    test('uses provided normals for smooth shading', async () => {
      const mockMeshData = {
        vertices: [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces: [[0, 1, 2]],
        normals: [[0, 0, 1], [0.707, 0, 0.707], [0, 0.707, 0.707]],
        uv_coords: [[0, 0], [1, 0], [0, 1]],
        vertex_colors: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        vertex_labels: [0, 0, 0]
      };

      mockedApiService.getAnalysisStatus.mockResolvedValue({
        data: { status: 'completed', progress: 1.0 }
      } as any);

      mockedApiService.getAnalysisMesh.mockResolvedValue({
        data: { mesh: mockMeshData }
      } as any);

      mockedApiService.getAnalysisTexture.mockResolvedValue({
        data: { texture_available: true }
      } as any);

      const { getByTestId } = render(<MeshViewer analysisId="test-123" />);

      await waitFor(() => {
        expect(getByTestId('has-normals')).toHaveTextContent('yes');
      });
    });

    test('computes normals when not provided', async () => {
      const mockMeshData = {
        vertices: [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces: [[0, 1, 2]],
        normals: [], // No normals provided
        uv_coords: [[0, 0], [1, 0], [0, 1]],
        vertex_colors: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        vertex_labels: [0, 0, 0]
      };

      mockedApiService.getAnalysisStatus.mockResolvedValue({
        data: { status: 'completed', progress: 1.0 }
      } as any);

      mockedApiService.getAnalysisMesh.mockResolvedValue({
        data: { mesh: mockMeshData }
      } as any);

      mockedApiService.getAnalysisTexture.mockResolvedValue({
        data: { texture_available: true }
      } as any);

      const { getByTestId } = render(<MeshViewer analysisId="test-123" />);

      await waitFor(() => {
        expect(getByTestId('viewer3d')).toBeInTheDocument();
      });

      // Should still render successfully with computed normals
      expect(getByTestId('vertex-count')).toHaveTextContent('3');
    });
  });

  describe('Texture mapping', () => {
    test('applies vertex colors as texture', async () => {
      const mockMeshData = {
        vertices: [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces: [[0, 1, 2]],
        normals: [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
        uv_coords: [[0, 0], [1, 0], [0, 1]],
        vertex_colors: [[230, 180, 140], [220, 170, 130], [210, 160, 120]],
        vertex_labels: [0, 0, 0],
        texture_map: ''
      };

      mockedApiService.getAnalysisStatus.mockResolvedValue({
        data: { status: 'completed', progress: 1.0 }
      } as any);

      mockedApiService.getAnalysisMesh.mockResolvedValue({
        data: { mesh: mockMeshData }
      } as any);

      mockedApiService.getAnalysisTexture.mockResolvedValue({
        data: { texture_available: true }
      } as any);

      const { getByTestId } = render(<MeshViewer analysisId="test-123" />);

      await waitFor(() => {
        expect(getByTestId('has-colors')).toHaveTextContent('yes');
      });
    });

    test('handles base64 texture maps', async () => {
      const mockMeshData = {
        vertices: [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces: [[0, 1, 2]],
        normals: [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
        uv_coords: [[0, 0], [1, 0], [0, 1]],
        vertex_colors: [],
        vertex_labels: [0, 0, 0],
        texture_map: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='
      };

      mockedApiService.getAnalysisStatus.mockResolvedValue({
        data: { status: 'completed', progress: 1.0 }
      } as any);

      mockedApiService.getAnalysisMesh.mockResolvedValue({
        data: { mesh: mockMeshData }
      } as any);

      mockedApiService.getAnalysisTexture.mockResolvedValue({
        data: { texture_available: true }
      } as any);

      const { getByTestId } = render(<MeshViewer analysisId="test-123" />);

      await waitFor(() => {
        expect(getByTestId('viewer3d')).toBeInTheDocument();
      });
    });
  });
});
