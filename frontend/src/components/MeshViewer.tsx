import React, { useState, useEffect } from 'react';
import Viewer3D from './Viewer3D';
import { Mesh } from '../types';
import apiService from '../services/api';

interface MeshViewerProps {
  analysisId: string;
  width?: number;
  height?: number;
  showWireframe?: boolean;
  enableControls?: boolean;
}

/**
 * Component that loads mesh data from API and displays it in Viewer3D
 * 
 * Features:
 * - Loads mesh data from backend API
 * - Handles loading states and errors
 * - Converts API mesh format to Viewer3D format
 * - Applies texture maps and vertex colors
 * 
 * Requirements: 3.1, 3.3
 */
const MeshViewer: React.FC<MeshViewerProps> = ({
  analysisId,
  width = 800,
  height = 600,
  showWireframe = false,
  enableControls = true
}) => {
  const [mesh, setMesh] = useState<Mesh | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [loadProgress, setLoadProgress] = useState<number>(0);

  useEffect(() => {
    loadMeshData();
  }, [analysisId]);

  /**
   * Load mesh data from API
   */
  const loadMeshData = async () => {
    try {
      setLoading(true);
      setError(null);
      setLoadProgress(0);

      // Check analysis status first
      setLoadProgress(10);
      const statusResponse = await apiService.getAnalysisStatus(analysisId);
      
      if (statusResponse.data.status !== 'completed') {
        throw new Error(`Analysis not completed. Status: ${statusResponse.data.status}`);
      }

      // Load mesh data
      setLoadProgress(30);
      const meshResponse = await apiService.getAnalysisMesh(analysisId);
      
      if (!meshResponse.data || !meshResponse.data.mesh) {
        throw new Error('No mesh data available');
      }

      // Convert API mesh format to Viewer3D format
      setLoadProgress(60);
      const apiMesh = meshResponse.data.mesh;
      const convertedMesh = convertApiMeshToViewerMesh(apiMesh);

      // Load texture if available
      setLoadProgress(80);
      try {
        const textureResponse = await apiService.getAnalysisTexture(analysisId);
        if (textureResponse.data.texture_available) {
          // Texture is embedded in vertex colors, already handled
          console.log('Texture data embedded in vertex colors');
        }
      } catch (textureError) {
        console.warn('Texture loading failed, using vertex colors only:', textureError);
      }

      setLoadProgress(100);
      setMesh(convertedMesh);
      setLoading(false);

    } catch (err: any) {
      console.error('Failed to load mesh:', err);
      setError(err.message || 'Failed to load mesh data');
      setLoading(false);
    }
  };

  /**
   * Convert API mesh format to Viewer3D mesh format
   */
  const convertApiMeshToViewerMesh = (apiMesh: any): Mesh => {
    // Ensure all required fields are present
    if (!apiMesh.vertices || !apiMesh.faces || 
        apiMesh.vertices.length === 0 || apiMesh.faces.length === 0) {
      throw new Error('Invalid mesh data: missing vertices or faces');
    }

    // Convert vertex colors from 0-255 to 0-1 range if needed
    let vertexColors = apiMesh.vertex_colors || [];
    if (vertexColors.length > 0 && vertexColors[0] && vertexColors[0][0] > 1) {
      // Colors are in 0-255 range, convert to 0-1
      vertexColors = vertexColors.map((color: number[]) => 
        color.map((c: number) => c / 255.0)
      );
    }

    // Ensure normals are present
    const normals = apiMesh.normals || [];
    
    // Ensure UV coordinates are present
    const uvCoordinates = apiMesh.uv_coords || [];

    // Create mesh object
    const mesh: Mesh = {
      vertices: apiMesh.vertices,
      faces: apiMesh.faces,
      normals: normals,
      uvCoordinates: uvCoordinates,
      vertexColors: vertexColors,
      vertexLabels: apiMesh.vertex_labels || [],
      textureMap: apiMesh.texture_map || ''
    };

    return mesh;
  };

  /**
   * Retry loading mesh data
   */
  const handleRetry = () => {
    loadMeshData();
  };

  if (loading) {
    return (
      <div style={{
        width: '100%',
        height: `${height}px`,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: '#f0f0f0',
        borderRadius: '8px'
      }}>
        <div style={{ marginBottom: '20px' }}>
          <div style={{
            width: '200px',
            height: '20px',
            backgroundColor: '#e0e0e0',
            borderRadius: '10px',
            overflow: 'hidden'
          }}>
            <div style={{
              width: `${loadProgress}%`,
              height: '100%',
              backgroundColor: '#4CAF50',
              transition: 'width 0.3s ease'
            }} />
          </div>
        </div>
        <p style={{ color: '#666', fontSize: '16px' }}>
          Loading 3D mesh... {loadProgress}%
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{
        width: '100%',
        height: `${height}px`,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: '#fff3f3',
        borderRadius: '8px',
        border: '1px solid #ffcccc'
      }}>
        <div style={{ color: '#d32f2f', fontSize: '18px', marginBottom: '10px' }}>
          ⚠️ Error Loading Mesh
        </div>
        <p style={{ color: '#666', fontSize: '14px', marginBottom: '20px' }}>
          {error}
        </p>
        <button
          onClick={handleRetry}
          style={{
            padding: '10px 20px',
            backgroundColor: '#2196F3',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '14px'
          }}
        >
          Retry
        </button>
      </div>
    );
  }

  if (!mesh) {
    return (
      <div style={{
        width: '100%',
        height: `${height}px`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: '#f0f0f0',
        borderRadius: '8px'
      }}>
        <p style={{ color: '#666', fontSize: '16px' }}>
          No mesh data available
        </p>
      </div>
    );
  }

  return (
    <Viewer3D
      mesh={mesh}
      width={width}
      height={height}
      showWireframe={showWireframe}
      enableControls={enableControls}
    />
  );
};

export default MeshViewer;
