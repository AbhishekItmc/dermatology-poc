import React, { useState, useEffect } from 'react';
import MeshViewer from './MeshViewer';
import Viewer3D from './Viewer3D';
import { Mesh } from '../types';
import apiService from '../services/api';

/**
 * Example component demonstrating MeshViewer usage with real API data
 * 
 * Features:
 * - Shows how to use MeshViewer with analysis ID
 * - Demonstrates loading states and error handling
 * - Provides controls for visualization options
 * - Shows mesh statistics and metadata
 */
const MeshViewerExample: React.FC = () => {
  const [analysisId, setAnalysisId] = useState<string>('');
  const [showWireframe, setShowWireframe] = useState<boolean>(false);
  const [useRealData, setUseRealData] = useState<boolean>(false);
  const [meshStats, setMeshStats] = useState<any>(null);

  // Create a more realistic test mesh (a simple face-like structure)
  const createTestMesh = (): Mesh => {
    // Create a simple face mesh with more vertices
    const vertices: number[][] = [];
    const faces: number[][] = [];
    const normals: number[][] = [];
    const uvCoordinates: number[][] = [];
    const vertexColors: number[][] = [];

    // Create a grid of vertices for a face-like surface
    const gridSize = 20;
    const spacing = 10;
    
    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        const x = (i - gridSize / 2) * spacing;
        const y = (j - gridSize / 2) * spacing;
        
        // Create a curved surface (ellipsoid-like)
        const z = Math.sqrt(
          Math.max(0, 10000 - (x * x * 0.8) - (y * y * 1.2))
        ) * 0.3;
        
        vertices.push([x, y, z]);
        
        // Normal pointing outward
        const nx = -x * 0.01;
        const ny = -y * 0.01;
        const nz = 1.0;
        const norm = Math.sqrt(nx * nx + ny * ny + nz * nz);
        normals.push([nx / norm, ny / norm, nz / norm]);
        
        // UV coordinates
        uvCoordinates.push([i / (gridSize - 1), j / (gridSize - 1)]);
        
        // Skin-like color with variation
        const baseR = 0.9 + Math.random() * 0.1;
        const baseG = 0.7 + Math.random() * 0.1;
        const baseB = 0.6 + Math.random() * 0.1;
        vertexColors.push([baseR, baseG, baseB]);
      }
    }

    // Create faces (triangles)
    for (let i = 0; i < gridSize - 1; i++) {
      for (let j = 0; j < gridSize - 1; j++) {
        const idx = i * gridSize + j;
        
        // Two triangles per grid cell
        faces.push([idx, idx + 1, idx + gridSize]);
        faces.push([idx + 1, idx + gridSize + 1, idx + gridSize]);
      }
    }

    return {
      vertices,
      faces,
      normals,
      uvCoordinates,
      vertexColors,
      vertexLabels: new Array(vertices.length).fill(0),
      textureMap: ''
    };
  };

  const testMesh = createTestMesh();

  useEffect(() => {
    if (testMesh) {
      setMeshStats({
        vertexCount: testMesh.vertices.length,
        faceCount: testMesh.faces.length,
        hasNormals: testMesh.normals.length > 0,
        hasUVs: testMesh.uvCoordinates.length > 0,
        hasColors: testMesh.vertexColors.length > 0
      });
    }
  }, []);

  const handleLoadRealMesh = () => {
    if (!analysisId.trim()) {
      alert('Please enter an analysis ID');
      return;
    }
    setUseRealData(true);
  };

  const handleBackToTest = () => {
    setUseRealData(false);
  };

  return (
    <div style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>
      <h2>3D Mesh Viewer Example</h2>
      
      <div style={{
        backgroundColor: '#f5f5f5',
        padding: '20px',
        borderRadius: '8px',
        marginBottom: '20px'
      }}>
        <h3>Load Options</h3>
        
        {!useRealData ? (
          <div>
            <div style={{ marginBottom: '15px' }}>
              <p style={{ marginBottom: '10px', color: '#666' }}>
                Currently showing: <strong>Test Mesh</strong> (synthetic face-like surface)
              </p>
            </div>
            
            <div style={{ marginBottom: '15px' }}>
              <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                Load Real Analysis Data:
              </label>
              <div style={{ display: 'flex', gap: '10px' }}>
                <input
                  type="text"
                  value={analysisId}
                  onChange={(e) => setAnalysisId(e.target.value)}
                  placeholder="Enter analysis ID"
                  style={{
                    flex: 1,
                    padding: '8px',
                    border: '1px solid #ccc',
                    borderRadius: '4px',
                    fontSize: '14px'
                  }}
                />
                <button
                  onClick={handleLoadRealMesh}
                  style={{
                    padding: '8px 20px',
                    backgroundColor: '#2196F3',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontSize: '14px',
                    fontWeight: 'bold'
                  }}
                >
                  Load Real Mesh
                </button>
              </div>
            </div>
          </div>
        ) : (
          <div>
            <p style={{ marginBottom: '10px', color: '#666' }}>
              Currently showing: <strong>Real Analysis Data</strong> (ID: {analysisId})
            </p>
            <button
              onClick={handleBackToTest}
              style={{
                padding: '8px 20px',
                backgroundColor: '#757575',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '14px'
              }}
            >
              ← Back to Test Mesh
            </button>
          </div>
        )}
      </div>

      <div style={{
        backgroundColor: '#fff',
        padding: '15px',
        borderRadius: '8px',
        marginBottom: '20px',
        border: '1px solid #e0e0e0'
      }}>
        <h3>Visualization Controls</h3>
        <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={showWireframe}
            onChange={(e) => setShowWireframe(e.target.checked)}
            style={{ marginRight: '8px' }}
          />
          <span>Show Wireframe</span>
        </label>
      </div>

      {meshStats && !useRealData && (
        <div style={{
          backgroundColor: '#e3f2fd',
          padding: '15px',
          borderRadius: '8px',
          marginBottom: '20px',
          border: '1px solid #90caf9'
        }}>
          <h3>Mesh Statistics</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
            <div>
              <strong>Vertices:</strong> {meshStats.vertexCount}
            </div>
            <div>
              <strong>Faces:</strong> {meshStats.faceCount}
            </div>
            <div>
              <strong>Normals:</strong> {meshStats.hasNormals ? '✓ Yes' : '✗ No'}
            </div>
            <div>
              <strong>UV Coords:</strong> {meshStats.hasUVs ? '✓ Yes' : '✗ No'}
            </div>
            <div>
              <strong>Vertex Colors:</strong> {meshStats.hasColors ? '✓ Yes' : '✗ No'}
            </div>
          </div>
        </div>
      )}

      <div style={{
        border: '2px solid #e0e0e0',
        borderRadius: '8px',
        overflow: 'hidden',
        backgroundColor: '#fff',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
      }}>
        {useRealData ? (
          <MeshViewer
            analysisId={analysisId}
            width={800}
            height={600}
            showWireframe={showWireframe}
            enableControls={true}
          />
        ) : (
          <Viewer3D
            mesh={testMesh}
            width={800}
            height={600}
            showWireframe={showWireframe}
            enableControls={true}
          />
        )}
      </div>

      <div style={{
        marginTop: '20px',
        padding: '15px',
        backgroundColor: '#f5f5f5',
        borderRadius: '8px',
        fontSize: '14px',
        color: '#666'
      }}>
        <h3>Interaction Controls:</h3>
        <ul style={{ marginLeft: '20px' }}>
          <li><strong>Left Mouse Button + Drag:</strong> Rotate the view around the mesh</li>
          <li><strong>Right Mouse Button + Drag:</strong> Pan the view</li>
          <li><strong>Mouse Wheel:</strong> Zoom in and out</li>
          <li><strong>Touch (Mobile):</strong> One finger to rotate, two fingers to zoom/pan</li>
        </ul>
        
        <h3 style={{ marginTop: '15px' }}>Features Implemented:</h3>
        <ul style={{ marginLeft: '20px' }}>
          <li>✓ Load mesh data from API (GET /api/v1/analyses/{'{id}'}/mesh)</li>
          <li>✓ Create Three.js geometry from vertices and faces</li>
          <li>✓ Apply vertex colors for texture representation</li>
          <li>✓ Implement smooth shading with normals</li>
          <li>✓ Real-time rendering with GPU acceleration</li>
          <li>✓ Interactive orbit controls (rotation, zoom, pan)</li>
          <li>✓ Loading states and error handling</li>
          <li>✓ Automatic mesh centering and scaling</li>
        </ul>
      </div>
    </div>
  );
};

export default MeshViewerExample;
