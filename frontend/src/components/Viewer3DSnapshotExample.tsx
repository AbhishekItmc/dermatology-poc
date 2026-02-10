import React, { useRef, useState } from 'react';
import Viewer3D, { Viewer3DHandle } from './Viewer3D';
import { Mesh } from '../types';

/**
 * Example component demonstrating snapshot export functionality
 * Shows how to use the Viewer3D ref to export snapshots in PNG and JPEG formats
 * 
 * Requirements: 3.8
 */
const Viewer3DSnapshotExample: React.FC = () => {
  const viewerRef = useRef<Viewer3DHandle>(null);
  const [exportStatus, setExportStatus] = useState<string>('');

  // Create a simple test mesh (cube)
  const testMesh: Mesh = {
    vertices: [
      [-50, -50, -50], [50, -50, -50], [50, 50, -50], [-50, 50, -50], // Front face
      [-50, -50, 50], [50, -50, 50], [50, 50, 50], [-50, 50, 50]      // Back face
    ],
    faces: [
      [0, 1, 2], [0, 2, 3], // Front
      [4, 6, 5], [4, 7, 6], // Back
      [0, 4, 5], [0, 5, 1], // Bottom
      [2, 6, 7], [2, 7, 3], // Top
      [0, 3, 7], [0, 7, 4], // Left
      [1, 5, 6], [1, 6, 2]  // Right
    ],
    normals: [],
    uvCoordinates: [],
    vertexColors: [
      [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0],
      [1, 0, 1], [0, 1, 1], [1, 1, 1], [0.5, 0.5, 0.5]
    ],
    vertexLabels: [],
    textureMap: ''
  };

  const handleExportPNG = () => {
    try {
      viewerRef.current?.exportSnapshot('png', 'test_snapshot.png');
      setExportStatus('PNG exported successfully!');
      setTimeout(() => setExportStatus(''), 3000);
    } catch (error) {
      setExportStatus(`Export failed: ${error}`);
    }
  };

  const handleExportJPEG = () => {
    try {
      viewerRef.current?.exportSnapshot('jpeg', 'test_snapshot.jpg');
      setExportStatus('JPEG exported successfully!');
      setTimeout(() => setExportStatus(''), 3000);
    } catch (error) {
      setExportStatus(`Export failed: ${error}`);
    }
  };

  const handleExportDefault = () => {
    try {
      viewerRef.current?.exportSnapshot('png'); // Uses default filename
      setExportStatus('Snapshot exported with default filename!');
      setTimeout(() => setExportStatus(''), 3000);
    } catch (error) {
      setExportStatus(`Export failed: ${error}`);
    }
  };

  return (
    <div style={{ padding: '20px' }}>
      <h2>3D Viewer Snapshot Export Example</h2>
      <p>Rotate the cube and click the buttons below to export snapshots.</p>
      
      <div style={{ 
        border: '1px solid #ccc', 
        borderRadius: '4px', 
        overflow: 'hidden',
        marginBottom: '20px'
      }}>
        <Viewer3D
          ref={viewerRef}
          mesh={testMesh}
          width={800}
          height={600}
          enableControls={true}
        />
      </div>

      <div style={{ display: 'flex', gap: '10px', marginBottom: '10px' }}>
        <button
          onClick={handleExportPNG}
          style={{
            padding: '10px 20px',
            backgroundColor: '#4CAF50',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '14px'
          }}
        >
          Export as PNG
        </button>

        <button
          onClick={handleExportJPEG}
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
          Export as JPEG
        </button>

        <button
          onClick={handleExportDefault}
          style={{
            padding: '10px 20px',
            backgroundColor: '#FF9800',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '14px'
          }}
        >
          Export (Default Name)
        </button>
      </div>

      {exportStatus && (
        <div style={{
          padding: '10px',
          backgroundColor: exportStatus.includes('failed') ? '#ffebee' : '#e8f5e9',
          color: exportStatus.includes('failed') ? '#c62828' : '#2e7d32',
          borderRadius: '4px',
          marginTop: '10px'
        }}>
          {exportStatus}
        </div>
      )}

      <div style={{ marginTop: '20px', fontSize: '14px', color: '#666' }}>
        <h3>Features:</h3>
        <ul>
          <li>Export current view as PNG or JPEG</li>
          <li>Includes all visible layers and annotations</li>
          <li>Custom or auto-generated filenames</li>
          <li>Downloads directly to user's device</li>
          <li>Captures current camera angle and zoom level</li>
        </ul>
      </div>
    </div>
  );
};

export default Viewer3DSnapshotExample;
