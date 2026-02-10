import React, { useState } from 'react';
import Viewer3D from './Viewer3D';
import { Mesh } from '../types';

/**
 * Example component demonstrating Viewer3D usage
 * Creates a simple test mesh for visualization
 */
const Viewer3DExample: React.FC = () => {
  const [showWireframe, setShowWireframe] = useState(false);
  const [isReady, setIsReady] = useState(false);

  // Create a simple test mesh (a pyramid)
  const testMesh: Mesh = {
    vertices: [
      // Base vertices
      [-50, -50, 0],
      [50, -50, 0],
      [50, 50, 0],
      [-50, 50, 0],
      // Apex
      [0, 0, 80]
    ],
    faces: [
      // Base
      [0, 1, 2],
      [0, 2, 3],
      // Sides
      [0, 1, 4],
      [1, 2, 4],
      [2, 3, 4],
      [3, 0, 4]
    ],
    normals: [
      [0, 0, -1],
      [0, 0, -1],
      [0, 0, -1],
      [0, 0, -1],
      [0, 0, 1]
    ],
    uvCoordinates: [
      [0, 0],
      [1, 0],
      [1, 1],
      [0, 1],
      [0.5, 0.5]
    ],
    vertexColors: [
      [1, 0, 0], // Red
      [0, 1, 0], // Green
      [0, 0, 1], // Blue
      [1, 1, 0], // Yellow
      [1, 0, 1]  // Magenta
    ],
    vertexLabels: [0, 0, 0, 0, 0],
    textureMap: ''
  };

  return (
    <div style={{ padding: '20px' }}>
      <h2>3D Viewer Example</h2>
      
      <div style={{ marginBottom: '20px' }}>
        <label style={{ marginRight: '10px' }}>
          <input
            type="checkbox"
            checked={showWireframe}
            onChange={(e) => setShowWireframe(e.target.checked)}
          />
          Show Wireframe
        </label>
        
        {isReady && (
          <span style={{ color: 'green', marginLeft: '20px' }}>
            ✓ Viewer Ready
          </span>
        )}
      </div>

      <div style={{ 
        border: '1px solid #ccc', 
        borderRadius: '8px',
        overflow: 'hidden',
        width: '100%',
        height: '600px'
      }}>
        <Viewer3D
          mesh={testMesh}
          width={800}
          height={600}
          showWireframe={showWireframe}
          enableControls={true}
          onReady={() => setIsReady(true)}
        />
      </div>

      <div style={{ marginTop: '20px', fontSize: '14px', color: '#666' }}>
        <h3>Controls:</h3>
        <ul>
          <li><strong>Left Mouse:</strong> Rotate the view</li>
          <li><strong>Right Mouse:</strong> Pan the view</li>
          <li><strong>Mouse Wheel:</strong> Zoom in/out</li>
        </ul>
      </div>
    </div>
  );
};

export default Viewer3DExample;
