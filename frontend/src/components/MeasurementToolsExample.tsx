import React from 'react';
import Viewer3DWithMeasurements from './Viewer3DWithMeasurements';
import { Mesh } from '../types';

/**
 * Example component demonstrating measurement tools
 * 
 * Shows:
 * - Distance measurement between two points
 * - Area measurement for polygonal regions
 * - Measurements displayed in mm
 * 
 * Requirements: 3.7
 */
const MeasurementToolsExample: React.FC = () => {
  // Create a simple test mesh (a cube)
  const testMesh: Mesh = {
    vertices: [
      // Front face
      [-50, -50, 50],
      [50, -50, 50],
      [50, 50, 50],
      [-50, 50, 50],
      // Back face
      [-50, -50, -50],
      [50, -50, -50],
      [50, 50, -50],
      [-50, 50, -50]
    ],
    faces: [
      // Front face
      [0, 1, 2], [0, 2, 3],
      // Back face
      [4, 6, 5], [4, 7, 6],
      // Top face
      [3, 2, 6], [3, 6, 7],
      // Bottom face
      [0, 5, 1], [0, 4, 5],
      // Right face
      [1, 5, 6], [1, 6, 2],
      // Left face
      [0, 3, 7], [0, 7, 4]
    ],
    normals: [
      [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
      [0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1]
    ],
    uvCoordinates: [
      [0, 0], [1, 0], [1, 1], [0, 1],
      [0, 0], [1, 0], [1, 1], [0, 1]
    ],
    vertexColors: [
      [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0],
      [1, 0, 1], [0, 1, 1], [1, 1, 1], [0.5, 0.5, 0.5]
    ],
    vertexLabels: [0, 0, 0, 0, 0, 0, 0, 0],
    textureMap: ''
  };

  // Pixel to mm scale (1 unit = 1 mm for this example)
  const pixelToMmScale = 1.0;

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Measurement Tools Example</h1>
      
      <div style={styles.instructions}>
        <h2>Instructions:</h2>
        <ol>
          <li>Click the "Distance" button to measure distance between two points</li>
          <li>Click the "Area" button to measure area of a polygonal region</li>
          <li>Click on the mesh to place measurement points</li>
          <li>For area measurement, click the first point again to close the polygon</li>
          <li>Click "Clear" to remove all measurements</li>
        </ol>
        
        <h3>Expected Behavior:</h3>
        <ul>
          <li>Distance measurements show the geodesic distance in mm</li>
          <li>Area measurements show the surface area in mm²</li>
          <li>Red markers and lines indicate distance measurements</li>
          <li>Green markers and lines indicate area measurements</li>
          <li>Measurements are displayed in the side panel</li>
        </ul>
      </div>
      
      <div style={styles.viewerContainer}>
        <Viewer3DWithMeasurements
          mesh={testMesh}
          width={800}
          height={600}
          enableControls={true}
          pixelToMmScale={pixelToMmScale}
        />
      </div>
      
      <div style={styles.notes}>
        <h3>Notes:</h3>
        <ul>
          <li>This example uses a simple cube mesh for demonstration</li>
          <li>In production, the mesh would be a 3D facial model from reconstruction</li>
          <li>The pixel-to-mm scale is calculated from facial landmarks (e.g., interpupillary distance)</li>
          <li>Measurements are accurate to within 2% for distance and 5% for area (Property 11)</li>
        </ul>
      </div>
    </div>
  );
};

const styles = {
  container: {
    padding: '20px',
    maxWidth: '1400px',
    margin: '0 auto'
  },
  title: {
    fontSize: '24px',
    fontWeight: 'bold' as const,
    marginBottom: '20px',
    color: '#333'
  },
  instructions: {
    backgroundColor: '#f5f5f5',
    padding: '20px',
    borderRadius: '8px',
    marginBottom: '20px'
  },
  viewerContainer: {
    marginBottom: '20px',
    border: '1px solid #ddd',
    borderRadius: '8px',
    padding: '10px',
    backgroundColor: '#fff'
  },
  notes: {
    backgroundColor: '#e3f2fd',
    padding: '20px',
    borderRadius: '8px',
    fontSize: '14px'
  }
};

export default MeasurementToolsExample;
