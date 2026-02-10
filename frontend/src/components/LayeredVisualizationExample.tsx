import React, { useState } from 'react';
import Viewer3D, { LayerConfig } from './Viewer3D';
import LayerControls from './LayerControls';
import { Mesh } from '../types';

/**
 * Example component demonstrating layered visualization with controls
 * 
 * Shows how to:
 * - Use Viewer3D with layered meshes
 * - Integrate LayerControls for user interaction
 * - Toggle layer visibility and adjust transparency
 * - Blend multiple layers together
 * 
 * Requirements: 4.6, 4.7
 */
const LayeredVisualizationExample: React.FC = () => {
  const [layerConfig, setLayerConfig] = useState<LayerConfig>({
    base: { visible: true, opacity: 1.0 },
    pigmentation: { visible: true, opacity: 0.7 },
    wrinkles: { visible: true, opacity: 0.7 }
  });
  
  // Example mesh with vertex labels
  // In a real application, this would come from the API
  const exampleMesh: Mesh = {
    vertices: [
      [0, 0, 0],
      [10, 0, 0],
      [10, 10, 0],
      [0, 10, 0],
      [5, 5, 5],
      [15, 5, 0],
      [15, 15, 0],
      [5, 15, 0]
    ],
    faces: [
      [0, 1, 4],
      [1, 2, 4],
      [2, 3, 4],
      [3, 0, 4],
      [1, 5, 4],
      [5, 6, 4],
      [6, 7, 4],
      [7, 3, 4]
    ],
    normals: [],
    uvCoordinates: [],
    vertexColors: [],
    // Vertex labels: 0=base, 1-3=pigmentation (low/med/high), 4-5=wrinkles (micro/regular)
    vertexLabels: [0, 1, 2, 3, 0, 4, 5, 0],
    textureMap: ''
  };
  
  const handleLayerConfigChange = (newConfig: LayerConfig) => {
    setLayerConfig(newConfig);
  };
  
  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h2 style={styles.title}>Layered Visualization Example</h2>
        <p style={styles.description}>
          This example demonstrates the layered visualization feature with separate
          materials for base mesh, pigmentation anomalies, and wrinkles. Use the
          controls to toggle layer visibility and adjust transparency.
        </p>
      </div>
      
      <div style={styles.content}>
        <div style={styles.viewerContainer}>
          <Viewer3D
            mesh={exampleMesh}
            width={800}
            height={600}
            enableControls={true}
            layerConfig={layerConfig}
            onLayerConfigChange={handleLayerConfigChange}
          />
        </div>
        
        <div style={styles.controlsContainer}>
          <LayerControls
            layerConfig={layerConfig}
            onLayerConfigChange={handleLayerConfigChange}
          />
          
          <div style={styles.info}>
            <h4 style={styles.infoTitle}>Layer Information</h4>
            <ul style={styles.infoList}>
              <li><strong>Base:</strong> The underlying facial mesh with skin tone</li>
              <li><strong>Pigmentation:</strong> Color-coded overlays for detected pigmentation
                <ul>
                  <li>Light Yellow: Low severity</li>
                  <li>Orange: Medium severity</li>
                  <li>Dark Red: High severity</li>
                </ul>
              </li>
              <li><strong>Wrinkles:</strong> Color-coded overlays for detected wrinkles
                <ul>
                  <li>Light Blue: Micro-wrinkles</li>
                  <li>Dark Blue: Regular wrinkles</li>
                </ul>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    padding: '20px',
    maxWidth: '1400px',
    margin: '0 auto'
  },
  header: {
    marginBottom: '30px'
  },
  title: {
    fontSize: '28px',
    fontWeight: 'bold',
    color: '#333',
    marginBottom: '10px'
  },
  description: {
    fontSize: '16px',
    color: '#666',
    lineHeight: '1.6'
  },
  content: {
    display: 'flex',
    gap: '20px',
    alignItems: 'flex-start'
  },
  viewerContainer: {
    flex: 1,
    minWidth: '600px'
  },
  controlsContainer: {
    display: 'flex',
    flexDirection: 'column',
    gap: '20px'
  },
  info: {
    padding: '20px',
    backgroundColor: '#f5f5f5',
    borderRadius: '8px',
    maxWidth: '350px'
  },
  infoTitle: {
    margin: '0 0 15px 0',
    fontSize: '16px',
    fontWeight: 'bold',
    color: '#333'
  },
  infoList: {
    margin: 0,
    paddingLeft: '20px',
    fontSize: '14px',
    color: '#666',
    lineHeight: '1.8'
  }
};

export default LayeredVisualizationExample;
