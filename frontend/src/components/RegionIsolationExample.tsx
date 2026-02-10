import React, { useState, useRef } from 'react';
import Viewer3D, { Viewer3DHandle, RegionConfig, FacialRegion } from './Viewer3D';
import RegionControls from './RegionControls';
import { Mesh } from '../types';

/**
 * Example component demonstrating region isolation functionality
 * Shows how to use RegionControls with Viewer3D for facial region analysis
 * 
 * Requirements: 3.6
 */
const RegionIsolationExample: React.FC = () => {
  const viewerRef = useRef<Viewer3DHandle>(null);
  const [isReady, setIsReady] = useState(false);
  const [regionConfig, setRegionConfig] = useState<RegionConfig>({
    selectedRegion: 'all',
    highlightIntensity: 0.5
  });

  // Create a test mesh representing a simplified face
  const testMesh: Mesh = {
    vertices: [
      // Forehead region (y > 20)
      [-40, 40, 0], [-20, 40, 0], [0, 40, 0], [20, 40, 0], [40, 40, 0],
      [-40, 30, 0], [-20, 30, 0], [0, 30, 0], [20, 30, 0], [40, 30, 0],
      [-40, 20, 0], [-20, 20, 0], [0, 20, 0], [20, 20, 0], [40, 20, 0],
      
      // Eye regions (y: 5-25)
      [-30, 15, 5], [-20, 15, 5], [-10, 15, 5], // Left eye
      [10, 15, 5], [20, 15, 5], [30, 15, 5],    // Right eye
      
      // Nose region (y: -10 to 15, z > 5)
      [-10, 10, 10], [0, 10, 15], [10, 10, 10],
      [-10, 0, 10], [0, 0, 15], [10, 0, 10],
      [-10, -10, 10], [0, -10, 15], [10, -10, 10],
      
      // Cheek regions
      [-35, 0, 0], [-25, 0, 0], [-15, 0, 0], // Left cheek
      [15, 0, 0], [25, 0, 0], [35, 0, 0],    // Right cheek
      [-35, -10, 0], [-25, -10, 0], [-15, -10, 0],
      [15, -10, 0], [25, -10, 0], [35, -10, 0],
      
      // Mouth region (y: -35 to -15)
      [-20, -25, 0], [-10, -25, 0], [0, -25, 0], [10, -25, 0], [20, -25, 0],
      
      // Chin region (y < -35)
      [-15, -40, 0], [0, -40, 0], [15, -40, 0],
      [-10, -50, 0], [0, -50, 0], [10, -50, 0]
    ],
    faces: [
      // Create triangular faces connecting the vertices
      // Forehead
      [0, 1, 5], [1, 6, 5], [1, 2, 6], [2, 7, 6],
      [2, 3, 7], [3, 8, 7], [3, 4, 8], [4, 9, 8],
      [5, 6, 10], [6, 11, 10], [6, 7, 11], [7, 12, 11],
      [7, 8, 12], [8, 13, 12], [8, 9, 13], [9, 14, 13],
      
      // Eyes and nose
      [15, 16, 17], [18, 19, 20],
      [21, 22, 23], [22, 24, 23], [24, 25, 26],
      [23, 24, 27], [24, 28, 27], [25, 26, 28],
      
      // Cheeks
      [29, 30, 31], [32, 33, 34],
      [35, 36, 37], [38, 39, 40],
      
      // Mouth
      [41, 42, 43], [43, 44, 45],
      
      // Chin
      [46, 47, 48], [49, 50, 51]
    ],
    normals: [],
    uvCoordinates: [],
    vertexColors: [],
    vertexLabels: [],
    textureMap: ''
  };

  const handleZoomToRegion = (region: FacialRegion) => {
    if (viewerRef.current) {
      viewerRef.current.zoomToRegion(region);
    }
  };

  return (
    <div style={styles.container}>
      <h2 style={styles.title}>Region Isolation Example</h2>
      
      <div style={styles.content}>
        {/* Controls Panel */}
        <div style={styles.sidebar}>
          <RegionControls
            regionConfig={regionConfig}
            onRegionConfigChange={setRegionConfig}
            onZoomToRegion={handleZoomToRegion}
          />
          
          {isReady && (
            <div style={styles.status}>
              ✓ Viewer Ready
            </div>
          )}
        </div>

        {/* 3D Viewer */}
        <div style={styles.viewerContainer}>
          <Viewer3D
            ref={viewerRef}
            mesh={testMesh}
            width={800}
            height={600}
            showWireframe={false}
            enableControls={true}
            onReady={() => setIsReady(true)}
            regionConfig={regionConfig}
          />
        </div>
      </div>

      <div style={styles.instructions}>
        <h3>Instructions:</h3>
        <ul>
          <li><strong>Select Region:</strong> Choose a facial region from the dropdown or quick buttons</li>
          <li><strong>Adjust Highlight:</strong> Use the slider to control how much the region stands out</li>
          <li><strong>Zoom to Region:</strong> Click the zoom button to focus the camera on the selected region</li>
          <li><strong>Mouse Controls:</strong> Left-click to rotate, right-click to pan, scroll to zoom</li>
        </ul>
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
  title: {
    fontSize: '24px',
    fontWeight: 'bold',
    marginBottom: '20px',
    color: '#333'
  },
  content: {
    display: 'flex',
    gap: '20px',
    marginBottom: '20px'
  },
  sidebar: {
    flexShrink: 0
  },
  viewerContainer: {
    flex: 1,
    border: '1px solid #ccc',
    borderRadius: '8px',
    overflow: 'hidden',
    backgroundColor: '#f0f0f0'
  },
  status: {
    marginTop: '20px',
    padding: '12px',
    backgroundColor: '#4CAF50',
    color: 'white',
    borderRadius: '6px',
    textAlign: 'center',
    fontWeight: '500'
  },
  instructions: {
    padding: '20px',
    backgroundColor: '#f5f5f5',
    borderRadius: '8px',
    fontSize: '14px',
    color: '#666'
  }
};

export default RegionIsolationExample;
