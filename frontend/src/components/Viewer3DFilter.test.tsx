import React from 'react';
import { render, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import Viewer3D, { SeverityFilterConfig } from './Viewer3D';
import { Mesh } from '../types';
import * as THREE from 'three';

// Mock Three.js WebGLRenderer to avoid WebGL context issues in tests
jest.mock('three', () => {
  const actualThree = jest.requireActual('three');
  return {
    ...actualThree,
    WebGLRenderer: jest.fn().mockImplementation(() => {
      // Always create a real canvas element
      const canvas = document.createElement('canvas');
      return {
        setSize: jest.fn(),
        setPixelRatio: jest.fn(),
        render: jest.fn(),
        dispose: jest.fn(),
        domElement: canvas,
        shadowMap: { enabled: false, type: 0 }
      };
    })
  };
});

describe('Viewer3D - Severity Filter Integration', () => {
  // Create a test mesh with vertex labels
  const createTestMesh = (): Mesh => ({
    vertices: [
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [1, 1, 0],
      [0, 0, 1],
      [1, 0, 1]
    ],
    faces: [
      [0, 1, 2],
      [1, 3, 2],
      [0, 4, 1],
      [1, 4, 5]
    ],
    normals: [
      [0, 0, 1],
      [0, 0, 1],
      [0, 0, 1],
      [0, 0, 1],
      [0, 0, 1],
      [0, 0, 1]
    ],
    uvCoordinates: [
      [0, 0],
      [1, 0],
      [0, 1],
      [1, 1],
      [0, 0],
      [1, 0]
    ],
    vertexColors: [
      [1, 1, 1],
      [1, 1, 1],
      [1, 1, 1],
      [1, 1, 1],
      [1, 1, 1],
      [1, 1, 1]
    ],
    // Vertex labels: 0=base, 1=low pig, 2=med pig, 3=high pig, 4=micro wrinkle, 5=regular wrinkle
    vertexLabels: [0, 1, 2, 3, 4, 5],
    textureMap: ''
  });

  describe('Filter Configuration', () => {
    it('should accept severity filter configuration prop', () => {
      const mesh = createTestMesh();
      const filterConfig: SeverityFilterConfig = {
        pigmentation: { low: true, medium: false, high: false },
        wrinkles: { micro: false, regular: false },
        allCombined: false
      };

      const { container } = render(
        <Viewer3D
          mesh={mesh}
          severityFilterConfig={filterConfig}
        />
      );

      expect(container).toBeInTheDocument();
    });

    it('should use default filter config when none provided', () => {
      const mesh = createTestMesh();

      const { container } = render(
        <Viewer3D mesh={mesh} />
      );

      expect(container).toBeInTheDocument();
    });
  });

  describe('Filter Behavior - Pigmentation', () => {
    it('should filter low severity pigmentation when disabled', async () => {
      const mesh = createTestMesh();
      const filterConfig: SeverityFilterConfig = {
        pigmentation: { low: false, medium: true, high: true },
        wrinkles: { micro: true, regular: true },
        allCombined: false
      };

      let meshObjects: THREE.Object3D[] = [];
      const onReady = jest.fn((scene, camera, objects) => {
        meshObjects = objects;
      });

      render(
        <Viewer3D
          mesh={mesh}
          severityFilterConfig={filterConfig}
          onReady={onReady}
        />
      );

      await waitFor(() => {
        expect(onReady).toHaveBeenCalled();
      });

      // Verify that pigmentation mesh exists
      const pigmentationMesh = meshObjects.find(obj => 
        obj instanceof THREE.Mesh && obj.position.z === 0.01
      ) as THREE.Mesh;

      expect(pigmentationMesh).toBeDefined();
      
      // Check that vertex colors are updated (low severity should be black/transparent)
      if (pigmentationMesh) {
        const geometry = pigmentationMesh.geometry as THREE.BufferGeometry;
        const colorAttribute = geometry.attributes.color;
        
        if (colorAttribute) {
          const colors = colorAttribute.array as Float32Array;
          // Vertex 1 has label 1 (low severity), should be black (0,0,0)
          expect(colors[1 * 3]).toBe(0);
          expect(colors[1 * 3 + 1]).toBe(0);
          expect(colors[1 * 3 + 2]).toBe(0);
        }
      }
    });

    it('should show medium severity pigmentation when enabled', async () => {
      const mesh = createTestMesh();
      const filterConfig: SeverityFilterConfig = {
        pigmentation: { low: false, medium: true, high: false },
        wrinkles: { micro: false, regular: false },
        allCombined: false
      };

      let meshObjects: THREE.Object3D[] = [];
      const onReady = jest.fn((scene, camera, objects) => {
        meshObjects = objects;
      });

      render(
        <Viewer3D
          mesh={mesh}
          severityFilterConfig={filterConfig}
          onReady={onReady}
        />
      );

      await waitFor(() => {
        expect(onReady).toHaveBeenCalled();
      });

      const pigmentationMesh = meshObjects.find(obj => 
        obj instanceof THREE.Mesh && obj.position.z === 0.01
      ) as THREE.Mesh;

      expect(pigmentationMesh).toBeDefined();
      
      if (pigmentationMesh) {
        const geometry = pigmentationMesh.geometry as THREE.BufferGeometry;
        const colorAttribute = geometry.attributes.color;
        
        if (colorAttribute) {
          const colors = colorAttribute.array as Float32Array;
          // Vertex 2 has label 2 (medium severity), should be orange (FFA500)
          const color = new THREE.Color(0xFFA500);
          expect(colors[2 * 3]).toBeCloseTo(color.r, 2);
          expect(colors[2 * 3 + 1]).toBeCloseTo(color.g, 2);
          expect(colors[2 * 3 + 2]).toBeCloseTo(color.b, 2);
        }
      }
    });

    it('should show high severity pigmentation when enabled', async () => {
      const mesh = createTestMesh();
      const filterConfig: SeverityFilterConfig = {
        pigmentation: { low: false, medium: false, high: true },
        wrinkles: { micro: false, regular: false },
        allCombined: false
      };

      let meshObjects: THREE.Object3D[] = [];
      const onReady = jest.fn((scene, camera, objects) => {
        meshObjects = objects;
      });

      render(
        <Viewer3D
          mesh={mesh}
          severityFilterConfig={filterConfig}
          onReady={onReady}
        />
      );

      await waitFor(() => {
        expect(onReady).toHaveBeenCalled();
      });

      const pigmentationMesh = meshObjects.find(obj => 
        obj instanceof THREE.Mesh && obj.position.z === 0.01
      ) as THREE.Mesh;

      expect(pigmentationMesh).toBeDefined();
      
      if (pigmentationMesh) {
        const geometry = pigmentationMesh.geometry as THREE.BufferGeometry;
        const colorAttribute = geometry.attributes.color;
        
        if (colorAttribute) {
          const colors = colorAttribute.array as Float32Array;
          // Vertex 3 has label 3 (high severity), should be dark red (8B0000)
          const color = new THREE.Color(0x8B0000);
          expect(colors[3 * 3]).toBeCloseTo(color.r, 2);
          expect(colors[3 * 3 + 1]).toBeCloseTo(color.g, 2);
          expect(colors[3 * 3 + 2]).toBeCloseTo(color.b, 2);
        }
      }
    });
  });

  describe('Filter Behavior - Wrinkles', () => {
    it('should filter micro wrinkles when disabled', async () => {
      const mesh = createTestMesh();
      const filterConfig: SeverityFilterConfig = {
        pigmentation: { low: false, medium: false, high: false },
        wrinkles: { micro: false, regular: true },
        allCombined: false
      };

      let meshObjects: THREE.Object3D[] = [];
      const onReady = jest.fn((scene, camera, objects) => {
        meshObjects = objects;
      });

      render(
        <Viewer3D
          mesh={mesh}
          severityFilterConfig={filterConfig}
          onReady={onReady}
        />
      );

      await waitFor(() => {
        expect(onReady).toHaveBeenCalled();
      });

      const wrinkleMesh = meshObjects.find(obj => 
        obj instanceof THREE.Mesh && obj.position.z === 0.02
      ) as THREE.Mesh;

      expect(wrinkleMesh).toBeDefined();
      
      if (wrinkleMesh) {
        const geometry = wrinkleMesh.geometry as THREE.BufferGeometry;
        const colorAttribute = geometry.attributes.color;
        
        if (colorAttribute) {
          const colors = colorAttribute.array as Float32Array;
          // Vertex 4 has label 4 (micro wrinkle), should be black (0,0,0)
          expect(colors[4 * 3]).toBe(0);
          expect(colors[4 * 3 + 1]).toBe(0);
          expect(colors[4 * 3 + 2]).toBe(0);
        }
      }
    });

    it('should show regular wrinkles when enabled', async () => {
      const mesh = createTestMesh();
      const filterConfig: SeverityFilterConfig = {
        pigmentation: { low: false, medium: false, high: false },
        wrinkles: { micro: false, regular: true },
        allCombined: false
      };

      let meshObjects: THREE.Object3D[] = [];
      const onReady = jest.fn((scene, camera, objects) => {
        meshObjects = objects;
      });

      render(
        <Viewer3D
          mesh={mesh}
          severityFilterConfig={filterConfig}
          onReady={onReady}
        />
      );

      await waitFor(() => {
        expect(onReady).toHaveBeenCalled();
      });

      const wrinkleMesh = meshObjects.find(obj => 
        obj instanceof THREE.Mesh && obj.position.z === 0.02
      ) as THREE.Mesh;

      expect(wrinkleMesh).toBeDefined();
      
      if (wrinkleMesh) {
        const geometry = wrinkleMesh.geometry as THREE.BufferGeometry;
        const colorAttribute = geometry.attributes.color;
        
        if (colorAttribute) {
          const colors = colorAttribute.array as Float32Array;
          // Vertex 5 has label 5 (regular wrinkle), should be dark blue (00008B)
          const color = new THREE.Color(0x00008B);
          expect(colors[5 * 3]).toBeCloseTo(color.r, 2);
          expect(colors[5 * 3 + 1]).toBeCloseTo(color.g, 2);
          expect(colors[5 * 3 + 2]).toBeCloseTo(color.b, 2);
        }
      }
    });
  });

  describe('Multi-select Filtering', () => {
    it('should show multiple severity levels when selected', async () => {
      const mesh = createTestMesh();
      const filterConfig: SeverityFilterConfig = {
        pigmentation: { low: true, medium: true, high: false },
        wrinkles: { micro: false, regular: false },
        allCombined: false
      };

      let meshObjects: THREE.Object3D[] = [];
      const onReady = jest.fn((scene, camera, objects) => {
        meshObjects = objects;
      });

      render(
        <Viewer3D
          mesh={mesh}
          severityFilterConfig={filterConfig}
          onReady={onReady}
        />
      );

      await waitFor(() => {
        expect(onReady).toHaveBeenCalled();
      });

      const pigmentationMesh = meshObjects.find(obj => 
        obj instanceof THREE.Mesh && obj.position.z === 0.01
      ) as THREE.Mesh;

      expect(pigmentationMesh).toBeDefined();
      
      if (pigmentationMesh) {
        const geometry = pigmentationMesh.geometry as THREE.BufferGeometry;
        const colorAttribute = geometry.attributes.color;
        
        if (colorAttribute) {
          const colors = colorAttribute.array as Float32Array;
          
          // Vertex 1 (low) should be visible
          const lowColor = new THREE.Color(0xFFE5B4);
          expect(colors[1 * 3]).toBeCloseTo(lowColor.r, 2);
          
          // Vertex 2 (medium) should be visible
          const medColor = new THREE.Color(0xFFA500);
          expect(colors[2 * 3]).toBeCloseTo(medColor.r, 2);
          
          // Vertex 3 (high) should be hidden (black)
          expect(colors[3 * 3]).toBe(0);
        }
      }
    });

    it('should show both pigmentation and wrinkles when selected', async () => {
      const mesh = createTestMesh();
      const filterConfig: SeverityFilterConfig = {
        pigmentation: { low: true, medium: false, high: false },
        wrinkles: { micro: true, regular: false },
        allCombined: false
      };

      let meshObjects: THREE.Object3D[] = [];
      const onReady = jest.fn((scene, camera, objects) => {
        meshObjects = objects;
      });

      render(
        <Viewer3D
          mesh={mesh}
          severityFilterConfig={filterConfig}
          onReady={onReady}
        />
      );

      await waitFor(() => {
        expect(onReady).toHaveBeenCalled();
      });

      // Should have both pigmentation and wrinkle meshes
      const pigmentationMesh = meshObjects.find(obj => 
        obj instanceof THREE.Mesh && obj.position.z === 0.01
      );
      const wrinkleMesh = meshObjects.find(obj => 
        obj instanceof THREE.Mesh && obj.position.z === 0.02
      );

      expect(pigmentationMesh).toBeDefined();
      expect(wrinkleMesh).toBeDefined();
    });
  });

  describe('All Combined Mode', () => {
    it('should show all anomalies when allCombined is true', async () => {
      const mesh = createTestMesh();
      const filterConfig: SeverityFilterConfig = {
        pigmentation: { low: true, medium: true, high: true },
        wrinkles: { micro: true, regular: true },
        allCombined: true
      };

      let meshObjects: THREE.Object3D[] = [];
      const onReady = jest.fn((scene, camera, objects) => {
        meshObjects = objects;
      });

      render(
        <Viewer3D
          mesh={mesh}
          severityFilterConfig={filterConfig}
          onReady={onReady}
        />
      );

      await waitFor(() => {
        expect(onReady).toHaveBeenCalled();
      });

      const pigmentationMesh = meshObjects.find(obj => 
        obj instanceof THREE.Mesh && obj.position.z === 0.01
      ) as THREE.Mesh;

      expect(pigmentationMesh).toBeDefined();
      
      if (pigmentationMesh) {
        const geometry = pigmentationMesh.geometry as THREE.BufferGeometry;
        const colorAttribute = geometry.attributes.color;
        
        if (colorAttribute) {
          const colors = colorAttribute.array as Float32Array;
          
          // All pigmentation vertices should be visible (not black)
          // Vertex 1 (low)
          expect(colors[1 * 3]).toBeGreaterThan(0);
          // Vertex 2 (medium)
          expect(colors[2 * 3]).toBeGreaterThan(0);
          // Vertex 3 (high)
          expect(colors[3 * 3]).toBeGreaterThan(0);
        }
      }
    });
  });

  describe('Requirements Validation', () => {
    it('should display only anomalies matching selected severity level (Req 5.2)', async () => {
      const mesh = createTestMesh();
      const filterConfig: SeverityFilterConfig = {
        pigmentation: { low: false, medium: true, high: false },
        wrinkles: { micro: false, regular: false },
        allCombined: false
      };

      let meshObjects: THREE.Object3D[] = [];
      const onReady = jest.fn((scene, camera, objects) => {
        meshObjects = objects;
      });

      render(
        <Viewer3D
          mesh={mesh}
          severityFilterConfig={filterConfig}
          onReady={onReady}
        />
      );

      await waitFor(() => {
        expect(onReady).toHaveBeenCalled();
      });

      // Only medium severity should be visible
      const pigmentationMesh = meshObjects.find(obj => 
        obj instanceof THREE.Mesh && obj.position.z === 0.01
      ) as THREE.Mesh;

      if (pigmentationMesh) {
        const geometry = pigmentationMesh.geometry as THREE.BufferGeometry;
        const colorAttribute = geometry.attributes.color;
        
        if (colorAttribute) {
          const colors = colorAttribute.array as Float32Array;
          
          // Low should be hidden
          expect(colors[1 * 3]).toBe(0);
          // Medium should be visible
          expect(colors[2 * 3]).toBeGreaterThan(0);
          // High should be hidden
          expect(colors[3 * 3]).toBe(0);
        }
      }
    });

    it('should hide anomalies when deselected (Req 5.3)', async () => {
      const mesh = createTestMesh();
      const filterConfig: SeverityFilterConfig = {
        pigmentation: { low: false, medium: false, high: false },
        wrinkles: { micro: false, regular: false },
        allCombined: false
      };

      let meshObjects: THREE.Object3D[] = [];
      const onReady = jest.fn((scene, camera, objects) => {
        meshObjects = objects;
      });

      render(
        <Viewer3D
          mesh={mesh}
          severityFilterConfig={filterConfig}
          onReady={onReady}
        />
      );

      await waitFor(() => {
        expect(onReady).toHaveBeenCalled();
      });

      // All anomalies should be hidden (black)
      const pigmentationMesh = meshObjects.find(obj => 
        obj instanceof THREE.Mesh && obj.position.z === 0.01
      ) as THREE.Mesh;

      if (pigmentationMesh) {
        const geometry = pigmentationMesh.geometry as THREE.BufferGeometry;
        const colorAttribute = geometry.attributes.color;
        
        if (colorAttribute) {
          const colors = colorAttribute.array as Float32Array;
          
          // All pigmentation vertices should be black
          expect(colors[1 * 3]).toBe(0);
          expect(colors[2 * 3]).toBe(0);
          expect(colors[3 * 3]).toBe(0);
        }
      }
    });

    it('should display anomalies matching any selected level (Req 5.4)', async () => {
      const mesh = createTestMesh();
      const filterConfig: SeverityFilterConfig = {
        pigmentation: { low: true, medium: false, high: true },
        wrinkles: { micro: false, regular: false },
        allCombined: false
      };

      let meshObjects: THREE.Object3D[] = [];
      const onReady = jest.fn((scene, camera, objects) => {
        meshObjects = objects;
      });

      render(
        <Viewer3D
          mesh={mesh}
          severityFilterConfig={filterConfig}
          onReady={onReady}
        />
      );

      await waitFor(() => {
        expect(onReady).toHaveBeenCalled();
      });

      const pigmentationMesh = meshObjects.find(obj => 
        obj instanceof THREE.Mesh && obj.position.z === 0.01
      ) as THREE.Mesh;

      if (pigmentationMesh) {
        const geometry = pigmentationMesh.geometry as THREE.BufferGeometry;
        const colorAttribute = geometry.attributes.color;
        
        if (colorAttribute) {
          const colors = colorAttribute.array as Float32Array;
          
          // Low should be visible
          expect(colors[1 * 3]).toBeGreaterThan(0);
          // Medium should be hidden
          expect(colors[2 * 3]).toBe(0);
          // High should be visible
          expect(colors[3 * 3]).toBeGreaterThan(0);
        }
      }
    });
  });
});
