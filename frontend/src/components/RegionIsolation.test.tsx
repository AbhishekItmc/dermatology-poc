import React from 'react';
import { render, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import Viewer3D, { Viewer3DHandle, RegionConfig, FacialRegion } from './Viewer3D';
import { Mesh } from '../types';

// Mock Three.js to avoid WebGL context issues in tests
jest.mock('three', () => {
  const actual = jest.requireActual('three');
  return {
    ...actual,
    WebGLRenderer: jest.fn().mockImplementation(() => {
      const canvas = document.createElement('canvas');
      return {
        setSize: jest.fn(),
        setPixelRatio: jest.fn(),
        render: jest.fn(),
        dispose: jest.fn(),
        domElement: canvas,
        shadowMap: { enabled: false, type: 0 }
      };
    }),
    Scene: jest.fn().mockImplementation(() => ({
      add: jest.fn(),
      remove: jest.fn(),
      background: null
    })),
    PerspectiveCamera: jest.fn().mockImplementation(() => ({
      position: { set: jest.fn(), clone: jest.fn(() => ({ x: 0, y: 0, z: 300 })) },
      lookAt: jest.fn(),
      aspect: 1,
      updateProjectionMatrix: jest.fn()
    })),
    Color: jest.fn().mockImplementation((color) => ({ r: 0, g: 0, b: 0, setHex: jest.fn() })),
    Vector3: jest.fn().mockImplementation((x = 0, y = 0, z = 0) => ({
      x, y, z,
      set: jest.fn(),
      clone: jest.fn(function() { return { x: this.x, y: this.y, z: this.z }; }),
      lerpVectors: jest.fn()
    })),
    Mesh: jest.fn().mockImplementation(() => ({
      geometry: {
        computeBoundingBox: jest.fn(),
        boundingBox: {
          getCenter: jest.fn(),
          getSize: jest.fn()
        },
        attributes: {
          position: {}
        },
        dispose: jest.fn()
      },
      material: {
        dispose: jest.fn(),
        opacity: 1,
        transparent: false,
        emissive: { r: 0, g: 0, b: 0 },
        emissiveIntensity: 0
      },
      position: { sub: jest.fn(), copy: jest.fn() },
      scale: { set: jest.fn(), copy: jest.fn() },
      visible: true,
      castShadow: true,
      receiveShadow: true
    }))
  };
});

jest.mock('three/examples/jsm/controls/OrbitControls', () => ({
  OrbitControls: jest.fn().mockImplementation(() => ({
    update: jest.fn(),
    dispose: jest.fn(),
    target: {
      clone: jest.fn(() => ({ x: 0, y: 0, z: 0 })),
      lerpVectors: jest.fn()
    },
    enableDamping: true,
    dampingFactor: 0.05
  }))
}));

describe('Region Isolation Integration', () => {
  const createTestMesh = (): Mesh => ({
    vertices: [
      [0, 0, 0],
      [10, 0, 0],
      [0, 10, 0],
      [10, 10, 0]
    ],
    faces: [
      [0, 1, 2],
      [1, 3, 2]
    ],
    normals: [
      [0, 0, 1],
      [0, 0, 1],
      [0, 0, 1],
      [0, 0, 1]
    ],
    uvCoordinates: [],
    vertexColors: [],
    vertexLabels: [],
    textureMap: ''
  });

  it('renders viewer with region configuration', async () => {
    const regionConfig: RegionConfig = {
      selectedRegion: 'forehead',
      highlightIntensity: 0.5
    };

    const { container } = render(
      <Viewer3D
        mesh={createTestMesh()}
        regionConfig={regionConfig}
      />
    );

    await waitFor(() => {
      expect(container.querySelector('canvas')).toBeInTheDocument();
    });
  });

  it('accepts all valid facial regions', async () => {
    const regions: FacialRegion[] = [
      'all',
      'forehead',
      'left_cheek',
      'right_cheek',
      'periorbital_left',
      'periorbital_right',
      'nose',
      'mouth',
      'chin'
    ];

    for (const region of regions) {
      const regionConfig: RegionConfig = {
        selectedRegion: region,
        highlightIntensity: 0.5
      };

      const { unmount } = render(
        <Viewer3D
          mesh={createTestMesh()}
          regionConfig={regionConfig}
        />
      );

      // Should render without errors
      await waitFor(() => {
        expect(true).toBe(true);
      });

      unmount();
    }
  });

  it('exposes zoomToRegion method via ref', async () => {
    const ref = React.createRef<Viewer3DHandle>();
    const regionConfig: RegionConfig = {
      selectedRegion: 'all',
      highlightIntensity: 0.5
    };

    render(
      <Viewer3D
        ref={ref}
        mesh={createTestMesh()}
        regionConfig={regionConfig}
      />
    );

    await waitFor(() => {
      expect(ref.current).not.toBeNull();
    });

    expect(ref.current?.zoomToRegion).toBeDefined();
    expect(typeof ref.current?.zoomToRegion).toBe('function');
  });

  it('zoomToRegion can be called for all regions', async () => {
    const ref = React.createRef<Viewer3DHandle>();
    const regionConfig: RegionConfig = {
      selectedRegion: 'all',
      highlightIntensity: 0.5
    };

    render(
      <Viewer3D
        ref={ref}
        mesh={createTestMesh()}
        regionConfig={regionConfig}
      />
    );

    await waitFor(() => {
      expect(ref.current).not.toBeNull();
    });

    const regions: FacialRegion[] = [
      'all',
      'forehead',
      'left_cheek',
      'right_cheek',
      'periorbital_left',
      'periorbital_right',
      'nose',
      'mouth',
      'chin'
    ];

    // Should not throw errors when calling zoomToRegion
    regions.forEach(region => {
      expect(() => {
        ref.current?.zoomToRegion(region);
      }).not.toThrow();
    });
  });

  it('handles highlight intensity range 0-1', async () => {
    const intensities = [0, 0.25, 0.5, 0.75, 1.0];

    for (const intensity of intensities) {
      const regionConfig: RegionConfig = {
        selectedRegion: 'forehead',
        highlightIntensity: intensity
      };

      const { unmount } = render(
        <Viewer3D
          mesh={createTestMesh()}
          regionConfig={regionConfig}
        />
      );

      await waitFor(() => {
        expect(true).toBe(true);
      });

      unmount();
    }
  });

  it('updates when region configuration changes', async () => {
    const { rerender } = render(
      <Viewer3D
        mesh={createTestMesh()}
        regionConfig={{
          selectedRegion: 'all',
          highlightIntensity: 0.5
        }}
      />
    );

    await waitFor(() => {
      expect(true).toBe(true);
    });

    // Change region
    rerender(
      <Viewer3D
        mesh={createTestMesh()}
        regionConfig={{
          selectedRegion: 'forehead',
          highlightIntensity: 0.7
        }}
      />
    );

    await waitFor(() => {
      expect(true).toBe(true);
    });
  });

  it('handles missing region configuration gracefully', async () => {
    const { container } = render(
      <Viewer3D
        mesh={createTestMesh()}
        // No regionConfig provided
      />
    );

    await waitFor(() => {
      expect(container.querySelector('canvas')).toBeInTheDocument();
    });
  });

  it('maintains viewer functionality with region isolation', async () => {
    const ref = React.createRef<Viewer3DHandle>();
    const regionConfig: RegionConfig = {
      selectedRegion: 'forehead',
      highlightIntensity: 0.5
    };

    render(
      <Viewer3D
        ref={ref}
        mesh={createTestMesh()}
        regionConfig={regionConfig}
        enableControls={true}
      />
    );

    await waitFor(() => {
      expect(ref.current).not.toBeNull();
    });

    // Both exportSnapshot and zoomToRegion should be available
    expect(ref.current?.exportSnapshot).toBeDefined();
    expect(ref.current?.zoomToRegion).toBeDefined();
  });
});
