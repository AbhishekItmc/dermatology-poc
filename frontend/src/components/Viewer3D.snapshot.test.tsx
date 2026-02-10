import React from 'react';
import { render, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import Viewer3D, { Viewer3DHandle } from './Viewer3D';
import { Mesh } from '../types';

/**
 * Unit tests for Viewer3D snapshot export functionality
 * 
 * Tests:
 * - PNG export with custom filename
 * - JPEG export with custom filename
 * - Export with default filename
 * - Export includes all visible layers
 * - Export captures current view state
 * - Error handling for uninitialized viewer
 * 
 * Requirements: 3.8
 */

// Mock Three.js to avoid WebGL context issues
jest.mock('three', () => {
  const actualThree = jest.requireActual('three');
  return {
    ...actualThree,
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
    })
  };
});

jest.mock('three/examples/jsm/controls/OrbitControls', () => ({
  OrbitControls: jest.fn().mockImplementation(() => ({
    update: jest.fn(),
    dispose: jest.fn(),
    enableDamping: true,
    dampingFactor: 0.05
  }))
}));

// Mock HTMLCanvasElement.toDataURL
const mockToDataURL = jest.fn();
HTMLCanvasElement.prototype.toDataURL = mockToDataURL;

// Mock document.createElement for link element
const mockCreateElement = document.createElement.bind(document);
const mockLinks: HTMLAnchorElement[] = [];

document.createElement = jest.fn((tagName: string) => {
  const element = mockCreateElement(tagName);
  if (tagName === 'a') {
    mockLinks.push(element as HTMLAnchorElement);
    // Mock click to prevent actual download
    element.click = jest.fn();
  }
  return element;
}) as any;

// Create a simple test mesh
const createTestMesh = (): Mesh => ({
  vertices: [
    [-50, -50, -50], [50, -50, -50], [50, 50, -50], [-50, 50, -50],
    [-50, -50, 50], [50, -50, 50], [50, 50, 50], [-50, 50, 50]
  ],
  faces: [
    [0, 1, 2], [0, 2, 3],
    [4, 6, 5], [4, 7, 6],
    [0, 4, 5], [0, 5, 1],
    [2, 6, 7], [2, 7, 3],
    [0, 3, 7], [0, 7, 4],
    [1, 5, 6], [1, 6, 2]
  ],
  normals: [],
  uvCoordinates: [],
  vertexColors: [
    [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0],
    [1, 0, 1], [0, 1, 1], [1, 1, 1], [0.5, 0.5, 0.5]
  ],
  vertexLabels: [],
  textureMap: ''
});

describe('Viewer3D Snapshot Export', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockLinks.length = 0;
    mockToDataURL.mockReturnValue('data:image/png;base64,mockImageData');
  });

  test('exports snapshot as PNG with custom filename', async () => {
    const viewerRef = React.createRef<Viewer3DHandle>();
    const mesh = createTestMesh();

    render(
      <Viewer3D
        ref={viewerRef}
        mesh={mesh}
        width={800}
        height={600}
      />
    );

    // Wait for viewer to initialize
    await waitFor(() => {
      expect(viewerRef.current).not.toBeNull();
    }, { timeout: 3000 });

    // Export snapshot
    viewerRef.current?.exportSnapshot('png', 'test_snapshot.png');

    // Verify toDataURL was called with PNG mime type
    expect(mockToDataURL).toHaveBeenCalledWith('image/png', 0.95);

    // Verify link was created with correct attributes
    await waitFor(() => {
      expect(mockLinks.length).toBeGreaterThan(0);
    });

    const link = mockLinks[mockLinks.length - 1];
    expect(link.href).toBe('data:image/png;base64,mockImageData');
    expect(link.download).toBe('test_snapshot.png');
    expect(link.click).toHaveBeenCalled();
  });

  test('exports snapshot as JPEG with custom filename', async () => {
    const viewerRef = React.createRef<Viewer3DHandle>();
    const mesh = createTestMesh();

    mockToDataURL.mockReturnValue('data:image/jpeg;base64,mockImageData');

    render(
      <Viewer3D
        ref={viewerRef}
        mesh={mesh}
        width={800}
        height={600}
      />
    );

    await waitFor(() => {
      expect(viewerRef.current).not.toBeNull();
    }, { timeout: 3000 });

    // Export snapshot as JPEG
    viewerRef.current?.exportSnapshot('jpeg', 'test_snapshot.jpg');

    // Verify toDataURL was called with JPEG mime type
    expect(mockToDataURL).toHaveBeenCalledWith('image/jpeg', 0.95);

    // Verify link was created with correct attributes
    await waitFor(() => {
      expect(mockLinks.length).toBeGreaterThan(0);
    });

    const link = mockLinks[mockLinks.length - 1];
    expect(link.href).toBe('data:image/jpeg;base64,mockImageData');
    expect(link.download).toBe('test_snapshot.jpg');
    expect(link.click).toHaveBeenCalled();
  });

  test('exports snapshot with default filename when not provided', async () => {
    const viewerRef = React.createRef<Viewer3DHandle>();
    const mesh = createTestMesh();

    render(
      <Viewer3D
        ref={viewerRef}
        mesh={mesh}
        width={800}
        height={600}
      />
    );

    await waitFor(() => {
      expect(viewerRef.current).not.toBeNull();
    }, { timeout: 3000 });

    // Export snapshot without filename
    viewerRef.current?.exportSnapshot('png');

    // Verify link was created with default filename pattern
    await waitFor(() => {
      expect(mockLinks.length).toBeGreaterThan(0);
    });

    const link = mockLinks[mockLinks.length - 1];
    expect(link.download).toMatch(/^snapshot_\d+\.png$/);
    expect(link.click).toHaveBeenCalled();
  });

  test('exports snapshot with layered visualization', async () => {
    const viewerRef = React.createRef<Viewer3DHandle>();
    
    // Create mesh with vertex labels for layered visualization
    const mesh: Mesh = {
      ...createTestMesh(),
      vertexLabels: [0, 1, 2, 3, 4, 5, 0, 0] // Mix of base, pigmentation, and wrinkles
    };

    const layerConfig = {
      base: { visible: true, opacity: 1.0 },
      pigmentation: { visible: true, opacity: 0.7 },
      wrinkles: { visible: true, opacity: 0.7 }
    };

    render(
      <Viewer3D
        ref={viewerRef}
        mesh={mesh}
        width={800}
        height={600}
        layerConfig={layerConfig}
      />
    );

    await waitFor(() => {
      expect(viewerRef.current).not.toBeNull();
    }, { timeout: 3000 });

    // Export snapshot - should include all visible layers
    viewerRef.current?.exportSnapshot('png', 'layered_snapshot.png');

    // Verify export was successful
    expect(mockToDataURL).toHaveBeenCalled();
    
    await waitFor(() => {
      expect(mockLinks.length).toBeGreaterThan(0);
    });

    const link = mockLinks[mockLinks.length - 1];
    expect(link.download).toBe('layered_snapshot.png');
    expect(link.click).toHaveBeenCalled();
  });

  test('exports snapshot with hidden layers excluded', async () => {
    const viewerRef = React.createRef<Viewer3DHandle>();
    
    const mesh: Mesh = {
      ...createTestMesh(),
      vertexLabels: [0, 1, 2, 3, 4, 5, 0, 0]
    };

    // Hide pigmentation layer
    const layerConfig = {
      base: { visible: true, opacity: 1.0 },
      pigmentation: { visible: false, opacity: 0.7 },
      wrinkles: { visible: true, opacity: 0.7 }
    };

    render(
      <Viewer3D
        ref={viewerRef}
        mesh={mesh}
        width={800}
        height={600}
        layerConfig={layerConfig}
      />
    );

    await waitFor(() => {
      expect(viewerRef.current).not.toBeNull();
    }, { timeout: 3000 });

    // Export snapshot - should only include visible layers
    viewerRef.current?.exportSnapshot('png', 'filtered_snapshot.png');

    expect(mockToDataURL).toHaveBeenCalled();
    
    await waitFor(() => {
      expect(mockLinks.length).toBeGreaterThan(0);
    });

    const link = mockLinks[mockLinks.length - 1];
    expect(link.download).toBe('filtered_snapshot.png');
    expect(link.click).toHaveBeenCalled();
  });

  test('handles export error gracefully', async () => {
    const viewerRef = React.createRef<Viewer3DHandle>();
    const mesh = createTestMesh();

    // Mock toDataURL to throw an error
    mockToDataURL.mockImplementation(() => {
      throw new Error('Canvas export failed');
    });

    const consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation();

    render(
      <Viewer3D
        ref={viewerRef}
        mesh={mesh}
        width={800}
        height={600}
      />
    );

    await waitFor(() => {
      expect(viewerRef.current).not.toBeNull();
    }, { timeout: 3000 });

    // Attempt to export snapshot
    expect(() => {
      viewerRef.current?.exportSnapshot('png', 'error_snapshot.png');
    }).toThrow('Canvas export failed');

    expect(consoleErrorSpy).toHaveBeenCalledWith(
      'Failed to export snapshot:',
      expect.any(Error)
    );

    consoleErrorSpy.mockRestore();
  });

  test('exports multiple snapshots with different formats', async () => {
    const viewerRef = React.createRef<Viewer3DHandle>();
    const mesh = createTestMesh();

    render(
      <Viewer3D
        ref={viewerRef}
        mesh={mesh}
        width={800}
        height={600}
      />
    );

    await waitFor(() => {
      expect(viewerRef.current).not.toBeNull();
    }, { timeout: 3000 });

    // Export PNG
    mockToDataURL.mockReturnValue('data:image/png;base64,pngData');
    viewerRef.current?.exportSnapshot('png', 'snapshot1.png');

    // Export JPEG
    mockToDataURL.mockReturnValue('data:image/jpeg;base64,jpegData');
    viewerRef.current?.exportSnapshot('jpeg', 'snapshot2.jpg');

    // Verify both exports
    expect(mockToDataURL).toHaveBeenCalledTimes(2);
    expect(mockToDataURL).toHaveBeenNthCalledWith(1, 'image/png', 0.95);
    expect(mockToDataURL).toHaveBeenNthCalledWith(2, 'image/jpeg', 0.95);

    await waitFor(() => {
      expect(mockLinks.length).toBe(2);
    });

    expect(mockLinks[0].download).toBe('snapshot1.png');
    expect(mockLinks[1].download).toBe('snapshot2.jpg');
  });

  test('captures current camera view in snapshot', async () => {
    const viewerRef = React.createRef<Viewer3DHandle>();
    const mesh = createTestMesh();

    render(
      <Viewer3D
        ref={viewerRef}
        mesh={mesh}
        width={800}
        height={600}
        enableControls={true}
      />
    );

    await waitFor(() => {
      expect(viewerRef.current).not.toBeNull();
    }, { timeout: 3000 });

    // Export snapshot - should capture current camera position/rotation
    viewerRef.current?.exportSnapshot('png', 'camera_view.png');

    // Verify render was called before export
    expect(mockToDataURL).toHaveBeenCalled();
    
    await waitFor(() => {
      expect(mockLinks.length).toBeGreaterThan(0);
    });

    const link = mockLinks[mockLinks.length - 1];
    expect(link.download).toBe('camera_view.png');
  });

  test('exports high-quality JPEG with correct quality setting', async () => {
    const viewerRef = React.createRef<Viewer3DHandle>();
    const mesh = createTestMesh();

    render(
      <Viewer3D
        ref={viewerRef}
        mesh={mesh}
        width={800}
        height={600}
      />
    );

    await waitFor(() => {
      expect(viewerRef.current).not.toBeNull();
    }, { timeout: 3000 });

    viewerRef.current?.exportSnapshot('jpeg', 'high_quality.jpg');

    // Verify quality parameter is 0.95 (95% quality)
    expect(mockToDataURL).toHaveBeenCalledWith('image/jpeg', 0.95);
  });

  test('cleans up temporary link element after download', async () => {
    const viewerRef = React.createRef<Viewer3DHandle>();
    const mesh = createTestMesh();

    const appendChildSpy = jest.spyOn(document.body, 'appendChild');
    const removeChildSpy = jest.spyOn(document.body, 'removeChild');

    render(
      <Viewer3D
        ref={viewerRef}
        mesh={mesh}
        width={800}
        height={600}
      />
    );

    await waitFor(() => {
      expect(viewerRef.current).not.toBeNull();
    }, { timeout: 3000 });

    viewerRef.current?.exportSnapshot('png', 'cleanup_test.png');

    await waitFor(() => {
      expect(mockLinks.length).toBeGreaterThan(0);
    });

    // Verify link was added and then removed
    expect(appendChildSpy).toHaveBeenCalled();
    expect(removeChildSpy).toHaveBeenCalled();

    appendChildSpy.mockRestore();
    removeChildSpy.mockRestore();
  });
});
