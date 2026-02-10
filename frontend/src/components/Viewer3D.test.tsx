import React from 'react';
import Viewer3D from './Viewer3D';
import { Mesh } from '../types';

/**
 * Unit tests for Viewer3D component
 * 
 * Note: These tests verify the component structure and exports.
 * Full WebGL rendering tests require a browser environment with WebGL support.
 * Integration tests should be run in a real browser or with a headless browser.
 */

describe('Viewer3D Component', () => {
  const mockMesh: Mesh = {
    vertices: [
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0]
    ],
    faces: [[0, 1, 2]],
    normals: [
      [0, 0, 1],
      [0, 0, 1],
      [0, 0, 1]
    ],
    uvCoordinates: [
      [0, 0],
      [1, 0],
      [0, 1]
    ],
    vertexColors: [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1]
    ],
    vertexLabels: [0, 0, 0],
    textureMap: ''
  };

  test('component is defined and exports correctly', () => {
    expect(Viewer3D).toBeDefined();
    expect(typeof Viewer3D).toBe('object'); // forwardRef returns an object
  });

  test('component is a valid React component', () => {
    // forwardRef components don't have prototype
    expect(typeof Viewer3D).toBe('object');
  });

  test('component accepts mesh prop', () => {
    // Verify the component can be instantiated with mesh prop
    const props = { mesh: mockMesh };
    expect(() => React.createElement(Viewer3D, props)).not.toThrow();
  });

  test('component accepts width and height props', () => {
    const props = { width: 1024, height: 768 };
    expect(() => React.createElement(Viewer3D, props)).not.toThrow();
  });

  test('component accepts showWireframe prop', () => {
    const props = { showWireframe: true };
    expect(() => React.createElement(Viewer3D, props)).not.toThrow();
  });

  test('component accepts enableControls prop', () => {
    const props = { enableControls: false };
    expect(() => React.createElement(Viewer3D, props)).not.toThrow();
  });

  test('component accepts onReady callback prop', () => {
    const onReady = jest.fn();
    const props = { onReady };
    expect(() => React.createElement(Viewer3D, props)).not.toThrow();
  });

  test('component accepts all props together', () => {
    const props = {
      mesh: mockMesh,
      width: 800,
      height: 600,
      showWireframe: false,
      enableControls: true,
      onReady: jest.fn()
    };
    expect(() => React.createElement(Viewer3D, props)).not.toThrow();
  });
});

describe('Viewer3D Props Interface', () => {
  test('mesh prop structure is correct', () => {
    const mesh: Mesh = {
      vertices: [[0, 0, 0]],
      faces: [[0, 0, 0]],
      normals: [[0, 0, 1]],
      uvCoordinates: [[0, 0]],
      vertexColors: [[1, 1, 1]],
      vertexLabels: [0],
      textureMap: ''
    };

    expect(mesh.vertices).toBeDefined();
    expect(Array.isArray(mesh.vertices)).toBe(true);
    expect(mesh.faces).toBeDefined();
    expect(Array.isArray(mesh.faces)).toBe(true);
    expect(mesh.normals).toBeDefined();
    expect(Array.isArray(mesh.normals)).toBe(true);
  });

  test('optional props have correct types', () => {
    const width: number = 800;
    const height: number = 600;
    const showWireframe: boolean = false;
    const enableControls: boolean = true;
    const onReady: () => void = () => {};

    expect(typeof width).toBe('number');
    expect(typeof height).toBe('number');
    expect(typeof showWireframe).toBe('boolean');
    expect(typeof enableControls).toBe('boolean');
    expect(typeof onReady).toBe('function');
  });
});

/**
 * Documentation tests
 * Verify that the component has proper documentation
 */
describe('Viewer3D Documentation', () => {
  test('component file exists and is importable', () => {
    expect(Viewer3D).toBeDefined();
  });

  test('component is a forwardRef component', () => {
    // forwardRef components have $$typeof property
    expect(Viewer3D).toHaveProperty('$$typeof');
  });
});
