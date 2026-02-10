import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import Viewer3D, { LayerConfig } from './Viewer3D';
import LayerControls from './LayerControls';
import { Mesh } from '../types';

// Mock Three.js to avoid WebGL context issues in tests
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
    }),
    Scene: jest.fn().mockImplementation(() => ({
      add: jest.fn(),
      remove: jest.fn(),
      background: null
    })),
    PerspectiveCamera: jest.fn().mockImplementation(() => ({
      position: { set: jest.fn(), copy: jest.fn() },
      lookAt: jest.fn(),
      aspect: 1,
      updateProjectionMatrix: jest.fn()
    })),
    Mesh: jest.fn().mockImplementation(() => ({
      position: { set: jest.fn(), sub: jest.fn(), copy: jest.fn(), z: 0 },
      scale: { set: jest.fn(), copy: jest.fn() },
      geometry: {
        computeBoundingBox: jest.fn(),
        boundingBox: {
          getCenter: jest.fn(),
          getSize: jest.fn()
        },
        dispose: jest.fn()
      },
      material: {
        dispose: jest.fn(),
        opacity: 1,
        transparent: false
      },
      visible: true,
      castShadow: false,
      receiveShadow: false
    })),
    BufferGeometry: jest.fn().mockImplementation(() => ({
      setAttribute: jest.fn(),
      setIndex: jest.fn(),
      computeVertexNormals: jest.fn(),
      computeBoundingBox: jest.fn(),
      dispose: jest.fn(),
      boundingBox: {
        getCenter: jest.fn((v) => v.set(0, 0, 0)),
        getSize: jest.fn((v) => v.set(10, 10, 10))
      }
    })),
    BufferAttribute: jest.fn(),
    MeshPhongMaterial: jest.fn().mockImplementation(() => ({
      dispose: jest.fn(),
      opacity: 1,
      transparent: false
    })),
    Color: jest.fn().mockImplementation(() => ({
      setHex: jest.fn(),
      r: 1,
      g: 1,
      b: 1
    })),
    Vector3: jest.fn().mockImplementation(() => ({
      set: jest.fn(),
      sub: jest.fn(),
      copy: jest.fn()
    })),
    AmbientLight: jest.fn(),
    DirectionalLight: jest.fn().mockImplementation(() => ({
      position: { set: jest.fn() },
      castShadow: false,
      shadow: {
        mapSize: { width: 0, height: 0 },
        camera: { near: 0, far: 0 }
      }
    })),
    HemisphereLight: jest.fn().mockImplementation(() => ({
      position: { set: jest.fn() }
    })),
    GridHelper: jest.fn().mockImplementation(() => ({
      position: { y: 0 }
    })),
    AxesHelper: jest.fn(),
    TextureLoader: jest.fn().mockImplementation(() => ({
      load: jest.fn()
    })),
    SRGBColorSpace: 'srgb',
    DoubleSide: 2,
    PCFSoftShadowMap: 1
  };
});

jest.mock('three/examples/jsm/controls/OrbitControls', () => ({
  OrbitControls: jest.fn().mockImplementation(() => ({
    update: jest.fn(),
    dispose: jest.fn(),
    enableDamping: true,
    dampingFactor: 0.05,
    screenSpacePanning: false,
    minDistance: 50,
    maxDistance: 500,
    maxPolarAngle: Math.PI
  }))
}));

describe('Layered Visualization', () => {
  const createMockMesh = (withLabels: boolean = true): Mesh => ({
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
      [3, 0, 4]
    ],
    normals: [],
    uvCoordinates: [],
    vertexColors: [],
    vertexLabels: withLabels ? [0, 1, 2, 3, 0, 4, 5, 0] : [],
    textureMap: ''
  });

  describe('Viewer3D with Layers', () => {
    it('should render without crashing', () => {
      const mesh = createMockMesh();
      const { container } = render(<Viewer3D mesh={mesh} />);
      expect(container).toBeInTheDocument();
    });

    it('should accept layerConfig prop', () => {
      const mesh = createMockMesh();
      const layerConfig: LayerConfig = {
        base: { visible: true, opacity: 1.0 },
        pigmentation: { visible: true, opacity: 0.7 },
        wrinkles: { visible: true, opacity: 0.7 }
      };
      
      const { container } = render(
        <Viewer3D mesh={mesh} layerConfig={layerConfig} />
      );
      expect(container).toBeInTheDocument();
    });

    it('should handle mesh without vertex labels', () => {
      const mesh = createMockMesh(false);
      const { container } = render(<Viewer3D mesh={mesh} />);
      expect(container).toBeInTheDocument();
    });

    it('should call onReady callback when initialized', async () => {
      const mesh = createMockMesh();
      const onReady = jest.fn();
      
      render(<Viewer3D mesh={mesh} onReady={onReady} />);
      
      await waitFor(() => {
        expect(onReady).toHaveBeenCalled();
      });
    });
  });

  describe('LayerControls', () => {
    it('should render all layer controls', () => {
      const layerConfig: LayerConfig = {
        base: { visible: true, opacity: 1.0 },
        pigmentation: { visible: true, opacity: 0.7 },
        wrinkles: { visible: true, opacity: 0.7 }
      };
      const onLayerConfigChange = jest.fn();
      
      render(
        <LayerControls
          layerConfig={layerConfig}
          onLayerConfigChange={onLayerConfigChange}
        />
      );
      
      expect(screen.getByText('Layer Controls')).toBeInTheDocument();
      expect(screen.getByText('Base Mesh')).toBeInTheDocument();
      expect(screen.getByText('Pigmentation')).toBeInTheDocument();
      expect(screen.getByText('Wrinkles')).toBeInTheDocument();
    });

    it('should toggle base layer visibility', () => {
      const layerConfig: LayerConfig = {
        base: { visible: true, opacity: 1.0 },
        pigmentation: { visible: true, opacity: 0.7 },
        wrinkles: { visible: true, opacity: 0.7 }
      };
      const onLayerConfigChange = jest.fn();
      
      render(
        <LayerControls
          layerConfig={layerConfig}
          onLayerConfigChange={onLayerConfigChange}
        />
      );
      
      const baseCheckbox = screen.getByRole('checkbox', { name: /Base Mesh/i });
      fireEvent.click(baseCheckbox);
      
      expect(onLayerConfigChange).toHaveBeenCalledWith({
        base: { visible: false, opacity: 1.0 },
        pigmentation: { visible: true, opacity: 0.7 },
        wrinkles: { visible: true, opacity: 0.7 }
      });
    });

    it('should toggle pigmentation layer visibility', () => {
      const layerConfig: LayerConfig = {
        base: { visible: true, opacity: 1.0 },
        pigmentation: { visible: true, opacity: 0.7 },
        wrinkles: { visible: true, opacity: 0.7 }
      };
      const onLayerConfigChange = jest.fn();
      
      render(
        <LayerControls
          layerConfig={layerConfig}
          onLayerConfigChange={onLayerConfigChange}
        />
      );
      
      const pigmentationCheckbox = screen.getByRole('checkbox', { name: /Pigmentation/i });
      fireEvent.click(pigmentationCheckbox);
      
      expect(onLayerConfigChange).toHaveBeenCalledWith({
        base: { visible: true, opacity: 1.0 },
        pigmentation: { visible: false, opacity: 0.7 },
        wrinkles: { visible: true, opacity: 0.7 }
      });
    });

    it('should toggle wrinkles layer visibility', () => {
      const layerConfig: LayerConfig = {
        base: { visible: true, opacity: 1.0 },
        pigmentation: { visible: true, opacity: 0.7 },
        wrinkles: { visible: true, opacity: 0.7 }
      };
      const onLayerConfigChange = jest.fn();
      
      render(
        <LayerControls
          layerConfig={layerConfig}
          onLayerConfigChange={onLayerConfigChange}
        />
      );
      
      const wrinklesCheckbox = screen.getByRole('checkbox', { name: /Wrinkles/i });
      fireEvent.click(wrinklesCheckbox);
      
      expect(onLayerConfigChange).toHaveBeenCalledWith({
        base: { visible: true, opacity: 1.0 },
        pigmentation: { visible: true, opacity: 0.7 },
        wrinkles: { visible: false, opacity: 0.7 }
      });
    });

    it('should adjust base layer opacity', () => {
      const layerConfig: LayerConfig = {
        base: { visible: true, opacity: 1.0 },
        pigmentation: { visible: true, opacity: 0.7 },
        wrinkles: { visible: true, opacity: 0.7 }
      };
      const onLayerConfigChange = jest.fn();
      
      render(
        <LayerControls
          layerConfig={layerConfig}
          onLayerConfigChange={onLayerConfigChange}
        />
      );
      
      const sliders = screen.getAllByRole('slider');
      const baseSlider = sliders[0]; // First slider is for base layer
      
      fireEvent.change(baseSlider, { target: { value: '50' } });
      
      expect(onLayerConfigChange).toHaveBeenCalledWith({
        base: { visible: true, opacity: 0.5 },
        pigmentation: { visible: true, opacity: 0.7 },
        wrinkles: { visible: true, opacity: 0.7 }
      });
    });

    it('should adjust pigmentation layer opacity', () => {
      const layerConfig: LayerConfig = {
        base: { visible: true, opacity: 1.0 },
        pigmentation: { visible: true, opacity: 0.7 },
        wrinkles: { visible: true, opacity: 0.7 }
      };
      const onLayerConfigChange = jest.fn();
      
      render(
        <LayerControls
          layerConfig={layerConfig}
          onLayerConfigChange={onLayerConfigChange}
        />
      );
      
      const sliders = screen.getAllByRole('slider');
      const pigmentationSlider = sliders[1]; // Second slider is for pigmentation layer
      
      fireEvent.change(pigmentationSlider, { target: { value: '30' } });
      
      expect(onLayerConfigChange).toHaveBeenCalledWith({
        base: { visible: true, opacity: 1.0 },
        pigmentation: { visible: true, opacity: 0.3 },
        wrinkles: { visible: true, opacity: 0.7 }
      });
    });

    it('should adjust wrinkles layer opacity', () => {
      const layerConfig: LayerConfig = {
        base: { visible: true, opacity: 1.0 },
        pigmentation: { visible: true, opacity: 0.7 },
        wrinkles: { visible: true, opacity: 0.7 }
      };
      const onLayerConfigChange = jest.fn();
      
      render(
        <LayerControls
          layerConfig={layerConfig}
          onLayerConfigChange={onLayerConfigChange}
        />
      );
      
      const sliders = screen.getAllByRole('slider');
      const wrinklesSlider = sliders[2]; // Third slider is for wrinkles layer
      
      fireEvent.change(wrinklesSlider, { target: { value: '90' } });
      
      expect(onLayerConfigChange).toHaveBeenCalledWith({
        base: { visible: true, opacity: 1.0 },
        pigmentation: { visible: true, opacity: 0.7 },
        wrinkles: { visible: true, opacity: 0.9 }
      });
    });

    it('should show all layers when "Show All" button is clicked', () => {
      const layerConfig: LayerConfig = {
        base: { visible: false, opacity: 0.5 },
        pigmentation: { visible: false, opacity: 0.3 },
        wrinkles: { visible: false, opacity: 0.2 }
      };
      const onLayerConfigChange = jest.fn();
      
      render(
        <LayerControls
          layerConfig={layerConfig}
          onLayerConfigChange={onLayerConfigChange}
        />
      );
      
      const showAllButton = screen.getByText('Show All');
      fireEvent.click(showAllButton);
      
      expect(onLayerConfigChange).toHaveBeenCalledWith({
        base: { visible: true, opacity: 1.0 },
        pigmentation: { visible: true, opacity: 0.7 },
        wrinkles: { visible: true, opacity: 0.7 }
      });
    });

    it('should show only base layer when "Base Only" button is clicked', () => {
      const layerConfig: LayerConfig = {
        base: { visible: true, opacity: 1.0 },
        pigmentation: { visible: true, opacity: 0.7 },
        wrinkles: { visible: true, opacity: 0.7 }
      };
      const onLayerConfigChange = jest.fn();
      
      render(
        <LayerControls
          layerConfig={layerConfig}
          onLayerConfigChange={onLayerConfigChange}
        />
      );
      
      const baseOnlyButton = screen.getByText('Base Only');
      fireEvent.click(baseOnlyButton);
      
      expect(onLayerConfigChange).toHaveBeenCalledWith({
        base: { visible: true, opacity: 1.0 },
        pigmentation: { visible: false, opacity: 0.7 },
        wrinkles: { visible: false, opacity: 0.7 }
      });
    });

    it('should display correct opacity percentages', () => {
      const layerConfig: LayerConfig = {
        base: { visible: true, opacity: 0.85 },
        pigmentation: { visible: true, opacity: 0.45 },
        wrinkles: { visible: true, opacity: 0.25 }
      };
      const onLayerConfigChange = jest.fn();
      
      render(
        <LayerControls
          layerConfig={layerConfig}
          onLayerConfigChange={onLayerConfigChange}
        />
      );
      
      expect(screen.getByText('Opacity: 85%')).toBeInTheDocument();
      expect(screen.getByText('Opacity: 45%')).toBeInTheDocument();
      expect(screen.getByText('Opacity: 25%')).toBeInTheDocument();
    });

    it('should disable opacity slider when layer is not visible', () => {
      const layerConfig: LayerConfig = {
        base: { visible: false, opacity: 1.0 },
        pigmentation: { visible: true, opacity: 0.7 },
        wrinkles: { visible: true, opacity: 0.7 }
      };
      const onLayerConfigChange = jest.fn();
      
      render(
        <LayerControls
          layerConfig={layerConfig}
          onLayerConfigChange={onLayerConfigChange}
        />
      );
      
      const sliders = screen.getAllByRole('slider');
      const baseSlider = sliders[0];
      
      expect(baseSlider).toBeDisabled();
    });
  });

  describe('Layer Blending', () => {
    it('should support multiple layers visible simultaneously', () => {
      const mesh = createMockMesh();
      const layerConfig: LayerConfig = {
        base: { visible: true, opacity: 1.0 },
        pigmentation: { visible: true, opacity: 0.7 },
        wrinkles: { visible: true, opacity: 0.7 }
      };
      
      const { container } = render(
        <Viewer3D mesh={mesh} layerConfig={layerConfig} />
      );
      
      expect(container).toBeInTheDocument();
    });

    it('should support transparency for overlay layers', () => {
      const mesh = createMockMesh();
      const layerConfig: LayerConfig = {
        base: { visible: true, opacity: 1.0 },
        pigmentation: { visible: true, opacity: 0.5 },
        wrinkles: { visible: true, opacity: 0.3 }
      };
      
      const { container } = render(
        <Viewer3D mesh={mesh} layerConfig={layerConfig} />
      );
      
      expect(container).toBeInTheDocument();
    });

    it('should handle all layers hidden', () => {
      const mesh = createMockMesh();
      const layerConfig: LayerConfig = {
        base: { visible: false, opacity: 1.0 },
        pigmentation: { visible: false, opacity: 0.7 },
        wrinkles: { visible: false, opacity: 0.7 }
      };
      
      const { container } = render(
        <Viewer3D mesh={mesh} layerConfig={layerConfig} />
      );
      
      expect(container).toBeInTheDocument();
    });
  });

  describe('Requirements Validation', () => {
    it('should satisfy Requirement 4.6: Users can toggle visibility of individual anomaly layers', () => {
      const layerConfig: LayerConfig = {
        base: { visible: true, opacity: 1.0 },
        pigmentation: { visible: true, opacity: 0.7 },
        wrinkles: { visible: true, opacity: 0.7 }
      };
      const onLayerConfigChange = jest.fn();
      
      render(
        <LayerControls
          layerConfig={layerConfig}
          onLayerConfigChange={onLayerConfigChange}
        />
      );
      
      // Toggle pigmentation layer
      const pigmentationCheckbox = screen.getByRole('checkbox', { name: /Pigmentation/i });
      fireEvent.click(pigmentationCheckbox);
      
      expect(onLayerConfigChange).toHaveBeenCalledWith(
        expect.objectContaining({
          pigmentation: { visible: false, opacity: 0.7 }
        })
      );
      
      // Toggle wrinkles layer
      const wrinklesCheckbox = screen.getByRole('checkbox', { name: /Wrinkles/i });
      fireEvent.click(wrinklesCheckbox);
      
      expect(onLayerConfigChange).toHaveBeenCalledWith(
        expect.objectContaining({
          wrinkles: { visible: false, opacity: 0.7 }
        })
      );
    });

    it('should satisfy Requirement 4.7: Users can adjust transparency of overlays', () => {
      const layerConfig: LayerConfig = {
        base: { visible: true, opacity: 1.0 },
        pigmentation: { visible: true, opacity: 0.7 },
        wrinkles: { visible: true, opacity: 0.7 }
      };
      const onLayerConfigChange = jest.fn();
      
      render(
        <LayerControls
          layerConfig={layerConfig}
          onLayerConfigChange={onLayerConfigChange}
        />
      );
      
      const sliders = screen.getAllByRole('slider');
      
      // Adjust pigmentation transparency
      fireEvent.change(sliders[1], { target: { value: '40' } });
      expect(onLayerConfigChange).toHaveBeenCalledWith(
        expect.objectContaining({
          pigmentation: { visible: true, opacity: 0.4 }
        })
      );
      
      // Adjust wrinkles transparency
      fireEvent.change(sliders[2], { target: { value: '60' } });
      expect(onLayerConfigChange).toHaveBeenCalledWith(
        expect.objectContaining({
          wrinkles: { visible: true, opacity: 0.6 }
        })
      );
    });
  });
});
