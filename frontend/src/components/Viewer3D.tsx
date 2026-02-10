import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { Mesh } from '../types';

export type FacialRegion = 
  | 'all' 
  | 'forehead' 
  | 'left_cheek' 
  | 'right_cheek' 
  | 'periorbital_left' 
  | 'periorbital_right'
  | 'nose'
  | 'mouth'
  | 'chin';

export interface LayerConfig {
  base: { visible: boolean; opacity: number };
  pigmentation: { visible: boolean; opacity: number };
  wrinkles: { visible: boolean; opacity: number };
}

export interface RegionConfig {
  selectedRegion: FacialRegion;
  highlightIntensity: number;
}

export interface SeverityFilterConfig {
  pigmentation: {
    low: boolean;
    medium: boolean;
    high: boolean;
  };
  wrinkles: {
    micro: boolean;
    regular: boolean;
  };
  allCombined: boolean;
}

export interface Viewer3DHandle {
  exportSnapshot: (format: 'png' | 'jpeg', filename?: string) => void;
  zoomToRegion: (region: FacialRegion) => void;
}

interface Viewer3DProps {
  mesh?: Mesh;
  width?: number;
  height?: number;
  showWireframe?: boolean;
  enableControls?: boolean;
  onReady?: (scene: THREE.Scene, camera: THREE.Camera, meshObjects: THREE.Object3D[]) => void;
  layerConfig?: LayerConfig;
  onLayerConfigChange?: (config: LayerConfig) => void;
  regionConfig?: RegionConfig;
  onRegionConfigChange?: (config: RegionConfig) => void;
  severityFilterConfig?: SeverityFilterConfig;
  onSeverityFilterChange?: (config: SeverityFilterConfig) => void;
}

/**
 * Interactive 3D viewer component using Three.js
 * Provides real-time rendering of facial meshes with orbit controls
 * 
 * Features:
 * - WebGL rendering with anti-aliasing
 * - Orbit controls (rotation, zoom, pan)
 * - Configurable lighting
 * - Smooth shading with normals
 * - Texture mapping support
 * - Layered visualization with separate materials for base, pigmentation, and wrinkles
 * - Layer visibility toggles and transparency controls
 * - Snapshot export to PNG/JPEG
 * 
 * Requirements: 3.1, 3.4, 3.8, 4.6, 4.7
 */
const Viewer3D = React.forwardRef<Viewer3DHandle, Viewer3DProps>(({
  mesh,
  width = 800,
  height = 600,
  showWireframe = false,
  enableControls = true,
  onReady,
  layerConfig,
  onLayerConfigChange,
  regionConfig,
  severityFilterConfig
}, ref) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const meshObjectRef = useRef<THREE.Mesh | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  
  // Refs for layered meshes
  const baseMeshRef = useRef<THREE.Mesh | null>(null);
  const pigmentationMeshRef = useRef<THREE.Mesh | null>(null);
  const wrinkleMeshRef = useRef<THREE.Mesh | null>(null);
  
  const [isInitialized, setIsInitialized] = useState(false);
  const [internalLayerConfig, setInternalLayerConfig] = useState<LayerConfig>({
    base: { visible: true, opacity: 1.0 },
    pigmentation: { visible: true, opacity: 0.7 },
    wrinkles: { visible: true, opacity: 0.7 }
  });
  const [internalRegionConfig, setInternalRegionConfig] = useState<RegionConfig>({
    selectedRegion: 'all',
    highlightIntensity: 0.5
  });
  const [internalSeverityFilterConfig, setInternalSeverityFilterConfig] = useState<SeverityFilterConfig>({
    pigmentation: { low: true, medium: true, high: true },
    wrinkles: { micro: true, regular: true },
    allCombined: true
  });

  /**
   * Export current view as PNG or JPEG
   * Captures the current renderer output including all visible layers and annotations
   * 
   * Requirements: 3.8
   */
  const exportSnapshot = React.useCallback((format: 'png' | 'jpeg', filename?: string) => {
    if (!rendererRef.current || !sceneRef.current || !cameraRef.current) {
      console.error('Viewer not initialized');
      return;
    }

    try {
      // Render one more frame to ensure we capture the current state
      rendererRef.current.render(sceneRef.current, cameraRef.current);

      // Get the canvas element
      const canvas = rendererRef.current.domElement;

      // Determine MIME type and file extension
      const mimeType = format === 'png' ? 'image/png' : 'image/jpeg';
      const extension = format === 'png' ? 'png' : 'jpg';
      const defaultFilename = `snapshot_${new Date().getTime()}.${extension}`;

      // Convert canvas to data URL
      const dataURL = canvas.toDataURL(mimeType, 0.95); // 0.95 quality for JPEG

      // Create a temporary link element and trigger download
      const link = document.createElement('a');
      link.href = dataURL;
      link.download = filename || defaultFilename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      console.log(`Snapshot exported as ${link.download}`);
    } catch (error) {
      console.error('Failed to export snapshot:', error);
      throw error;
    }
  }, []);

  /**
   * Zoom camera to focus on a specific facial region
   * Animates the camera to center and zoom on the selected region
   * 
   * Requirements: 3.6
   */
  const zoomToRegion = React.useCallback((region: FacialRegion) => {
    if (!cameraRef.current || !controlsRef.current || !mesh) {
      console.error('Viewer not initialized or no mesh loaded');
      return;
    }

    // Define region bounding boxes based on typical facial proportions
    // These are approximate regions in normalized coordinates (-1 to 1)
    const regionBounds: Record<FacialRegion, { center: [number, number, number]; size: number }> = {
      all: { center: [0, 0, 0], size: 100 },
      forehead: { center: [0, 30, 0], size: 40 },
      left_cheek: { center: [-25, -10, 0], size: 30 },
      right_cheek: { center: [25, -10, 0], size: 30 },
      periorbital_left: { center: [-20, 15, 0], size: 25 },
      periorbital_right: { center: [20, 15, 0], size: 25 },
      nose: { center: [0, 0, 10], size: 25 },
      mouth: { center: [0, -25, 0], size: 30 },
      chin: { center: [0, -45, 0], size: 25 }
    };

    const bounds = regionBounds[region];
    if (!bounds) return;

    // Calculate target camera position
    const targetCenter = new THREE.Vector3(...bounds.center);
    const distance = bounds.size * 2; // Distance from region
    const targetPosition = new THREE.Vector3(
      targetCenter.x,
      targetCenter.y,
      targetCenter.z + distance
    );

    // Animate camera to target position
    const startPosition = cameraRef.current.position.clone();
    const startTarget = controlsRef.current.target.clone();
    const duration = 1000; // 1 second animation
    const startTime = Date.now();

    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      
      // Easing function (ease-in-out)
      const eased = progress < 0.5
        ? 2 * progress * progress
        : 1 - Math.pow(-2 * progress + 2, 2) / 2;

      // Interpolate camera position
      cameraRef.current!.position.lerpVectors(startPosition, targetPosition, eased);
      
      // Interpolate controls target
      controlsRef.current!.target.lerpVectors(startTarget, targetCenter, eased);
      controlsRef.current!.update();

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };

    animate();
  }, [mesh]);

  // Expose methods via ref
  React.useImperativeHandle(ref, () => ({
    exportSnapshot,
    zoomToRegion
  }), [exportSnapshot, zoomToRegion]);

  // Initialize Three.js scene
  useEffect(() => {
    if (!containerRef.current || isInitialized) return;

    // Create WebGL renderer with anti-aliasing
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance'
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Create scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);
    sceneRef.current = scene;

    // Create camera
    const camera = new THREE.PerspectiveCamera(
      45, // Field of view
      width / height, // Aspect ratio
      0.1, // Near clipping plane
      1000 // Far clipping plane
    );
    camera.position.set(0, 0, 300);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    // Set up lighting
    setupLighting(scene);

    // Set up orbit controls
    if (enableControls) {
      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;
      controls.screenSpacePanning = false;
      controls.minDistance = 50;
      controls.maxDistance = 500;
      controls.maxPolarAngle = Math.PI;
      controlsRef.current = controls;
    }

    // Add grid helper for reference
    const gridHelper = new THREE.GridHelper(200, 20, 0x888888, 0xcccccc);
    gridHelper.position.y = -100;
    scene.add(gridHelper);

    // Add axes helper for debugging
    const axesHelper = new THREE.AxesHelper(100);
    scene.add(axesHelper);

    setIsInitialized(true);
    
    if (onReady) {
      // Call onReady with scene, camera, and empty mesh array (meshes will be added later)
      onReady(scene, camera, []);
    }

    // Start animation loop
    const animate = () => {
      animationFrameRef.current = requestAnimationFrame(animate);
      
      if (controlsRef.current) {
        controlsRef.current.update();
      }
      
      if (rendererRef.current && sceneRef.current && cameraRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current);
      }
    };
    animate();

    // Cleanup
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (controlsRef.current) {
        controlsRef.current.dispose();
      }
      if (rendererRef.current) {
        rendererRef.current.dispose();
      }
      if (containerRef.current && rendererRef.current) {
        containerRef.current.removeChild(rendererRef.current.domElement);
      }
    };
  }, [width, height, enableControls, isInitialized, onReady]);

  // Update mesh when data changes
  useEffect(() => {
    if (!mesh || !sceneRef.current || !isInitialized) return;

    // Remove existing meshes
    if (baseMeshRef.current) {
      sceneRef.current.remove(baseMeshRef.current);
      disposeMesh(baseMeshRef.current);
      baseMeshRef.current = null;
    }
    if (pigmentationMeshRef.current) {
      sceneRef.current.remove(pigmentationMeshRef.current);
      disposeMesh(pigmentationMeshRef.current);
      pigmentationMeshRef.current = null;
    }
    if (wrinkleMeshRef.current) {
      sceneRef.current.remove(wrinkleMeshRef.current);
      disposeMesh(wrinkleMeshRef.current);
      wrinkleMeshRef.current = null;
    }
    if (meshObjectRef.current) {
      sceneRef.current.remove(meshObjectRef.current);
      disposeMesh(meshObjectRef.current);
      meshObjectRef.current = null;
    }

    // Create layered meshes if vertex labels are available
    if (mesh.vertexLabels && mesh.vertexLabels.length > 0) {
      const filterConfig = severityFilterConfig || internalSeverityFilterConfig;
      const layers = createLayeredMeshes(mesh, showWireframe, layerConfig || internalLayerConfig, filterConfig);
      
      const meshObjects: THREE.Object3D[] = [];
      
      if (layers.base) {
        sceneRef.current.add(layers.base);
        baseMeshRef.current = layers.base;
        centerAndScaleMesh(layers.base);
        meshObjects.push(layers.base);
      }
      
      if (layers.pigmentation) {
        sceneRef.current.add(layers.pigmentation);
        pigmentationMeshRef.current = layers.pigmentation;
        // Match position and scale of base mesh
        if (baseMeshRef.current) {
          pigmentationMeshRef.current.position.copy(baseMeshRef.current.position);
          pigmentationMeshRef.current.scale.copy(baseMeshRef.current.scale);
        }
        meshObjects.push(layers.pigmentation);
      }
      
      if (layers.wrinkles) {
        sceneRef.current.add(layers.wrinkles);
        wrinkleMeshRef.current = layers.wrinkles;
        // Match position and scale of base mesh
        if (baseMeshRef.current) {
          wrinkleMeshRef.current.position.copy(baseMeshRef.current.position);
          wrinkleMeshRef.current.scale.copy(baseMeshRef.current.scale);
        }
        meshObjects.push(layers.wrinkles);
      }
      
      // Notify parent with updated mesh objects
      if (onReady && sceneRef.current && cameraRef.current) {
        onReady(sceneRef.current, cameraRef.current, meshObjects);
      }
    } else {
      // Create single mesh without layers
      const threeMesh = createThreeMesh(mesh, showWireframe);
      sceneRef.current.add(threeMesh);
      meshObjectRef.current = threeMesh;
      centerAndScaleMesh(threeMesh);
      
      // Notify parent with mesh object
      if (onReady && sceneRef.current && cameraRef.current) {
        onReady(sceneRef.current, cameraRef.current, [threeMesh]);
      }
    }

  }, [mesh, showWireframe, isInitialized, layerConfig, internalLayerConfig, onReady]);

  // Update meshes when severity filter changes
  useEffect(() => {
    if (!mesh || !sceneRef.current || !isInitialized) return;
    if (!mesh.vertexLabels || mesh.vertexLabels.length === 0) return;

    const filterConfig = severityFilterConfig || internalSeverityFilterConfig;
    
    // Update pigmentation mesh vertex colors based on filter
    if (pigmentationMeshRef.current) {
      updatePigmentationColors(pigmentationMeshRef.current, mesh, filterConfig);
    }
    
    // Update wrinkle mesh vertex colors based on filter
    if (wrinkleMeshRef.current) {
      updateWrinkleColors(wrinkleMeshRef.current, mesh, filterConfig);
    }
  }, [severityFilterConfig, internalSeverityFilterConfig, mesh, isInitialized]);

  // Update layer visibility and transparency
  useEffect(() => {
    const config = layerConfig || internalLayerConfig;
    
    if (baseMeshRef.current) {
      baseMeshRef.current.visible = config.base.visible;
      if (baseMeshRef.current.material instanceof THREE.Material) {
        baseMeshRef.current.material.opacity = config.base.opacity;
        baseMeshRef.current.material.transparent = config.base.opacity < 1.0;
      }
    }
    
    if (pigmentationMeshRef.current) {
      pigmentationMeshRef.current.visible = config.pigmentation.visible;
      if (pigmentationMeshRef.current.material instanceof THREE.Material) {
        pigmentationMeshRef.current.material.opacity = config.pigmentation.opacity;
        pigmentationMeshRef.current.material.transparent = config.pigmentation.opacity < 1.0;
      }
    }
    
    if (wrinkleMeshRef.current) {
      wrinkleMeshRef.current.visible = config.wrinkles.visible;
      if (wrinkleMeshRef.current.material instanceof THREE.Material) {
        wrinkleMeshRef.current.material.opacity = config.wrinkles.opacity;
        wrinkleMeshRef.current.material.transparent = config.wrinkles.opacity < 1.0;
      }
    }
  }, [layerConfig, internalLayerConfig]);

  // Update region highlighting
  useEffect(() => {
    const config = regionConfig || internalRegionConfig;
    
    if (!mesh || !mesh.vertices || config.selectedRegion === 'all') {
      // Reset all meshes to normal brightness
      [baseMeshRef.current, pigmentationMeshRef.current, wrinkleMeshRef.current].forEach(meshRef => {
        if (meshRef && meshRef.material instanceof THREE.Material) {
          meshRef.material.emissive = new THREE.Color(0x000000);
          meshRef.material.emissiveIntensity = 0.0;
        }
      });
      return;
    }

    // Apply region highlighting based on vertex positions
    // This is a simplified approach - in production, you'd use actual landmark-based regions
    const highlightRegion = (meshRef: THREE.Mesh | null) => {
      if (!meshRef || !mesh.vertices) return;
      
      const geometry = meshRef.geometry;
      const positions = geometry.attributes.position;
      
      // Calculate which vertices belong to the selected region
      const vertexInRegion = new Array(mesh.vertices.length).fill(false);
      
      for (let i = 0; i < mesh.vertices.length; i++) {
        const vertex = mesh.vertices[i];
        const [x, y, z] = vertex;
        
        // Simple region detection based on vertex position
        // These thresholds are approximate and would need calibration with real data
        let inRegion = false;
        
        switch (config.selectedRegion) {
          case 'forehead':
            inRegion = y > 20;
            break;
          case 'left_cheek':
            inRegion = x < -10 && y > -30 && y < 20;
            break;
          case 'right_cheek':
            inRegion = x > 10 && y > -30 && y < 20;
            break;
          case 'periorbital_left':
            inRegion = x < -10 && y > 5 && y < 25;
            break;
          case 'periorbital_right':
            inRegion = x > 10 && y > 5 && y < 25;
            break;
          case 'nose':
            inRegion = Math.abs(x) < 15 && y > -10 && y < 15 && z > 5;
            break;
          case 'mouth':
            inRegion = Math.abs(x) < 20 && y > -35 && y < -15;
            break;
          case 'chin':
            inRegion = Math.abs(x) < 20 && y < -35;
            break;
        }
        
        vertexInRegion[i] = inRegion;
      }
      
      // Apply highlighting by adjusting material properties
      if (meshRef.material instanceof THREE.Material) {
        // For vertices in the region, increase brightness
        // For vertices outside, dim them
        const dimFactor = 1.0 - (config.highlightIntensity * 0.7);
        
        // Create a simple dimming effect by adjusting emissive color
        // Vertices in region stay bright, others are dimmed
        const avgInRegion = vertexInRegion.reduce((sum, val) => sum + (val ? 1 : 0), 0) / vertexInRegion.length;
        
        if (avgInRegion > 0.01) {
          // Some vertices are in the region, apply dimming to the whole mesh
          // and let the region vertices stay bright
          meshRef.material.emissive = new THREE.Color(0x000000);
          meshRef.material.emissiveIntensity = 0.0;
          
          // Adjust opacity to create dimming effect for non-region areas
          const originalOpacity = meshRef.material.opacity;
          meshRef.material.opacity = originalOpacity * (avgInRegion > 0.5 ? 1.0 : dimFactor);
        }
      }
    };
    
    // Apply highlighting to all mesh layers
    highlightRegion(baseMeshRef.current);
    highlightRegion(pigmentationMeshRef.current);
    highlightRegion(wrinkleMeshRef.current);
    
  }, [regionConfig, internalRegionConfig, mesh]);

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      if (!rendererRef.current || !cameraRef.current) return;
      
      const newWidth = containerRef.current?.clientWidth || width;
      const newHeight = containerRef.current?.clientHeight || height;
      
      cameraRef.current.aspect = newWidth / newHeight;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(newWidth, newHeight);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [width, height]);

  return (
    <div 
      ref={containerRef} 
      style={{ 
        width: '100%', 
        height: '100%',
        minHeight: `${height}px`,
        position: 'relative'
      }}
    />
  );
});

Viewer3D.displayName = 'Viewer3D';

/**
 * Set up scene lighting for optimal facial visualization
 * Uses a combination of ambient, directional, and hemisphere lights
 */
function setupLighting(scene: THREE.Scene): void {
  // Ambient light for base illumination
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
  scene.add(ambientLight);

  // Main directional light (key light)
  const keyLight = new THREE.DirectionalLight(0xffffff, 0.8);
  keyLight.position.set(100, 100, 100);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.width = 2048;
  keyLight.shadow.mapSize.height = 2048;
  keyLight.shadow.camera.near = 0.5;
  keyLight.shadow.camera.far = 500;
  scene.add(keyLight);

  // Fill light (softer, from opposite side)
  const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
  fillLight.position.set(-100, 50, -50);
  scene.add(fillLight);

  // Back light (rim light for depth)
  const backLight = new THREE.DirectionalLight(0xffffff, 0.2);
  backLight.position.set(0, 100, -100);
  scene.add(backLight);

  // Hemisphere light for natural sky/ground lighting
  const hemisphereLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.3);
  hemisphereLight.position.set(0, 200, 0);
  scene.add(hemisphereLight);
}

/**
 * Create Three.js mesh from application mesh data
 * 
 * Handles:
 * - Vertex positions, normals, UV coordinates
 * - Vertex colors for texture representation
 * - Smooth shading with computed or provided normals
 * - Texture maps (base64 or URL)
 * 
 * Requirements: 3.1, 3.3
 */
function createThreeMesh(meshData: Mesh, wireframe: boolean): THREE.Mesh {
  // Create geometry
  const geometry = new THREE.BufferGeometry();

  // Convert vertices to Float32Array
  const vertices = new Float32Array(meshData.vertices.flat());
  geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

  // Set faces (indices)
  const indices = new Uint32Array(meshData.faces.flat());
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));

  // Set normals if available, otherwise compute them
  if (meshData.normals && meshData.normals.length > 0) {
    const normals = new Float32Array(meshData.normals.flat());
    geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  } else {
    // Compute smooth normals for better shading
    geometry.computeVertexNormals();
  }

  // Set UV coordinates if available
  if (meshData.uvCoordinates && meshData.uvCoordinates.length > 0) {
    const uvs = new Float32Array(meshData.uvCoordinates.flat());
    geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  }

  // Set vertex colors if available
  let hasVertexColors = false;
  if (meshData.vertexColors && meshData.vertexColors.length > 0) {
    const colors = new Float32Array(meshData.vertexColors.flat());
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    hasVertexColors = true;
  }

  // Create material with smooth shading
  const material = new THREE.MeshPhongMaterial({
    color: 0xffffff,
    specular: 0x222222,
    shininess: 25,
    vertexColors: hasVertexColors,
    wireframe: wireframe,
    side: THREE.DoubleSide,
    flatShading: false, // Enable smooth shading
    emissive: 0x000000,
    emissiveIntensity: 0.0
  });

  // Load texture if available (supports base64 and URLs)
  if (meshData.textureMap && meshData.textureMap.length > 0 && !wireframe) {
    const textureLoader = new THREE.TextureLoader();
    
    // Handle base64 encoded textures
    if (meshData.textureMap.startsWith('data:')) {
      const texture = textureLoader.load(meshData.textureMap);
      texture.colorSpace = THREE.SRGBColorSpace;
      material.map = texture;
    } else if (meshData.textureMap.startsWith('http')) {
      // Handle URL textures
      const texture = textureLoader.load(meshData.textureMap);
      texture.colorSpace = THREE.SRGBColorSpace;
      material.map = texture;
    }
  }

  // Create mesh
  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = true;
  mesh.receiveShadow = true;

  return mesh;
}

/**
 * Center and scale mesh to fit in view
 */
function centerAndScaleMesh(mesh: THREE.Mesh): void {
  // Compute bounding box
  mesh.geometry.computeBoundingBox();
  const boundingBox = mesh.geometry.boundingBox;
  
  if (!boundingBox) return;

  // Calculate center
  const center = new THREE.Vector3();
  boundingBox.getCenter(center);

  // Center the mesh
  mesh.position.sub(center);

  // Calculate size
  const size = new THREE.Vector3();
  boundingBox.getSize(size);
  const maxDim = Math.max(size.x, size.y, size.z);

  // Scale to fit in view (target size ~100 units)
  const targetSize = 100;
  const scale = targetSize / maxDim;
  mesh.scale.set(scale, scale, scale);
}

/**
 * Dispose of a mesh and its resources
 */
function disposeMesh(mesh: THREE.Mesh): void {
  mesh.geometry.dispose();
  if (Array.isArray(mesh.material)) {
    mesh.material.forEach(mat => mat.dispose());
  } else {
    mesh.material.dispose();
  }
}

/**
 * Create layered meshes for base, pigmentation, and wrinkles
 * 
 * Vertex labels encoding:
 * - 0: Base/normal skin
 * - 1-3: Pigmentation (1=Low, 2=Medium, 3=High)
 * - 4-5: Wrinkles (4=micro, 5=regular)
 * 
 * Requirements: 4.6, 4.7, 5.2, 5.3, 5.4
 */
function createLayeredMeshes(
  meshData: Mesh, 
  wireframe: boolean,
  layerConfig: LayerConfig,
  filterConfig: SeverityFilterConfig
): { base: THREE.Mesh | null; pigmentation: THREE.Mesh | null; wrinkles: THREE.Mesh | null } {
  
  // Create base geometry (all vertices)
  const baseGeometry = createGeometry(meshData);
  const baseMaterial = new THREE.MeshPhongMaterial({
    color: 0xffd7b5, // Skin tone
    specular: 0x222222,
    shininess: 25,
    wireframe: wireframe,
    side: THREE.DoubleSide,
    flatShading: false,
    transparent: layerConfig.base.opacity < 1.0,
    opacity: layerConfig.base.opacity
  });
  const baseMesh = new THREE.Mesh(baseGeometry, baseMaterial);
  baseMesh.castShadow = true;
  baseMesh.receiveShadow = true;
  baseMesh.visible = layerConfig.base.visible;
  
  // Create pigmentation overlay (only pigmentation vertices)
  let pigmentationMesh: THREE.Mesh | null = null;
  const pigmentationVertices = meshData.vertexLabels
    .map((label, idx) => (label >= 1 && label <= 3) ? idx : -1)
    .filter(idx => idx !== -1);
  
  if (pigmentationVertices.length > 0) {
    const pigmentationGeometry = createGeometry(meshData);
    
    // Create vertex colors for pigmentation severity with filtering
    const colors = new Float32Array(meshData.vertices.length * 3);
    for (let i = 0; i < meshData.vertexLabels.length; i++) {
      const label = meshData.vertexLabels[i];
      let color = new THREE.Color();
      let visible = false;
      
      // Check if this severity level is enabled in the filter
      if (label === 1 && (filterConfig.allCombined || filterConfig.pigmentation.low)) {
        color.setHex(0xFFE5B4); // Light yellow for Low
        visible = true;
      } else if (label === 2 && (filterConfig.allCombined || filterConfig.pigmentation.medium)) {
        color.setHex(0xFFA500); // Orange for Medium
        visible = true;
      } else if (label === 3 && (filterConfig.allCombined || filterConfig.pigmentation.high)) {
        color.setHex(0x8B0000); // Dark red for High
        visible = true;
      } else {
        color.setHex(0x000000); // Black for non-pigmentation or filtered out (will be transparent)
      }
      
      // If not visible, set alpha to 0 by using black color
      // The material will handle transparency
      if (!visible) {
        color.setHex(0x000000);
      }
      
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }
    pigmentationGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    
    const pigmentationMaterial = new THREE.MeshPhongMaterial({
      vertexColors: true,
      specular: 0x111111,
      shininess: 15,
      wireframe: wireframe,
      side: THREE.DoubleSide,
      flatShading: false,
      transparent: true,
      opacity: layerConfig.pigmentation.opacity,
      depthWrite: false // Allow blending with base layer
    });
    
    pigmentationMesh = new THREE.Mesh(pigmentationGeometry, pigmentationMaterial);
    pigmentationMesh.castShadow = false;
    pigmentationMesh.receiveShadow = false;
    pigmentationMesh.visible = layerConfig.pigmentation.visible;
    // Offset slightly to prevent z-fighting
    pigmentationMesh.position.z = 0.01;
  }
  
  // Create wrinkle overlay (only wrinkle vertices)
  let wrinkleMesh: THREE.Mesh | null = null;
  const wrinkleVertices = meshData.vertexLabels
    .map((label, idx) => (label >= 4 && label <= 5) ? idx : -1)
    .filter(idx => idx !== -1);
  
  if (wrinkleVertices.length > 0) {
    const wrinkleGeometry = createGeometry(meshData);
    
    // Create vertex colors for wrinkle types with filtering
    const colors = new Float32Array(meshData.vertices.length * 3);
    for (let i = 0; i < meshData.vertexLabels.length; i++) {
      const label = meshData.vertexLabels[i];
      let color = new THREE.Color();
      let visible = false;
      
      // Check if this wrinkle type is enabled in the filter
      if (label === 4 && (filterConfig.allCombined || filterConfig.wrinkles.micro)) {
        color.setHex(0xADD8E6); // Light blue for micro-wrinkles
        visible = true;
      } else if (label === 5 && (filterConfig.allCombined || filterConfig.wrinkles.regular)) {
        color.setHex(0x00008B); // Dark blue for regular wrinkles
        visible = true;
      } else {
        color.setHex(0x000000); // Black for non-wrinkle or filtered out (will be transparent)
      }
      
      // If not visible, set to black
      if (!visible) {
        color.setHex(0x000000);
      }
      
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }
    wrinkleGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    
    const wrinkleMaterial = new THREE.MeshPhongMaterial({
      vertexColors: true,
      specular: 0x111111,
      shininess: 15,
      wireframe: wireframe,
      side: THREE.DoubleSide,
      flatShading: false,
      transparent: true,
      opacity: layerConfig.wrinkles.opacity,
      depthWrite: false // Allow blending with base layer
    });
    
    wrinkleMesh = new THREE.Mesh(wrinkleGeometry, wrinkleMaterial);
    wrinkleMesh.castShadow = false;
    wrinkleMesh.receiveShadow = false;
    wrinkleMesh.visible = layerConfig.wrinkles.visible;
    // Offset slightly to prevent z-fighting
    wrinkleMesh.position.z = 0.02;
  }
  
  return {
    base: baseMesh,
    pigmentation: pigmentationMesh,
    wrinkles: wrinkleMesh
  };
}

/**
 * Create Three.js geometry from mesh data
 */
function createGeometry(meshData: Mesh): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();

  // Convert vertices to Float32Array
  const vertices = new Float32Array(meshData.vertices.flat());
  geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

  // Set faces (indices)
  const indices = new Uint32Array(meshData.faces.flat());
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));

  // Set normals if available, otherwise compute them
  if (meshData.normals && meshData.normals.length > 0) {
    const normals = new Float32Array(meshData.normals.flat());
    geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  } else {
    geometry.computeVertexNormals();
  }

  // Set UV coordinates if available
  if (meshData.uvCoordinates && meshData.uvCoordinates.length > 0) {
    const uvs = new Float32Array(meshData.uvCoordinates.flat());
    geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  }

  return geometry;
}

/**
 * Update pigmentation mesh vertex colors based on filter configuration
 * Enables real-time filtering without recreating the entire mesh
 * 
 * Requirements: 5.2, 5.3, 5.4, 5.6
 */
function updatePigmentationColors(
  mesh: THREE.Mesh,
  meshData: Mesh,
  filterConfig: SeverityFilterConfig
): void {
  const geometry = mesh.geometry;
  const colorAttribute = geometry.attributes.color;
  
  if (!colorAttribute || !meshData.vertexLabels) return;
  
  const colors = colorAttribute.array as Float32Array;
  
  for (let i = 0; i < meshData.vertexLabels.length; i++) {
    const label = meshData.vertexLabels[i];
    let color = new THREE.Color();
    let visible = false;
    
    // Check if this severity level is enabled in the filter
    if (label === 1 && (filterConfig.allCombined || filterConfig.pigmentation.low)) {
      color.setHex(0xFFE5B4); // Light yellow for Low
      visible = true;
    } else if (label === 2 && (filterConfig.allCombined || filterConfig.pigmentation.medium)) {
      color.setHex(0xFFA500); // Orange for Medium
      visible = true;
    } else if (label === 3 && (filterConfig.allCombined || filterConfig.pigmentation.high)) {
      color.setHex(0x8B0000); // Dark red for High
      visible = true;
    }
    
    // If not visible, set to black (transparent)
    if (!visible) {
      color.setHex(0x000000);
    }
    
    colors[i * 3] = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;
  }
  
  colorAttribute.needsUpdate = true;
}

/**
 * Update wrinkle mesh vertex colors based on filter configuration
 * Enables real-time filtering without recreating the entire mesh
 * 
 * Requirements: 5.2, 5.3, 5.4, 5.6
 */
function updateWrinkleColors(
  mesh: THREE.Mesh,
  meshData: Mesh,
  filterConfig: SeverityFilterConfig
): void {
  const geometry = mesh.geometry;
  const colorAttribute = geometry.attributes.color;
  
  if (!colorAttribute || !meshData.vertexLabels) return;
  
  const colors = colorAttribute.array as Float32Array;
  
  for (let i = 0; i < meshData.vertexLabels.length; i++) {
    const label = meshData.vertexLabels[i];
    let color = new THREE.Color();
    let visible = false;
    
    // Check if this wrinkle type is enabled in the filter
    if (label === 4 && (filterConfig.allCombined || filterConfig.wrinkles.micro)) {
      color.setHex(0xADD8E6); // Light blue for micro-wrinkles
      visible = true;
    } else if (label === 5 && (filterConfig.allCombined || filterConfig.wrinkles.regular)) {
      color.setHex(0x00008B); // Dark blue for regular wrinkles
      visible = true;
    }
    
    // If not visible, set to black (transparent)
    if (!visible) {
      color.setHex(0x000000);
    }
    
    colors[i * 3] = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;
  }
  
  colorAttribute.needsUpdate = true;
}

export type { LayerConfig, RegionConfig, FacialRegion, Viewer3DHandle, SeverityFilterConfig };
export default Viewer3D;
