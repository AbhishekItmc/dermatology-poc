import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as THREE from 'three';
import Viewer3D, { LayerConfig } from './Viewer3D';
import MeasurementTools from './MeasurementTools';
import { useMeasurements } from '../hooks/useMeasurements';
import { Mesh } from '../types';
import { MeasurementPoint } from './MeasurementTools';

interface Viewer3DWithMeasurementsProps {
  mesh?: Mesh;
  width?: number;
  height?: number;
  showWireframe?: boolean;
  enableControls?: boolean;
  layerConfig?: LayerConfig;
  onLayerConfigChange?: (config: LayerConfig) => void;
  pixelToMmScale?: number; // Conversion factor from mesh units to mm
}

/**
 * Enhanced 3D viewer with integrated measurement tools
 * 
 * Features:
 * - Click-to-measure distance tool
 * - Area selection and measurement tool
 * - Display measurements in mm
 * - Visual indicators for measurement points and lines
 * 
 * Requirements: 3.7
 */
const Viewer3DWithMeasurements: React.FC<Viewer3DWithMeasurementsProps> = ({
  mesh,
  width = 800,
  height = 600,
  showWireframe = false,
  enableControls = true,
  layerConfig,
  onLayerConfigChange,
  pixelToMmScale = 1.0
}) => {
  const viewerContainerRef = useRef<HTMLDivElement>(null);
  const [viewerReady, setViewerReady] = useState(false);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.Camera | null>(null);
  const meshObjectsRef = useRef<THREE.Object3D[]>([]);
  
  const {
    mode,
    setMode,
    distanceMeasurements,
    areaMeasurements,
    tempPoints,
    handleMeasurementClick,
    clearMeasurements
  } = useMeasurements(pixelToMmScale);

  // Refs for measurement visualization objects
  const measurementObjectsRef = useRef<THREE.Group>(new THREE.Group());

  /**
   * Handle viewer ready callback
   */
  const handleViewerReady = useCallback((scene: THREE.Scene, camera: THREE.Camera, meshObjects: THREE.Object3D[]) => {
    sceneRef.current = scene;
    cameraRef.current = camera;
    meshObjectsRef.current = meshObjects;
    setViewerReady(true);
    
    // Add measurement objects group to scene
    scene.add(measurementObjectsRef.current);
  }, []);

  /**
   * Handle click on the 3D viewer for measurements
   */
  const handleViewerClick = useCallback((event: MouseEvent) => {
    if (mode === 'none' || !viewerReady || !sceneRef.current || !cameraRef.current || meshObjectsRef.current.length === 0) {
      return;
    }

    const canvas = event.target as HTMLCanvasElement;
    const rect = canvas.getBoundingClientRect();
    
    // Calculate mouse position in normalized device coordinates (-1 to +1)
    const mouse = new THREE.Vector2();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    // Raycast to find intersection with mesh
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(mouse, cameraRef.current);

    // Find all meshes in the scene (excluding measurement objects)
    const meshes: THREE.Mesh[] = [];
    sceneRef.current.traverse((object) => {
      if (object instanceof THREE.Mesh && object !== measurementObjectsRef.current && !measurementObjectsRef.current.children.includes(object)) {
        meshes.push(object);
      }
    });

    const intersects = raycaster.intersectObjects(meshes, false);

    if (intersects.length > 0) {
      const intersection = intersects[0];
      const point: MeasurementPoint = {
        position: intersection.point.clone(),
        screenPosition: { x: event.clientX, y: event.clientY }
      };
      
      handleMeasurementClick(point);
    }
  }, [mode, viewerReady, handleMeasurementClick]);

  /**
   * Set up click event listener
   */
  useEffect(() => {
    if (!viewerContainerRef.current) return;

    const canvas = viewerContainerRef.current.querySelector('canvas');
    if (!canvas) return;

    canvas.addEventListener('click', handleViewerClick);

    return () => {
      canvas.removeEventListener('click', handleViewerClick);
    };
  }, [handleViewerClick]);

  /**
   * Access Three.js scene, camera, and mesh from the Viewer3D component
   */
  useEffect(() => {
    if (!viewerReady || !sceneRef.current) return;

    // Scene, camera, and meshes are now available through refs
    // The measurement objects group has been added to the scene in handleViewerReady
  }, [viewerReady]);

  /**
   * Update measurement visualizations
   */
  useEffect(() => {
    if (!sceneRef.current || !measurementObjectsRef.current) return;

    // Clear existing measurement objects
    measurementObjectsRef.current.clear();

    // Add distance measurement lines
    distanceMeasurements.forEach((measurement) => {
      // Create line
      const geometry = new THREE.BufferGeometry().setFromPoints([
        measurement.point1.position,
        measurement.point2.position
      ]);
      const material = new THREE.LineBasicMaterial({ color: 0xff0000, linewidth: 2 });
      const line = new THREE.Line(geometry, material);
      measurementObjectsRef.current.add(line);

      // Add spheres at endpoints
      const sphereGeometry = new THREE.SphereGeometry(0.5, 16, 16);
      const sphereMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 });
      
      const sphere1 = new THREE.Mesh(sphereGeometry, sphereMaterial);
      sphere1.position.copy(measurement.point1.position);
      measurementObjectsRef.current.add(sphere1);
      
      const sphere2 = new THREE.Mesh(sphereGeometry, sphereMaterial);
      sphere2.position.copy(measurement.point2.position);
      measurementObjectsRef.current.add(sphere2);
    });

    // Add area measurement polygons
    areaMeasurements.forEach((measurement) => {
      if (measurement.points.length < 3) return;

      // Create line loop for the perimeter
      const points = measurement.points.map(p => p.position);
      points.push(measurement.points[0].position); // Close the loop
      
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const material = new THREE.LineBasicMaterial({ color: 0x00ff00, linewidth: 2 });
      const line = new THREE.Line(geometry, material);
      measurementObjectsRef.current.add(line);

      // Add spheres at vertices
      const sphereGeometry = new THREE.SphereGeometry(0.5, 16, 16);
      const sphereMaterial = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
      
      measurement.points.forEach(point => {
        const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
        sphere.position.copy(point.position);
        measurementObjectsRef.current.add(sphere);
      });

      // Create semi-transparent fill
      const shape = new THREE.Shape();
      const positions2D = measurement.points.map(p => new THREE.Vector2(p.position.x, p.position.y));
      shape.moveTo(positions2D[0].x, positions2D[0].y);
      for (let i = 1; i < positions2D.length; i++) {
        shape.lineTo(positions2D[i].x, positions2D[i].y);
      }
      shape.lineTo(positions2D[0].x, positions2D[0].y);

      const fillGeometry = new THREE.ShapeGeometry(shape);
      const fillMaterial = new THREE.MeshBasicMaterial({
        color: 0x00ff00,
        transparent: true,
        opacity: 0.3,
        side: THREE.DoubleSide
      });
      const fill = new THREE.Mesh(fillGeometry, fillMaterial);
      measurementObjectsRef.current.add(fill);
    });

    // Add temporary points
    tempPoints.forEach((point) => {
      const sphereGeometry = new THREE.SphereGeometry(0.5, 16, 16);
      const color = mode === 'distance' ? 0xff0000 : 0x00ff00;
      const sphereMaterial = new THREE.MeshBasicMaterial({ color });
      
      const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
      sphere.position.copy(point.position);
      measurementObjectsRef.current.add(sphere);
    });

    // Add temporary line for distance measurement
    if (mode === 'distance' && tempPoints.length === 1) {
      // This would show a preview line following the cursor
      // Implementation would require mouse move tracking
    }

    // Add temporary polygon for area measurement
    if (mode === 'area' && tempPoints.length >= 2) {
      const points = tempPoints.map(p => p.position);
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const material = new THREE.LineBasicMaterial({ color: 0x00ff00, linewidth: 2 });
      const line = new THREE.LineStrip(geometry, material);
      measurementObjectsRef.current.add(line);
    }

  }, [distanceMeasurements, areaMeasurements, tempPoints, mode]);

  /**
   * Cleanup measurement objects when component unmounts
   */
  useEffect(() => {
    return () => {
      if (sceneRef.current && measurementObjectsRef.current) {
        sceneRef.current.remove(measurementObjectsRef.current);
      }
    };
  }, []);

  return (
    <div style={{ display: 'flex', gap: '20px' }}>
      <div ref={viewerContainerRef} style={{ flex: 1 }}>
        <Viewer3D
          mesh={mesh}
          width={width}
          height={height}
          showWireframe={showWireframe}
          enableControls={enableControls}
          layerConfig={layerConfig}
          onLayerConfigChange={onLayerConfigChange}
          onReady={handleViewerReady}
        />
      </div>
      
      <MeasurementTools
        mode={mode}
        onModeChange={setMode}
        distanceMeasurements={distanceMeasurements}
        areaMeasurements={areaMeasurements}
        onAddDistanceMeasurement={() => {}}
        onAddAreaMeasurement={() => {}}
        onClearMeasurements={clearMeasurements}
      />
    </div>
  );
};

export default Viewer3DWithMeasurements;
