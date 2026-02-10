import { useState, useCallback, useRef } from 'react';
import * as THREE from 'three';
import {
  MeasurementMode,
  MeasurementPoint,
  DistanceMeasurement,
  AreaMeasurement
} from '../components/MeasurementTools';

/**
 * Hook for managing measurement state and calculations
 * 
 * Provides:
 * - Distance measurement between two points
 * - Area measurement for polygonal regions
 * - Conversion from 3D coordinates to mm using mesh scale
 * 
 * Requirements: 3.7
 */
export function useMeasurements(pixelToMmScale: number = 1.0) {
  const [mode, setMode] = useState<MeasurementMode>('none');
  const [distanceMeasurements, setDistanceMeasurements] = useState<DistanceMeasurement[]>([]);
  const [areaMeasurements, setAreaMeasurements] = useState<AreaMeasurement[]>([]);
  const [tempPoints, setTempPoints] = useState<MeasurementPoint[]>([]);
  
  const measurementIdCounter = useRef(0);

  /**
   * Calculate geodesic distance between two 3D points
   * Returns distance in mm
   */
  const calculateDistance = useCallback((p1: THREE.Vector3, p2: THREE.Vector3): number => {
    const distance = p1.distanceTo(p2);
    return distance * pixelToMmScale;
  }, [pixelToMmScale]);

  /**
   * Calculate area of a polygon defined by points
   * Uses the shoelace formula for 3D polygons projected onto best-fit plane
   * Returns area in mm²
   */
  const calculateArea = useCallback((points: MeasurementPoint[]): number => {
    if (points.length < 3) return 0;

    // Extract 3D positions
    const positions = points.map(p => p.position);

    // Calculate the best-fit plane using PCA (simplified: use first 3 points)
    const v1 = new THREE.Vector3().subVectors(positions[1], positions[0]);
    const v2 = new THREE.Vector3().subVectors(positions[2], positions[0]);
    const normal = new THREE.Vector3().crossVectors(v1, v2).normalize();

    // Project points onto the plane and calculate area using shoelace formula
    // Create a local 2D coordinate system on the plane
    const u = v1.normalize();
    const v = new THREE.Vector3().crossVectors(normal, u).normalize();

    // Project all points to 2D
    const points2D = positions.map(pos => {
      const relative = new THREE.Vector3().subVectors(pos, positions[0]);
      return {
        x: relative.dot(u),
        y: relative.dot(v)
      };
    });

    // Shoelace formula for 2D polygon area
    let area = 0;
    for (let i = 0; i < points2D.length; i++) {
      const j = (i + 1) % points2D.length;
      area += points2D[i].x * points2D[j].y;
      area -= points2D[j].x * points2D[i].y;
    }
    area = Math.abs(area) / 2;

    // Convert to mm²
    return area * pixelToMmScale * pixelToMmScale;
  }, [pixelToMmScale]);

  /**
   * Handle click on mesh for measurement
   */
  const handleMeasurementClick = useCallback((point: MeasurementPoint) => {
    if (mode === 'distance') {
      if (tempPoints.length === 0) {
        // First point
        setTempPoints([point]);
      } else {
        // Second point - complete measurement
        const distance = calculateDistance(tempPoints[0].position, point.position);
        const measurement: DistanceMeasurement = {
          id: `dist-${measurementIdCounter.current++}`,
          point1: tempPoints[0],
          point2: point,
          distance
        };
        setDistanceMeasurements(prev => [...prev, measurement]);
        setTempPoints([]);
      }
    } else if (mode === 'area') {
      // Check if clicking near the first point to close the polygon
      if (tempPoints.length >= 3) {
        const firstPoint = tempPoints[0];
        const distanceToFirst = point.position.distanceTo(firstPoint.position);
        
        // If close to first point (within 5 units), close the polygon
        if (distanceToFirst < 5) {
          const area = calculateArea(tempPoints);
          const measurement: AreaMeasurement = {
            id: `area-${measurementIdCounter.current++}`,
            points: [...tempPoints],
            area
          };
          setAreaMeasurements(prev => [...prev, measurement]);
          setTempPoints([]);
          return;
        }
      }
      
      // Add point to temporary list
      setTempPoints(prev => [...prev, point]);
    }
  }, [mode, tempPoints, calculateDistance, calculateArea]);

  /**
   * Clear all measurements
   */
  const clearMeasurements = useCallback(() => {
    setDistanceMeasurements([]);
    setAreaMeasurements([]);
    setTempPoints([]);
  }, []);

  /**
   * Cancel current measurement in progress
   */
  const cancelCurrentMeasurement = useCallback(() => {
    setTempPoints([]);
  }, []);

  /**
   * Change measurement mode
   */
  const changeMeasurementMode = useCallback((newMode: MeasurementMode) => {
    setMode(newMode);
    setTempPoints([]);
  }, []);

  return {
    mode,
    setMode: changeMeasurementMode,
    distanceMeasurements,
    areaMeasurements,
    tempPoints,
    handleMeasurementClick,
    clearMeasurements,
    cancelCurrentMeasurement,
    calculateDistance,
    calculateArea
  };
}
