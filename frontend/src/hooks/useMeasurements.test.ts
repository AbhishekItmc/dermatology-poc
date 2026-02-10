import { renderHook, act } from '@testing-library/react';
import { useMeasurements } from './useMeasurements';
import * as THREE from 'three';
import { MeasurementPoint } from '../components/MeasurementTools';

/**
 * Unit tests for useMeasurements hook
 * 
 * Tests:
 * - Distance calculation
 * - Area calculation
 * - Measurement state management
 * - Mode switching
 * 
 * Requirements: 3.7
 */

describe('useMeasurements Hook', () => {
  test('hook initializes with correct default values', () => {
    const { result } = renderHook(() => useMeasurements());

    expect(result.current.mode).toBe('none');
    expect(result.current.distanceMeasurements).toEqual([]);
    expect(result.current.areaMeasurements).toEqual([]);
    expect(result.current.tempPoints).toEqual([]);
  });

  test('hook accepts pixelToMmScale parameter', () => {
    const { result } = renderHook(() => useMeasurements(2.0));

    expect(result.current).toBeDefined();
  });

  test('setMode changes measurement mode', () => {
    const { result } = renderHook(() => useMeasurements());

    act(() => {
      result.current.setMode('distance');
    });

    expect(result.current.mode).toBe('distance');

    act(() => {
      result.current.setMode('area');
    });

    expect(result.current.mode).toBe('area');

    act(() => {
      result.current.setMode('none');
    });

    expect(result.current.mode).toBe('none');
  });

  test('calculateDistance returns correct distance in mm', () => {
    const { result } = renderHook(() => useMeasurements(1.0));

    const p1 = new THREE.Vector3(0, 0, 0);
    const p2 = new THREE.Vector3(10, 0, 0);

    const distance = result.current.calculateDistance(p1, p2);

    expect(distance).toBe(10.0);
  });

  test('calculateDistance applies pixelToMmScale', () => {
    const { result } = renderHook(() => useMeasurements(2.0));

    const p1 = new THREE.Vector3(0, 0, 0);
    const p2 = new THREE.Vector3(10, 0, 0);

    const distance = result.current.calculateDistance(p1, p2);

    expect(distance).toBe(20.0); // 10 * 2.0
  });

  test('calculateDistance works with 3D points', () => {
    const { result } = renderHook(() => useMeasurements(1.0));

    const p1 = new THREE.Vector3(0, 0, 0);
    const p2 = new THREE.Vector3(3, 4, 0);

    const distance = result.current.calculateDistance(p1, p2);

    expect(distance).toBe(5.0); // sqrt(3^2 + 4^2) = 5
  });

  test('calculateArea returns correct area for triangle', () => {
    const { result } = renderHook(() => useMeasurements(1.0));

    const points: MeasurementPoint[] = [
      {
        position: new THREE.Vector3(0, 0, 0),
        screenPosition: { x: 0, y: 0 }
      },
      {
        position: new THREE.Vector3(10, 0, 0),
        screenPosition: { x: 100, y: 0 }
      },
      {
        position: new THREE.Vector3(0, 10, 0),
        screenPosition: { x: 0, y: 100 }
      }
    ];

    const area = result.current.calculateArea(points);

    expect(area).toBeCloseTo(50.0, 1); // Area of right triangle: 0.5 * 10 * 10 = 50
  });

  test('calculateArea returns correct area for square', () => {
    const { result } = renderHook(() => useMeasurements(1.0));

    const points: MeasurementPoint[] = [
      {
        position: new THREE.Vector3(0, 0, 0),
        screenPosition: { x: 0, y: 0 }
      },
      {
        position: new THREE.Vector3(10, 0, 0),
        screenPosition: { x: 100, y: 0 }
      },
      {
        position: new THREE.Vector3(10, 10, 0),
        screenPosition: { x: 100, y: 100 }
      },
      {
        position: new THREE.Vector3(0, 10, 0),
        screenPosition: { x: 0, y: 100 }
      }
    ];

    const area = result.current.calculateArea(points);

    expect(area).toBeCloseTo(100.0, 1); // Area of square: 10 * 10 = 100
  });

  test('calculateArea applies pixelToMmScale squared', () => {
    const { result } = renderHook(() => useMeasurements(2.0));

    const points: MeasurementPoint[] = [
      {
        position: new THREE.Vector3(0, 0, 0),
        screenPosition: { x: 0, y: 0 }
      },
      {
        position: new THREE.Vector3(10, 0, 0),
        screenPosition: { x: 100, y: 0 }
      },
      {
        position: new THREE.Vector3(0, 10, 0),
        screenPosition: { x: 0, y: 100 }
      }
    ];

    const area = result.current.calculateArea(points);

    expect(area).toBeCloseTo(200.0, 1); // 50 * (2.0^2) = 200
  });

  test('calculateArea returns 0 for less than 3 points', () => {
    const { result } = renderHook(() => useMeasurements(1.0));

    const points: MeasurementPoint[] = [
      {
        position: new THREE.Vector3(0, 0, 0),
        screenPosition: { x: 0, y: 0 }
      },
      {
        position: new THREE.Vector3(10, 0, 0),
        screenPosition: { x: 100, y: 0 }
      }
    ];

    const area = result.current.calculateArea(points);

    expect(area).toBe(0);
  });

  test('handleMeasurementClick adds distance measurement', () => {
    const { result } = renderHook(() => useMeasurements(1.0));

    act(() => {
      result.current.setMode('distance');
    });

    const point1: MeasurementPoint = {
      position: new THREE.Vector3(0, 0, 0),
      screenPosition: { x: 0, y: 0 }
    };

    const point2: MeasurementPoint = {
      position: new THREE.Vector3(10, 0, 0),
      screenPosition: { x: 100, y: 0 }
    };

    act(() => {
      result.current.handleMeasurementClick(point1);
    });

    expect(result.current.tempPoints.length).toBe(1);

    act(() => {
      result.current.handleMeasurementClick(point2);
    });

    expect(result.current.distanceMeasurements.length).toBe(1);
    expect(result.current.distanceMeasurements[0].distance).toBe(10.0);
    expect(result.current.tempPoints.length).toBe(0);
  });

  test('handleMeasurementClick adds area measurement points', () => {
    const { result } = renderHook(() => useMeasurements(1.0));

    act(() => {
      result.current.setMode('area');
    });

    const point1: MeasurementPoint = {
      position: new THREE.Vector3(0, 0, 0),
      screenPosition: { x: 0, y: 0 }
    };

    const point2: MeasurementPoint = {
      position: new THREE.Vector3(10, 0, 0),
      screenPosition: { x: 100, y: 0 }
    };

    const point3: MeasurementPoint = {
      position: new THREE.Vector3(0, 10, 0),
      screenPosition: { x: 0, y: 100 }
    };

    act(() => {
      result.current.handleMeasurementClick(point1);
    });

    expect(result.current.tempPoints.length).toBe(1);

    act(() => {
      result.current.handleMeasurementClick(point2);
    });

    expect(result.current.tempPoints.length).toBe(2);

    act(() => {
      result.current.handleMeasurementClick(point3);
    });

    expect(result.current.tempPoints.length).toBe(3);
  });

  test('clearMeasurements removes all measurements', () => {
    const { result } = renderHook(() => useMeasurements(1.0));

    act(() => {
      result.current.setMode('distance');
    });

    const point1: MeasurementPoint = {
      position: new THREE.Vector3(0, 0, 0),
      screenPosition: { x: 0, y: 0 }
    };

    const point2: MeasurementPoint = {
      position: new THREE.Vector3(10, 0, 0),
      screenPosition: { x: 100, y: 0 }
    };

    act(() => {
      result.current.handleMeasurementClick(point1);
    });
    
    act(() => {
      result.current.handleMeasurementClick(point2);
    });

    expect(result.current.distanceMeasurements.length).toBe(1);

    act(() => {
      result.current.clearMeasurements();
    });

    expect(result.current.distanceMeasurements.length).toBe(0);
    expect(result.current.areaMeasurements.length).toBe(0);
    expect(result.current.tempPoints.length).toBe(0);
  });

  test('cancelCurrentMeasurement clears temp points', () => {
    const { result } = renderHook(() => useMeasurements(1.0));

    act(() => {
      result.current.setMode('distance');
    });

    const point1: MeasurementPoint = {
      position: new THREE.Vector3(0, 0, 0),
      screenPosition: { x: 0, y: 0 }
    };

    act(() => {
      result.current.handleMeasurementClick(point1);
    });

    expect(result.current.tempPoints.length).toBe(1);

    act(() => {
      result.current.cancelCurrentMeasurement();
    });

    expect(result.current.tempPoints.length).toBe(0);
  });

  test('changing mode clears temp points', () => {
    const { result } = renderHook(() => useMeasurements(1.0));

    act(() => {
      result.current.setMode('distance');
    });

    const point1: MeasurementPoint = {
      position: new THREE.Vector3(0, 0, 0),
      screenPosition: { x: 0, y: 0 }
    };

    act(() => {
      result.current.handleMeasurementClick(point1);
    });

    expect(result.current.tempPoints.length).toBe(1);

    act(() => {
      result.current.setMode('area');
    });

    expect(result.current.tempPoints.length).toBe(0);
  });
});

describe('useMeasurements Accuracy', () => {
  test('distance measurement is accurate within 2%', () => {
    const { result } = renderHook(() => useMeasurements(1.0));

    // Test various distances
    const testCases = [
      { p1: new THREE.Vector3(0, 0, 0), p2: new THREE.Vector3(100, 0, 0), expected: 100 },
      { p1: new THREE.Vector3(0, 0, 0), p2: new THREE.Vector3(0, 50, 0), expected: 50 },
      { p1: new THREE.Vector3(0, 0, 0), p2: new THREE.Vector3(30, 40, 0), expected: 50 },
      { p1: new THREE.Vector3(0, 0, 0), p2: new THREE.Vector3(3, 4, 12), expected: 13 }
    ];

    testCases.forEach(({ p1, p2, expected }) => {
      const distance = result.current.calculateDistance(p1, p2);
      const error = Math.abs(distance - expected) / expected;
      expect(error).toBeLessThan(0.02); // Within 2% error
    });
  });

  test('area measurement is accurate within 5%', () => {
    const { result } = renderHook(() => useMeasurements(1.0));

    // Test square
    const squarePoints: MeasurementPoint[] = [
      { position: new THREE.Vector3(0, 0, 0), screenPosition: { x: 0, y: 0 } },
      { position: new THREE.Vector3(10, 0, 0), screenPosition: { x: 100, y: 0 } },
      { position: new THREE.Vector3(10, 10, 0), screenPosition: { x: 100, y: 100 } },
      { position: new THREE.Vector3(0, 10, 0), screenPosition: { x: 0, y: 100 } }
    ];

    const squareArea = result.current.calculateArea(squarePoints);
    const squareError = Math.abs(squareArea - 100) / 100;
    expect(squareError).toBeLessThan(0.05); // Within 5% error

    // Test triangle
    const trianglePoints: MeasurementPoint[] = [
      { position: new THREE.Vector3(0, 0, 0), screenPosition: { x: 0, y: 0 } },
      { position: new THREE.Vector3(10, 0, 0), screenPosition: { x: 100, y: 0 } },
      { position: new THREE.Vector3(0, 10, 0), screenPosition: { x: 0, y: 100 } }
    ];

    const triangleArea = result.current.calculateArea(trianglePoints);
    const triangleError = Math.abs(triangleArea - 50) / 50;
    expect(triangleError).toBeLessThan(0.05); // Within 5% error
  });
});
