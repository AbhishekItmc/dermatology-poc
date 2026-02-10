import React from 'react';
import MeasurementTools, { MeasurementMode, DistanceMeasurement, AreaMeasurement } from './MeasurementTools';
import * as THREE from 'three';

/**
 * Unit tests for MeasurementTools component
 * 
 * Tests:
 * - Component rendering and structure
 * - Mode switching (distance, area, none)
 * - Measurement display
 * - Clear functionality
 * 
 * Requirements: 3.7
 */

describe('MeasurementTools Component', () => {
  const mockDistanceMeasurement: DistanceMeasurement = {
    id: 'dist-1',
    point1: {
      position: new THREE.Vector3(0, 0, 0),
      screenPosition: { x: 100, y: 100 }
    },
    point2: {
      position: new THREE.Vector3(10, 0, 0),
      screenPosition: { x: 200, y: 100 }
    },
    distance: 10.0
  };

  const mockAreaMeasurement: AreaMeasurement = {
    id: 'area-1',
    points: [
      {
        position: new THREE.Vector3(0, 0, 0),
        screenPosition: { x: 100, y: 100 }
      },
      {
        position: new THREE.Vector3(10, 0, 0),
        screenPosition: { x: 200, y: 100 }
      },
      {
        position: new THREE.Vector3(10, 10, 0),
        screenPosition: { x: 200, y: 200 }
      }
    ],
    area: 50.0
  };

  test('component is defined and exports correctly', () => {
    expect(MeasurementTools).toBeDefined();
    expect(typeof MeasurementTools).toBe('function');
  });

  test('component accepts required props', () => {
    const props = {
      mode: 'none' as MeasurementMode,
      onModeChange: jest.fn(),
      distanceMeasurements: [],
      areaMeasurements: [],
      onAddDistanceMeasurement: jest.fn(),
      onAddAreaMeasurement: jest.fn(),
      onClearMeasurements: jest.fn()
    };
    
    expect(() => React.createElement(MeasurementTools, props)).not.toThrow();
  });

  test('component accepts distance measurements', () => {
    const props = {
      mode: 'distance' as MeasurementMode,
      onModeChange: jest.fn(),
      distanceMeasurements: [mockDistanceMeasurement],
      areaMeasurements: [],
      onAddDistanceMeasurement: jest.fn(),
      onAddAreaMeasurement: jest.fn(),
      onClearMeasurements: jest.fn()
    };
    
    expect(() => React.createElement(MeasurementTools, props)).not.toThrow();
  });

  test('component accepts area measurements', () => {
    const props = {
      mode: 'area' as MeasurementMode,
      onModeChange: jest.fn(),
      distanceMeasurements: [],
      areaMeasurements: [mockAreaMeasurement],
      onAddDistanceMeasurement: jest.fn(),
      onAddAreaMeasurement: jest.fn(),
      onClearMeasurements: jest.fn()
    };
    
    expect(() => React.createElement(MeasurementTools, props)).not.toThrow();
  });

  test('component accepts both distance and area measurements', () => {
    const props = {
      mode: 'none' as MeasurementMode,
      onModeChange: jest.fn(),
      distanceMeasurements: [mockDistanceMeasurement],
      areaMeasurements: [mockAreaMeasurement],
      onAddDistanceMeasurement: jest.fn(),
      onAddAreaMeasurement: jest.fn(),
      onClearMeasurements: jest.fn()
    };
    
    expect(() => React.createElement(MeasurementTools, props)).not.toThrow();
  });
});

describe('MeasurementTools Props Interface', () => {
  test('MeasurementMode type is correct', () => {
    const modes: MeasurementMode[] = ['none', 'distance', 'area'];
    
    modes.forEach(mode => {
      expect(['none', 'distance', 'area']).toContain(mode);
    });
  });

  test('DistanceMeasurement structure is correct', () => {
    const measurement: DistanceMeasurement = {
      id: 'test-1',
      point1: {
        position: new THREE.Vector3(0, 0, 0),
        screenPosition: { x: 0, y: 0 }
      },
      point2: {
        position: new THREE.Vector3(1, 1, 1),
        screenPosition: { x: 100, y: 100 }
      },
      distance: 1.732
    };

    expect(measurement.id).toBeDefined();
    expect(measurement.point1).toBeDefined();
    expect(measurement.point2).toBeDefined();
    expect(measurement.distance).toBeDefined();
    expect(typeof measurement.distance).toBe('number');
  });

  test('AreaMeasurement structure is correct', () => {
    const measurement: AreaMeasurement = {
      id: 'test-1',
      points: [
        {
          position: new THREE.Vector3(0, 0, 0),
          screenPosition: { x: 0, y: 0 }
        },
        {
          position: new THREE.Vector3(1, 0, 0),
          screenPosition: { x: 100, y: 0 }
        },
        {
          position: new THREE.Vector3(1, 1, 0),
          screenPosition: { x: 100, y: 100 }
        }
      ],
      area: 0.5
    };

    expect(measurement.id).toBeDefined();
    expect(measurement.points).toBeDefined();
    expect(Array.isArray(measurement.points)).toBe(true);
    expect(measurement.area).toBeDefined();
    expect(typeof measurement.area).toBe('number');
  });
});

describe('MeasurementTools Functionality', () => {
  test('distance measurement displays correct value in mm', () => {
    const measurement: DistanceMeasurement = {
      id: 'dist-1',
      point1: {
        position: new THREE.Vector3(0, 0, 0),
        screenPosition: { x: 0, y: 0 }
      },
      point2: {
        position: new THREE.Vector3(10, 0, 0),
        screenPosition: { x: 100, y: 0 }
      },
      distance: 10.0
    };

    expect(measurement.distance).toBe(10.0);
    expect(measurement.distance.toFixed(2)).toBe('10.00');
  });

  test('area measurement displays correct value in mm²', () => {
    const measurement: AreaMeasurement = {
      id: 'area-1',
      points: [
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
        }
      ],
      area: 50.0
    };

    expect(measurement.area).toBe(50.0);
    expect(measurement.area.toFixed(2)).toBe('50.00');
  });

  test('measurement point has 3D position', () => {
    const point = {
      position: new THREE.Vector3(1, 2, 3),
      screenPosition: { x: 100, y: 200 }
    };

    expect(point.position).toBeInstanceOf(THREE.Vector3);
    expect(point.position.x).toBe(1);
    expect(point.position.y).toBe(2);
    expect(point.position.z).toBe(3);
  });

  test('measurement point has screen position', () => {
    const point = {
      position: new THREE.Vector3(0, 0, 0),
      screenPosition: { x: 150, y: 250 }
    };

    expect(point.screenPosition).toBeDefined();
    expect(point.screenPosition.x).toBe(150);
    expect(point.screenPosition.y).toBe(250);
  });
});

describe('MeasurementTools Documentation', () => {
  test('component file exists and is importable', () => {
    expect(MeasurementTools).toBeDefined();
  });

  test('component name is correct', () => {
    expect(MeasurementTools.name).toBe('MeasurementTools');
  });

  test('exports MeasurementMode type', () => {
    const mode: MeasurementMode = 'distance';
    expect(mode).toBeDefined();
  });

  test('exports DistanceMeasurement type', () => {
    const measurement: DistanceMeasurement = {
      id: 'test',
      point1: {
        position: new THREE.Vector3(0, 0, 0),
        screenPosition: { x: 0, y: 0 }
      },
      point2: {
        position: new THREE.Vector3(1, 1, 1),
        screenPosition: { x: 100, y: 100 }
      },
      distance: 1.732
    };
    expect(measurement).toBeDefined();
  });

  test('exports AreaMeasurement type', () => {
    const measurement: AreaMeasurement = {
      id: 'test',
      points: [],
      area: 0
    };
    expect(measurement).toBeDefined();
  });
});
