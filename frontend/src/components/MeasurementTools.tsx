import React, { useState, useCallback } from 'react';
import * as THREE from 'three';

export interface MeasurementPoint {
  position: THREE.Vector3;
  screenPosition: { x: number; y: number };
}

export interface DistanceMeasurement {
  id: string;
  point1: MeasurementPoint;
  point2: MeasurementPoint;
  distance: number; // in mm
}

export interface AreaMeasurement {
  id: string;
  points: MeasurementPoint[];
  area: number; // in mm²
}

export type MeasurementMode = 'none' | 'distance' | 'area';

export interface MeasurementToolsProps {
  mode: MeasurementMode;
  onModeChange: (mode: MeasurementMode) => void;
  distanceMeasurements: DistanceMeasurement[];
  areaMeasurements: AreaMeasurement[];
  onAddDistanceMeasurement: (measurement: DistanceMeasurement) => void;
  onAddAreaMeasurement: (measurement: AreaMeasurement) => void;
  onClearMeasurements: () => void;
}

/**
 * Measurement tools UI component
 * Provides controls for distance and area measurements
 * 
 * Requirements: 3.7
 */
const MeasurementTools: React.FC<MeasurementToolsProps> = ({
  mode,
  onModeChange,
  distanceMeasurements,
  areaMeasurements,
  onAddDistanceMeasurement,
  onAddAreaMeasurement,
  onClearMeasurements
}) => {
  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Measurement Tools</h3>
      
      <div style={styles.buttonGroup}>
        <button
          style={{
            ...styles.button,
            ...(mode === 'distance' ? styles.buttonActive : {})
          }}
          onClick={() => onModeChange(mode === 'distance' ? 'none' : 'distance')}
        >
          📏 Distance
        </button>
        
        <button
          style={{
            ...styles.button,
            ...(mode === 'area' ? styles.buttonActive : {})
          }}
          onClick={() => onModeChange(mode === 'area' ? 'none' : 'area')}
        >
          📐 Area
        </button>
        
        <button
          style={styles.button}
          onClick={onClearMeasurements}
          disabled={distanceMeasurements.length === 0 && areaMeasurements.length === 0}
        >
          🗑️ Clear
        </button>
      </div>
      
      {mode !== 'none' && (
        <div style={styles.instructions}>
          {mode === 'distance' && (
            <p>Click two points on the mesh to measure distance</p>
          )}
          {mode === 'area' && (
            <p>Click multiple points to define an area (click first point again to close)</p>
          )}
        </div>
      )}
      
      <div style={styles.measurements}>
        {distanceMeasurements.length > 0 && (
          <div style={styles.measurementSection}>
            <h4 style={styles.sectionTitle}>Distance Measurements</h4>
            {distanceMeasurements.map((m, idx) => (
              <div key={m.id} style={styles.measurementItem}>
                <span>#{idx + 1}:</span>
                <span style={styles.value}>{m.distance.toFixed(2)} mm</span>
              </div>
            ))}
          </div>
        )}
        
        {areaMeasurements.length > 0 && (
          <div style={styles.measurementSection}>
            <h4 style={styles.sectionTitle}>Area Measurements</h4>
            {areaMeasurements.map((m, idx) => (
              <div key={m.id} style={styles.measurementItem}>
                <span>#{idx + 1}:</span>
                <span style={styles.value}>{m.area.toFixed(2)} mm²</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

const styles = {
  container: {
    padding: '15px',
    backgroundColor: '#f5f5f5',
    borderRadius: '8px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
    minWidth: '250px'
  },
  title: {
    margin: '0 0 15px 0',
    fontSize: '16px',
    fontWeight: 'bold' as const,
    color: '#333'
  },
  buttonGroup: {
    display: 'flex',
    gap: '8px',
    marginBottom: '15px'
  },
  button: {
    flex: 1,
    padding: '8px 12px',
    border: '1px solid #ccc',
    borderRadius: '4px',
    backgroundColor: '#fff',
    cursor: 'pointer',
    fontSize: '14px',
    transition: 'all 0.2s'
  },
  buttonActive: {
    backgroundColor: '#007bff',
    color: '#fff',
    borderColor: '#007bff'
  },
  instructions: {
    padding: '10px',
    backgroundColor: '#e3f2fd',
    borderRadius: '4px',
    marginBottom: '15px',
    fontSize: '13px',
    color: '#1976d2'
  },
  measurements: {
    marginTop: '15px'
  },
  measurementSection: {
    marginBottom: '15px'
  },
  sectionTitle: {
    margin: '0 0 8px 0',
    fontSize: '14px',
    fontWeight: 'bold' as const,
    color: '#555'
  },
  measurementItem: {
    display: 'flex',
    justifyContent: 'space-between',
    padding: '6px 8px',
    backgroundColor: '#fff',
    borderRadius: '4px',
    marginBottom: '4px',
    fontSize: '13px'
  },
  value: {
    fontWeight: 'bold' as const,
    color: '#007bff'
  }
};

export default MeasurementTools;
