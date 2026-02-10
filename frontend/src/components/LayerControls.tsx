import React from 'react';
import { LayerConfig } from './Viewer3D';

interface LayerControlsProps {
  layerConfig: LayerConfig;
  onLayerConfigChange: (config: LayerConfig) => void;
}

/**
 * Layer controls component for managing visibility and transparency of mesh layers
 * 
 * Features:
 * - Toggle visibility for base, pigmentation, and wrinkle layers
 * - Adjust transparency (opacity) for each layer
 * - Real-time updates to 3D visualization
 * 
 * Requirements: 4.6, 4.7
 */
const LayerControls: React.FC<LayerControlsProps> = ({
  layerConfig,
  onLayerConfigChange
}) => {
  
  const handleVisibilityToggle = (layer: 'base' | 'pigmentation' | 'wrinkles') => {
    const newConfig = {
      ...layerConfig,
      [layer]: {
        ...layerConfig[layer],
        visible: !layerConfig[layer].visible
      }
    };
    onLayerConfigChange(newConfig);
  };
  
  const handleOpacityChange = (layer: 'base' | 'pigmentation' | 'wrinkles', opacity: number) => {
    const newConfig = {
      ...layerConfig,
      [layer]: {
        ...layerConfig[layer],
        opacity: opacity
      }
    };
    onLayerConfigChange(newConfig);
  };
  
  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Layer Controls</h3>
      
      {/* Base Layer */}
      <div style={styles.layerSection}>
        <div style={styles.layerHeader}>
          <label style={styles.label}>
            <input
              type="checkbox"
              checked={layerConfig.base.visible}
              onChange={() => handleVisibilityToggle('base')}
              style={styles.checkbox}
            />
            <span style={styles.layerName}>Base Mesh</span>
          </label>
        </div>
        <div style={styles.sliderContainer}>
          <label style={styles.sliderLabel}>
            Opacity: {Math.round(layerConfig.base.opacity * 100)}%
          </label>
          <input
            type="range"
            min="0"
            max="100"
            value={layerConfig.base.opacity * 100}
            onChange={(e) => handleOpacityChange('base', parseInt(e.target.value) / 100)}
            disabled={!layerConfig.base.visible}
            style={styles.slider}
          />
        </div>
      </div>
      
      {/* Pigmentation Layer */}
      <div style={styles.layerSection}>
        <div style={styles.layerHeader}>
          <label style={styles.label}>
            <input
              type="checkbox"
              checked={layerConfig.pigmentation.visible}
              onChange={() => handleVisibilityToggle('pigmentation')}
              style={styles.checkbox}
            />
            <span style={styles.layerName}>Pigmentation</span>
          </label>
          <div style={styles.colorLegend}>
            <span style={{...styles.colorBox, backgroundColor: '#FFE5B4'}} title="Low" />
            <span style={{...styles.colorBox, backgroundColor: '#FFA500'}} title="Medium" />
            <span style={{...styles.colorBox, backgroundColor: '#8B0000'}} title="High" />
          </div>
        </div>
        <div style={styles.sliderContainer}>
          <label style={styles.sliderLabel}>
            Opacity: {Math.round(layerConfig.pigmentation.opacity * 100)}%
          </label>
          <input
            type="range"
            min="0"
            max="100"
            value={layerConfig.pigmentation.opacity * 100}
            onChange={(e) => handleOpacityChange('pigmentation', parseInt(e.target.value) / 100)}
            disabled={!layerConfig.pigmentation.visible}
            style={styles.slider}
          />
        </div>
      </div>
      
      {/* Wrinkles Layer */}
      <div style={styles.layerSection}>
        <div style={styles.layerHeader}>
          <label style={styles.label}>
            <input
              type="checkbox"
              checked={layerConfig.wrinkles.visible}
              onChange={() => handleVisibilityToggle('wrinkles')}
              style={styles.checkbox}
            />
            <span style={styles.layerName}>Wrinkles</span>
          </label>
          <div style={styles.colorLegend}>
            <span style={{...styles.colorBox, backgroundColor: '#ADD8E6'}} title="Micro" />
            <span style={{...styles.colorBox, backgroundColor: '#00008B'}} title="Regular" />
          </div>
        </div>
        <div style={styles.sliderContainer}>
          <label style={styles.sliderLabel}>
            Opacity: {Math.round(layerConfig.wrinkles.opacity * 100)}%
          </label>
          <input
            type="range"
            min="0"
            max="100"
            value={layerConfig.wrinkles.opacity * 100}
            onChange={(e) => handleOpacityChange('wrinkles', parseInt(e.target.value) / 100)}
            disabled={!layerConfig.wrinkles.visible}
            style={styles.slider}
          />
        </div>
      </div>
      
      {/* Quick Actions */}
      <div style={styles.quickActions}>
        <button
          onClick={() => onLayerConfigChange({
            base: { visible: true, opacity: 1.0 },
            pigmentation: { visible: true, opacity: 0.7 },
            wrinkles: { visible: true, opacity: 0.7 }
          })}
          style={styles.button}
        >
          Show All
        </button>
        <button
          onClick={() => onLayerConfigChange({
            base: { visible: true, opacity: 1.0 },
            pigmentation: { visible: false, opacity: 0.7 },
            wrinkles: { visible: false, opacity: 0.7 }
          })}
          style={styles.button}
        >
          Base Only
        </button>
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    padding: '20px',
    backgroundColor: '#f5f5f5',
    borderRadius: '8px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
    minWidth: '280px',
    maxWidth: '350px'
  },
  title: {
    margin: '0 0 20px 0',
    fontSize: '18px',
    fontWeight: 'bold',
    color: '#333'
  },
  layerSection: {
    marginBottom: '20px',
    padding: '15px',
    backgroundColor: 'white',
    borderRadius: '6px',
    border: '1px solid #e0e0e0'
  },
  layerHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '10px'
  },
  label: {
    display: 'flex',
    alignItems: 'center',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: '500'
  },
  checkbox: {
    marginRight: '8px',
    cursor: 'pointer',
    width: '18px',
    height: '18px'
  },
  layerName: {
    color: '#333'
  },
  colorLegend: {
    display: 'flex',
    gap: '4px'
  },
  colorBox: {
    width: '20px',
    height: '20px',
    borderRadius: '3px',
    border: '1px solid #ccc',
    cursor: 'help'
  },
  sliderContainer: {
    marginTop: '10px'
  },
  sliderLabel: {
    display: 'block',
    fontSize: '12px',
    color: '#666',
    marginBottom: '5px'
  },
  slider: {
    width: '100%',
    cursor: 'pointer'
  },
  quickActions: {
    display: 'flex',
    gap: '10px',
    marginTop: '20px'
  },
  button: {
    flex: 1,
    padding: '10px',
    backgroundColor: '#2196F3',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: '500',
    transition: 'background-color 0.2s'
  }
};

export default LayerControls;
