import React from 'react';

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

export interface RegionConfig {
  selectedRegion: FacialRegion;
  highlightIntensity: number; // 0-1
}

interface RegionControlsProps {
  regionConfig: RegionConfig;
  onRegionConfigChange: (config: RegionConfig) => void;
  onZoomToRegion?: (region: FacialRegion) => void;
}

/**
 * Region controls component for isolating and highlighting specific facial regions
 * 
 * Features:
 * - Select specific facial regions to isolate
 * - Adjust highlight intensity for selected region
 * - Zoom to region functionality
 * - Quick access to common regions
 * 
 * Requirements: 3.6
 */
const RegionControls: React.FC<RegionControlsProps> = ({
  regionConfig,
  onRegionConfigChange,
  onZoomToRegion
}) => {
  
  const regions: { value: FacialRegion; label: string }[] = [
    { value: 'all', label: 'All Regions' },
    { value: 'forehead', label: 'Forehead' },
    { value: 'left_cheek', label: 'Left Cheek' },
    { value: 'right_cheek', label: 'Right Cheek' },
    { value: 'periorbital_left', label: 'Left Eye Area' },
    { value: 'periorbital_right', label: 'Right Eye Area' },
    { value: 'nose', label: 'Nose' },
    { value: 'mouth', label: 'Mouth' },
    { value: 'chin', label: 'Chin' }
  ];
  
  const handleRegionChange = (region: FacialRegion) => {
    const newConfig = {
      ...regionConfig,
      selectedRegion: region
    };
    onRegionConfigChange(newConfig);
  };
  
  const handleHighlightIntensityChange = (intensity: number) => {
    const newConfig = {
      ...regionConfig,
      highlightIntensity: intensity
    };
    onRegionConfigChange(newConfig);
  };
  
  const handleZoomToRegion = () => {
    if (onZoomToRegion && regionConfig.selectedRegion !== 'all') {
      onZoomToRegion(regionConfig.selectedRegion);
    }
  };
  
  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Region Isolation</h3>
      
      {/* Region Selection */}
      <div style={styles.section}>
        <label style={styles.label}>Select Region:</label>
        <select
          value={regionConfig.selectedRegion}
          onChange={(e) => handleRegionChange(e.target.value as FacialRegion)}
          style={styles.select}
        >
          {regions.map(region => (
            <option key={region.value} value={region.value}>
              {region.label}
            </option>
          ))}
        </select>
      </div>
      
      {/* Highlight Intensity */}
      {regionConfig.selectedRegion !== 'all' && (
        <div style={styles.section}>
          <label style={styles.sliderLabel}>
            Highlight Intensity: {Math.round(regionConfig.highlightIntensity * 100)}%
          </label>
          <input
            type="range"
            min="0"
            max="100"
            value={regionConfig.highlightIntensity * 100}
            onChange={(e) => handleHighlightIntensityChange(parseInt(e.target.value) / 100)}
            style={styles.slider}
          />
          <div style={styles.hint}>
            Higher values make the selected region more prominent
          </div>
        </div>
      )}
      
      {/* Zoom to Region Button */}
      {regionConfig.selectedRegion !== 'all' && onZoomToRegion && (
        <div style={styles.section}>
          <button
            onClick={handleZoomToRegion}
            style={styles.zoomButton}
          >
            🔍 Zoom to {regions.find(r => r.value === regionConfig.selectedRegion)?.label}
          </button>
        </div>
      )}
      
      {/* Quick Region Buttons */}
      <div style={styles.quickRegions}>
        <div style={styles.quickRegionsLabel}>Quick Select:</div>
        <div style={styles.quickRegionsGrid}>
          <button
            onClick={() => handleRegionChange('forehead')}
            style={{
              ...styles.quickButton,
              ...(regionConfig.selectedRegion === 'forehead' ? styles.quickButtonActive : {})
            }}
          >
            Forehead
          </button>
          <button
            onClick={() => handleRegionChange('left_cheek')}
            style={{
              ...styles.quickButton,
              ...(regionConfig.selectedRegion === 'left_cheek' ? styles.quickButtonActive : {})
            }}
          >
            L Cheek
          </button>
          <button
            onClick={() => handleRegionChange('right_cheek')}
            style={{
              ...styles.quickButton,
              ...(regionConfig.selectedRegion === 'right_cheek' ? styles.quickButtonActive : {})
            }}
          >
            R Cheek
          </button>
          <button
            onClick={() => handleRegionChange('periorbital_left')}
            style={{
              ...styles.quickButton,
              ...(regionConfig.selectedRegion === 'periorbital_left' ? styles.quickButtonActive : {})
            }}
          >
            L Eye
          </button>
          <button
            onClick={() => handleRegionChange('periorbital_right')}
            style={{
              ...styles.quickButton,
              ...(regionConfig.selectedRegion === 'periorbital_right' ? styles.quickButtonActive : {})
            }}
          >
            R Eye
          </button>
          <button
            onClick={() => handleRegionChange('all')}
            style={{
              ...styles.quickButton,
              ...(regionConfig.selectedRegion === 'all' ? styles.quickButtonActive : {})
            }}
          >
            All
          </button>
        </div>
      </div>
      
      {/* Info */}
      <div style={styles.info}>
        <strong>Tip:</strong> Select a region to isolate and highlight it. 
        Use the zoom button to focus the camera on the selected area.
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
  section: {
    marginBottom: '20px',
    padding: '15px',
    backgroundColor: 'white',
    borderRadius: '6px',
    border: '1px solid #e0e0e0'
  },
  label: {
    display: 'block',
    fontSize: '14px',
    fontWeight: '500',
    color: '#333',
    marginBottom: '8px'
  },
  select: {
    width: '100%',
    padding: '8px',
    fontSize: '14px',
    border: '1px solid #ccc',
    borderRadius: '4px',
    backgroundColor: 'white',
    cursor: 'pointer'
  },
  sliderLabel: {
    display: 'block',
    fontSize: '14px',
    fontWeight: '500',
    color: '#333',
    marginBottom: '8px'
  },
  slider: {
    width: '100%',
    cursor: 'pointer',
    marginBottom: '8px'
  },
  hint: {
    fontSize: '12px',
    color: '#666',
    fontStyle: 'italic'
  },
  zoomButton: {
    width: '100%',
    padding: '12px',
    backgroundColor: '#4CAF50',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: '500',
    transition: 'background-color 0.2s'
  },
  quickRegions: {
    marginBottom: '20px'
  },
  quickRegionsLabel: {
    fontSize: '14px',
    fontWeight: '500',
    color: '#333',
    marginBottom: '10px'
  },
  quickRegionsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gap: '8px'
  },
  quickButton: {
    padding: '8px 4px',
    backgroundColor: 'white',
    color: '#333',
    border: '1px solid #ccc',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '12px',
    fontWeight: '500',
    transition: 'all 0.2s'
  },
  quickButtonActive: {
    backgroundColor: '#2196F3',
    color: 'white',
    borderColor: '#2196F3'
  },
  info: {
    padding: '12px',
    backgroundColor: '#e3f2fd',
    borderRadius: '4px',
    fontSize: '12px',
    color: '#1976d2',
    lineHeight: '1.5'
  }
};

export default RegionControls;
export type { RegionConfig };
