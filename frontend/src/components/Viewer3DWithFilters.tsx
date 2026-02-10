import React, { useState } from 'react';
import Viewer3D, { SeverityFilterConfig } from './Viewer3D';
import SeverityFilter from './SeverityFilter';
import { Mesh } from '../types';

interface Viewer3DWithFiltersProps {
  mesh: Mesh;
  width?: number;
  height?: number;
}

/**
 * Example component demonstrating Viewer3D with severity filtering
 * 
 * Features:
 * - Integrated severity filter controls
 * - Real-time filter updates
 * - Side-by-side layout with viewer and controls
 * 
 * Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
 */
const Viewer3DWithFilters: React.FC<Viewer3DWithFiltersProps> = ({
  mesh,
  width = 800,
  height = 600
}) => {
  const [filterConfig, setFilterConfig] = useState<SeverityFilterConfig>({
    pigmentation: { low: true, medium: true, high: true },
    wrinkles: { micro: true, regular: true },
    allCombined: true
  });

  const handleFilterChange = (newConfig: SeverityFilterConfig) => {
    setFilterConfig(newConfig);
  };

  return (
    <div style={styles.container}>
      <div style={styles.viewerContainer}>
        <Viewer3D
          mesh={mesh}
          width={width}
          height={height}
          enableControls={true}
          severityFilterConfig={filterConfig}
        />
      </div>
      <div style={styles.controlsContainer}>
        <SeverityFilter
          filterConfig={filterConfig}
          onFilterChange={handleFilterChange}
        />
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    gap: '20px',
    padding: '20px',
    backgroundColor: '#fafafa',
    minHeight: '100vh'
  },
  viewerContainer: {
    flex: 1,
    backgroundColor: 'white',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
    overflow: 'hidden'
  },
  controlsContainer: {
    flexShrink: 0
  }
};

export default Viewer3DWithFilters;
