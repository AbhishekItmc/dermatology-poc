import React, { useState } from 'react';
import SeverityFilter, { SeverityFilterConfig } from './SeverityFilter';

/**
 * Example component demonstrating SeverityFilter usage
 * 
 * This example shows:
 * - Basic setup with state management
 * - Integration with visualization logic
 * - Real-time filter updates
 */
const SeverityFilterExample: React.FC = () => {
  
  const [filterConfig, setFilterConfig] = useState<SeverityFilterConfig>({
    selectedSeverities: ['Low', 'Medium', 'High'],
    allCombined: true
  });
  
  // Mock anomaly data for demonstration
  const mockAnomalies = [
    { id: '1', type: 'pigmentation', severity: 'Low', region: 'forehead' },
    { id: '2', type: 'pigmentation', severity: 'Medium', region: 'left_cheek' },
    { id: '3', type: 'pigmentation', severity: 'High', region: 'right_cheek' },
    { id: '4', type: 'wrinkle', severity: 'Low', region: 'forehead' },
    { id: '5', type: 'wrinkle', severity: 'Medium', region: 'periorbital_left' },
    { id: '6', type: 'wrinkle', severity: 'High', region: 'periorbital_right' }
  ];
  
  // Filter anomalies based on current filter config
  const filteredAnomalies = mockAnomalies.filter(anomaly => {
    if (filterConfig.selectedSeverities.length === 0) {
      return false; // Hide all if nothing selected
    }
    return filterConfig.selectedSeverities.includes(anomaly.severity as any);
  });
  
  const handleFilterChange = (newConfig: SeverityFilterConfig) => {
    console.log('Filter config changed:', newConfig);
    setFilterConfig(newConfig);
  };
  
  return (
    <div style={styles.container}>
      <h2 style={styles.pageTitle}>Severity Filter Example</h2>
      
      <div style={styles.layout}>
        {/* Filter Controls */}
        <div style={styles.sidebar}>
          <SeverityFilter
            filterConfig={filterConfig}
            onFilterConfigChange={handleFilterChange}
          />
        </div>
        
        {/* Visualization Area */}
        <div style={styles.mainContent}>
          <div style={styles.visualizationPanel}>
            <h3 style={styles.panelTitle}>3D Visualization (Mock)</h3>
            <div style={styles.mockViewer}>
              <div style={styles.mockViewerContent}>
                <p style={styles.mockText}>
                  🎨 3D Viewer would render here
                </p>
                <p style={styles.mockSubtext}>
                  Showing {filteredAnomalies.length} of {mockAnomalies.length} anomalies
                </p>
              </div>
            </div>
          </div>
          
          {/* Anomaly List */}
          <div style={styles.anomalyPanel}>
            <h3 style={styles.panelTitle}>Detected Anomalies</h3>
            <div style={styles.anomalyList}>
              {filteredAnomalies.length > 0 ? (
                filteredAnomalies.map(anomaly => (
                  <div key={anomaly.id} style={styles.anomalyItem}>
                    <div style={styles.anomalyHeader}>
                      <span style={styles.anomalyType}>
                        {anomaly.type === 'pigmentation' ? '🟡' : '📏'} {anomaly.type}
                      </span>
                      <span style={{
                        ...styles.severityBadge,
                        backgroundColor: getSeverityColor(anomaly.severity)
                      }}>
                        {anomaly.severity}
                      </span>
                    </div>
                    <div style={styles.anomalyRegion}>
                      Region: {anomaly.region}
                    </div>
                  </div>
                ))
              ) : (
                <div style={styles.emptyState}>
                  <p style={styles.emptyText}>No anomalies match the current filter</p>
                  <p style={styles.emptyHint}>
                    Select one or more severity levels to view anomalies
                  </p>
                </div>
              )}
            </div>
          </div>
          
          {/* Filter Status */}
          <div style={styles.statusPanel}>
            <h4 style={styles.statusTitle}>Current Filter Status</h4>
            <div style={styles.statusContent}>
              <div style={styles.statusRow}>
                <span style={styles.statusLabel}>All Combined:</span>
                <span style={styles.statusValue}>
                  {filterConfig.allCombined ? '✓ Yes' : '✗ No'}
                </span>
              </div>
              <div style={styles.statusRow}>
                <span style={styles.statusLabel}>Selected Severities:</span>
                <span style={styles.statusValue}>
                  {filterConfig.selectedSeverities.length > 0
                    ? filterConfig.selectedSeverities.join(', ')
                    : 'None'}
                </span>
              </div>
              <div style={styles.statusRow}>
                <span style={styles.statusLabel}>Visible Anomalies:</span>
                <span style={styles.statusValue}>
                  {filteredAnomalies.length} / {mockAnomalies.length}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Usage Instructions */}
      <div style={styles.instructions}>
        <h3 style={styles.instructionsTitle}>How to Use</h3>
        <ol style={styles.instructionsList}>
          <li>Click "All Combined" to show all severity levels at once</li>
          <li>Click individual severity buttons (Low/Medium/High) to toggle specific levels</li>
          <li>Use quick action buttons for common filter combinations</li>
          <li>The visualization updates in real-time as you change filters</li>
          <li>The active filters summary shows which severities are currently visible</li>
        </ol>
      </div>
    </div>
  );
};

// Helper function to get severity color
const getSeverityColor = (severity: string): string => {
  switch (severity) {
    case 'Low':
      return '#FFE5B4';
    case 'Medium':
      return '#FFA500';
    case 'High':
      return '#8B0000';
    default:
      return '#ccc';
  }
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    padding: '20px',
    maxWidth: '1400px',
    margin: '0 auto',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
  },
  pageTitle: {
    fontSize: '28px',
    fontWeight: 'bold',
    color: '#333',
    marginBottom: '30px',
    textAlign: 'center'
  },
  layout: {
    display: 'flex',
    gap: '20px',
    marginBottom: '30px'
  },
  sidebar: {
    flexShrink: 0
  },
  mainContent: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    gap: '20px'
  },
  visualizationPanel: {
    backgroundColor: 'white',
    borderRadius: '8px',
    padding: '20px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
  },
  panelTitle: {
    margin: '0 0 15px 0',
    fontSize: '18px',
    fontWeight: 'bold',
    color: '#333'
  },
  mockViewer: {
    backgroundColor: '#f0f0f0',
    borderRadius: '6px',
    height: '300px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    border: '2px dashed #ccc'
  },
  mockViewerContent: {
    textAlign: 'center'
  },
  mockText: {
    fontSize: '24px',
    color: '#666',
    margin: '0 0 10px 0'
  },
  mockSubtext: {
    fontSize: '14px',
    color: '#999',
    margin: 0
  },
  anomalyPanel: {
    backgroundColor: 'white',
    borderRadius: '8px',
    padding: '20px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
  },
  anomalyList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
    maxHeight: '400px',
    overflowY: 'auto'
  },
  anomalyItem: {
    padding: '12px',
    backgroundColor: '#f9f9f9',
    borderRadius: '6px',
    border: '1px solid #e0e0e0'
  },
  anomalyHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '6px'
  },
  anomalyType: {
    fontSize: '14px',
    fontWeight: '500',
    color: '#333',
    textTransform: 'capitalize'
  },
  severityBadge: {
    padding: '3px 10px',
    borderRadius: '12px',
    fontSize: '12px',
    fontWeight: '500',
    color: '#333',
    border: '1px solid rgba(0,0,0,0.1)'
  },
  anomalyRegion: {
    fontSize: '12px',
    color: '#666'
  },
  emptyState: {
    textAlign: 'center',
    padding: '40px 20px'
  },
  emptyText: {
    fontSize: '16px',
    color: '#666',
    margin: '0 0 10px 0'
  },
  emptyHint: {
    fontSize: '14px',
    color: '#999',
    margin: 0
  },
  statusPanel: {
    backgroundColor: '#e3f2fd',
    borderRadius: '8px',
    padding: '15px',
    border: '1px solid #90caf9'
  },
  statusTitle: {
    margin: '0 0 12px 0',
    fontSize: '16px',
    fontWeight: 'bold',
    color: '#1976d2'
  },
  statusContent: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px'
  },
  statusRow: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: '14px'
  },
  statusLabel: {
    fontWeight: '500',
    color: '#1565c0'
  },
  statusValue: {
    color: '#0d47a1',
    fontFamily: 'monospace'
  },
  instructions: {
    backgroundColor: '#fff3e0',
    borderRadius: '8px',
    padding: '20px',
    border: '1px solid #ffb74d'
  },
  instructionsTitle: {
    margin: '0 0 15px 0',
    fontSize: '18px',
    fontWeight: 'bold',
    color: '#e65100'
  },
  instructionsList: {
    margin: 0,
    paddingLeft: '20px',
    color: '#e65100',
    lineHeight: '1.8'
  }
};

export default SeverityFilterExample;
