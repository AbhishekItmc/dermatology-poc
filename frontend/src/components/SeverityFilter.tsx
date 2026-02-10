import React from 'react';

export type SeverityLevel = 'Low' | 'Medium' | 'High';
export type AnomalyType = 'pigmentation' | 'wrinkles';

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

interface SeverityFilterProps {
  filterConfig: SeverityFilterConfig;
  onFilterChange: (config: SeverityFilterConfig) => void;
}

/**
 * Severity filter component for filtering anomalies by severity level
 * 
 * Features:
 * - Toggle controls for each severity level (Low, Medium, High)
 * - Separate controls for pigmentation and wrinkles
 * - "All Combined" option to show all anomalies
 * - Multi-select filtering support
 * - Real-time visualization updates
 * - Clear visual feedback for selected filters
 * 
 * Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
 */
const SeverityFilter: React.FC<SeverityFilterProps> = ({
  filterConfig,
  onFilterChange
}) => {
  
  const handlePigmentationToggle = (level: 'low' | 'medium' | 'high') => {
    const newConfig = {
      ...filterConfig,
      pigmentation: {
        ...filterConfig.pigmentation,
        [level]: !filterConfig.pigmentation[level]
      },
      allCombined: false // Disable "all combined" when individual filters are toggled
    };
    onFilterChange(newConfig);
  };
  
  const handleWrinkleToggle = (type: 'micro' | 'regular') => {
    const newConfig = {
      ...filterConfig,
      wrinkles: {
        ...filterConfig.wrinkles,
        [type]: !filterConfig.wrinkles[type]
      },
      allCombined: false // Disable "all combined" when individual filters are toggled
    };
    onFilterChange(newConfig);
  };
  
  const handleAllCombinedToggle = () => {
    if (!filterConfig.allCombined) {
      // Enable all filters
      onFilterChange({
        pigmentation: { low: true, medium: true, high: true },
        wrinkles: { micro: true, regular: true },
        allCombined: true
      });
    } else {
      // Disable all filters
      onFilterChange({
        pigmentation: { low: false, medium: false, high: false },
        wrinkles: { micro: false, regular: false },
        allCombined: false
      });
    }
  };
  
  const handleShowAllPigmentation = () => {
    onFilterChange({
      ...filterConfig,
      pigmentation: { low: true, medium: true, high: true },
      allCombined: false
    });
  };
  
  const handleShowAllWrinkles = () => {
    onFilterChange({
      ...filterConfig,
      wrinkles: { micro: true, regular: true },
      allCombined: false
    });
  };
  
  const handleClearAll = () => {
    onFilterChange({
      pigmentation: { low: false, medium: false, high: false },
      wrinkles: { micro: false, regular: false },
      allCombined: false
    });
  };
  
  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Severity Filters</h3>
      
      {/* All Combined Toggle */}
      <div style={styles.allCombinedSection}>
        <label style={styles.allCombinedLabel}>
          <input
            type="checkbox"
            checked={filterConfig.allCombined}
            onChange={handleAllCombinedToggle}
            style={styles.checkbox}
          />
          <span style={styles.allCombinedText}>Show All Combined</span>
        </label>
        <p style={styles.helpText}>Display all detected anomalies regardless of severity</p>
      </div>
      
      <div style={styles.divider} />
      
      {/* Pigmentation Filters */}
      <div style={styles.filterSection}>
        <div style={styles.sectionHeader}>
          <h4 style={styles.sectionTitle}>Pigmentation</h4>
          <button
            onClick={handleShowAllPigmentation}
            style={styles.showAllButton}
            title="Show all pigmentation levels"
          >
            All
          </button>
        </div>
        
        <div style={styles.filterOptions}>
          <label style={styles.filterLabel}>
            <input
              type="checkbox"
              checked={filterConfig.pigmentation.low}
              onChange={() => handlePigmentationToggle('low')}
              disabled={filterConfig.allCombined}
              style={styles.checkbox}
            />
            <span 
              style={{
                ...styles.colorIndicator, 
                backgroundColor: '#FFE5B4',
                border: '1px solid #ddd'
              }} 
            />
            <span style={styles.filterText}>Low Severity</span>
          </label>
          
          <label style={styles.filterLabel}>
            <input
              type="checkbox"
              checked={filterConfig.pigmentation.medium}
              onChange={() => handlePigmentationToggle('medium')}
              disabled={filterConfig.allCombined}
              style={styles.checkbox}
            />
            <span 
              style={{
                ...styles.colorIndicator, 
                backgroundColor: '#FFA500'
              }} 
            />
            <span style={styles.filterText}>Medium Severity</span>
          </label>
          
          <label style={styles.filterLabel}>
            <input
              type="checkbox"
              checked={filterConfig.pigmentation.high}
              onChange={() => handlePigmentationToggle('high')}
              disabled={filterConfig.allCombined}
              style={styles.checkbox}
            />
            <span 
              style={{
                ...styles.colorIndicator, 
                backgroundColor: '#8B0000'
              }} 
            />
            <span style={styles.filterText}>High Severity</span>
          </label>
        </div>
      </div>
      
      <div style={styles.divider} />
      
      {/* Wrinkle Filters */}
      <div style={styles.filterSection}>
        <div style={styles.sectionHeader}>
          <h4 style={styles.sectionTitle}>Wrinkles</h4>
          <button
            onClick={handleShowAllWrinkles}
            style={styles.showAllButton}
            title="Show all wrinkle types"
          >
            All
          </button>
        </div>
        
        <div style={styles.filterOptions}>
          <label style={styles.filterLabel}>
            <input
              type="checkbox"
              checked={filterConfig.wrinkles.micro}
              onChange={() => handleWrinkleToggle('micro')}
              disabled={filterConfig.allCombined}
              style={styles.checkbox}
            />
            <span 
              style={{
                ...styles.colorIndicator, 
                backgroundColor: '#ADD8E6',
                border: '1px solid #999'
              }} 
            />
            <span style={styles.filterText}>Micro-wrinkles (&lt;0.5mm)</span>
          </label>
          
          <label style={styles.filterLabel}>
            <input
              type="checkbox"
              checked={filterConfig.wrinkles.regular}
              onChange={() => handleWrinkleToggle('regular')}
              disabled={filterConfig.allCombined}
              style={styles.checkbox}
            />
            <span 
              style={{
                ...styles.colorIndicator, 
                backgroundColor: '#00008B'
              }} 
            />
            <span style={styles.filterText}>Regular Wrinkles</span>
          </label>
        </div>
      </div>
      
      {/* Quick Actions */}
      <div style={styles.quickActions}>
        <button
          onClick={handleClearAll}
          style={styles.clearButton}
        >
          Clear All
        </button>
      </div>
      
      {/* Active Filter Summary */}
      <div style={styles.summary}>
        <p style={styles.summaryText}>
          {getSummaryText(filterConfig)}
        </p>
      </div>
    </div>
  );
};

/**
 * Generate summary text describing active filters
 */
function getSummaryText(config: SeverityFilterConfig): string {
  if (config.allCombined) {
    return 'Showing all anomalies';
  }
  
  const activeFilters: string[] = [];
  
  // Count pigmentation filters
  const pigCount = [config.pigmentation.low, config.pigmentation.medium, config.pigmentation.high]
    .filter(Boolean).length;
  if (pigCount > 0) {
    const levels: string[] = [];
    if (config.pigmentation.low) levels.push('Low');
    if (config.pigmentation.medium) levels.push('Medium');
    if (config.pigmentation.high) levels.push('High');
    activeFilters.push(`Pigmentation: ${levels.join(', ')}`);
  }
  
  // Count wrinkle filters
  const wrinkleCount = [config.wrinkles.micro, config.wrinkles.regular]
    .filter(Boolean).length;
  if (wrinkleCount > 0) {
    const types: string[] = [];
    if (config.wrinkles.micro) types.push('Micro');
    if (config.wrinkles.regular) types.push('Regular');
    activeFilters.push(`Wrinkles: ${types.join(', ')}`);
  }
  
  if (activeFilters.length === 0) {
    return 'No filters active (all anomalies hidden)';
  }
  
  return `Active: ${activeFilters.join(' | ')}`;
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    padding: '20px',
    backgroundColor: '#f5f5f5',
    borderRadius: '8px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
    minWidth: '300px',
    maxWidth: '400px'
  },
  title: {
    margin: '0 0 20px 0',
    fontSize: '18px',
    fontWeight: 'bold',
    color: '#333'
  },
  allCombinedSection: {
    padding: '15px',
    backgroundColor: 'white',
    borderRadius: '6px',
    border: '2px solid #2196F3',
    marginBottom: '15px'
  },
  allCombinedLabel: {
    display: 'flex',
    alignItems: 'center',
    cursor: 'pointer',
    fontSize: '15px',
    fontWeight: '600'
  },
  allCombinedText: {
    color: '#2196F3',
    fontWeight: 'bold'
  },
  helpText: {
    margin: '8px 0 0 28px',
    fontSize: '12px',
    color: '#666',
    fontStyle: 'italic'
  },
  divider: {
    height: '1px',
    backgroundColor: '#ddd',
    margin: '15px 0'
  },
  filterSection: {
    marginBottom: '15px'
  },
  sectionHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '12px'
  },
  sectionTitle: {
    margin: 0,
    fontSize: '15px',
    fontWeight: '600',
    color: '#333'
  },
  showAllButton: {
    padding: '4px 12px',
    backgroundColor: '#f0f0f0',
    border: '1px solid #ccc',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '12px',
    fontWeight: '500',
    color: '#555',
    transition: 'all 0.2s'
  },
  filterOptions: {
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
    padding: '10px',
    backgroundColor: 'white',
    borderRadius: '6px',
    border: '1px solid #e0e0e0'
  },
  filterLabel: {
    display: 'flex',
    alignItems: 'center',
    cursor: 'pointer',
    fontSize: '14px',
    padding: '5px',
    borderRadius: '4px',
    transition: 'background-color 0.2s'
  },
  checkbox: {
    marginRight: '10px',
    cursor: 'pointer',
    width: '18px',
    height: '18px'
  },
  colorIndicator: {
    width: '24px',
    height: '24px',
    borderRadius: '4px',
    marginRight: '10px',
    flexShrink: 0,
    boxShadow: '0 1px 3px rgba(0,0,0,0.2)'
  },
  filterText: {
    color: '#333',
    flex: 1
  },
  quickActions: {
    marginTop: '20px',
    display: 'flex',
    gap: '10px'
  },
  clearButton: {
    flex: 1,
    padding: '10px',
    backgroundColor: '#f44336',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: '500',
    transition: 'background-color 0.2s'
  },
  summary: {
    marginTop: '15px',
    padding: '12px',
    backgroundColor: 'white',
    borderRadius: '6px',
    border: '1px solid #e0e0e0'
  },
  summaryText: {
    margin: 0,
    fontSize: '13px',
    color: '#555',
    fontWeight: '500'
  }
};

export default SeverityFilter;
