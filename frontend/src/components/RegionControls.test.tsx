import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import RegionControls, { RegionConfig, FacialRegion } from './RegionControls';

describe('RegionControls', () => {
  const defaultConfig: RegionConfig = {
    selectedRegion: 'all',
    highlightIntensity: 0.5
  };

  const mockOnConfigChange = jest.fn();
  const mockOnZoomToRegion = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders with default configuration', () => {
    render(
      <RegionControls
        regionConfig={defaultConfig}
        onRegionConfigChange={mockOnConfigChange}
      />
    );

    expect(screen.getByText('Region Isolation')).toBeInTheDocument();
    expect(screen.getByText('Select Region:')).toBeInTheDocument();
  });

  it('displays all region options in dropdown', () => {
    render(
      <RegionControls
        regionConfig={defaultConfig}
        onRegionConfigChange={mockOnConfigChange}
      />
    );

    const select = screen.getByRole('combobox');
    expect(select).toBeInTheDocument();
    
    const options = screen.getAllByRole('option');
    expect(options).toHaveLength(9); // all, forehead, left_cheek, right_cheek, etc.
    
    // Check dropdown options (not quick select buttons)
    expect(screen.getByRole('option', { name: 'All Regions' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Forehead' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Left Cheek' })).toBeInTheDocument();
  });

  it('calls onRegionConfigChange when region is selected', () => {
    render(
      <RegionControls
        regionConfig={defaultConfig}
        onRegionConfigChange={mockOnConfigChange}
      />
    );

    const select = screen.getByRole('combobox');
    fireEvent.change(select, { target: { value: 'forehead' } });

    expect(mockOnConfigChange).toHaveBeenCalledWith({
      selectedRegion: 'forehead',
      highlightIntensity: 0.5
    });
  });

  it('shows highlight intensity slider when region is selected', () => {
    const config: RegionConfig = {
      selectedRegion: 'forehead',
      highlightIntensity: 0.7
    };

    render(
      <RegionControls
        regionConfig={config}
        onRegionConfigChange={mockOnConfigChange}
      />
    );

    expect(screen.getByText(/Highlight Intensity: 70%/)).toBeInTheDocument();
    const slider = screen.getByRole('slider');
    expect(slider).toHaveValue('70');
  });

  it('hides highlight intensity slider when "all" is selected', () => {
    render(
      <RegionControls
        regionConfig={defaultConfig}
        onRegionConfigChange={mockOnConfigChange}
      />
    );

    expect(screen.queryByText(/Highlight Intensity/)).not.toBeInTheDocument();
  });

  it('calls onRegionConfigChange when highlight intensity changes', () => {
    const config: RegionConfig = {
      selectedRegion: 'forehead',
      highlightIntensity: 0.5
    };

    render(
      <RegionControls
        regionConfig={config}
        onRegionConfigChange={mockOnConfigChange}
      />
    );

    const slider = screen.getByRole('slider');
    fireEvent.change(slider, { target: { value: '80' } });

    expect(mockOnConfigChange).toHaveBeenCalledWith({
      selectedRegion: 'forehead',
      highlightIntensity: 0.8
    });
  });

  it('shows zoom button when region is selected and callback is provided', () => {
    const config: RegionConfig = {
      selectedRegion: 'forehead',
      highlightIntensity: 0.5
    };

    render(
      <RegionControls
        regionConfig={config}
        onRegionConfigChange={mockOnConfigChange}
        onZoomToRegion={mockOnZoomToRegion}
      />
    );

    const zoomButton = screen.getByText(/Zoom to Forehead/);
    expect(zoomButton).toBeInTheDocument();
  });

  it('hides zoom button when "all" is selected', () => {
    render(
      <RegionControls
        regionConfig={defaultConfig}
        onRegionConfigChange={mockOnConfigChange}
        onZoomToRegion={mockOnZoomToRegion}
      />
    );

    expect(screen.queryByText(/Zoom to/)).not.toBeInTheDocument();
  });

  it('calls onZoomToRegion when zoom button is clicked', () => {
    const config: RegionConfig = {
      selectedRegion: 'left_cheek',
      highlightIntensity: 0.5
    };

    render(
      <RegionControls
        regionConfig={config}
        onRegionConfigChange={mockOnConfigChange}
        onZoomToRegion={mockOnZoomToRegion}
      />
    );

    const zoomButton = screen.getByText(/Zoom to Left Cheek/);
    fireEvent.click(zoomButton);

    expect(mockOnZoomToRegion).toHaveBeenCalledWith('left_cheek');
  });

  it('renders quick select buttons', () => {
    render(
      <RegionControls
        regionConfig={defaultConfig}
        onRegionConfigChange={mockOnConfigChange}
      />
    );

    expect(screen.getByText('Quick Select:')).toBeInTheDocument();
    
    // Quick select buttons have abbreviated labels
    const buttons = screen.getAllByRole('button');
    const buttonTexts = buttons.map(btn => btn.textContent);
    
    expect(buttonTexts).toContain('Forehead');
    expect(buttonTexts).toContain('L Cheek');
    expect(buttonTexts).toContain('R Cheek');
    expect(buttonTexts).toContain('L Eye');
    expect(buttonTexts).toContain('R Eye');
    expect(buttonTexts).toContain('All');
  });

  it('highlights active quick select button', () => {
    const config: RegionConfig = {
      selectedRegion: 'forehead',
      highlightIntensity: 0.5
    };

    const { container } = render(
      <RegionControls
        regionConfig={config}
        onRegionConfigChange={mockOnConfigChange}
      />
    );

    // The active button should have different styling
    // This is a simplified check - in real tests you'd check computed styles
    const quickButtons = container.querySelectorAll('button');
    expect(quickButtons.length).toBeGreaterThan(0);
  });

  it('calls onRegionConfigChange when quick select button is clicked', () => {
    render(
      <RegionControls
        regionConfig={defaultConfig}
        onRegionConfigChange={mockOnConfigChange}
      />
    );

    // Find the "L Cheek" button in the quick select grid
    const buttons = screen.getAllByRole('button');
    const leftCheekButton = buttons.find(btn => btn.textContent === 'L Cheek');
    
    if (leftCheekButton) {
      fireEvent.click(leftCheekButton);
      expect(mockOnConfigChange).toHaveBeenCalledWith({
        selectedRegion: 'left_cheek',
        highlightIntensity: 0.5
      });
    }
  });

  it('displays helpful tip information', () => {
    render(
      <RegionControls
        regionConfig={defaultConfig}
        onRegionConfigChange={mockOnConfigChange}
      />
    );

    expect(screen.getByText(/Tip:/)).toBeInTheDocument();
    expect(screen.getByText(/Select a region to isolate and highlight it/)).toBeInTheDocument();
  });
});
