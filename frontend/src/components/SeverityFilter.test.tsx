import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import SeverityFilter, { SeverityFilterConfig } from './SeverityFilter';

describe('SeverityFilter', () => {
  const defaultConfig: SeverityFilterConfig = {
    pigmentation: { low: true, medium: true, high: true },
    wrinkles: { micro: true, regular: true },
    allCombined: true
  };

  const mockOnFilterChange = jest.fn();

  beforeEach(() => {
    mockOnFilterChange.mockClear();
  });

  describe('Rendering', () => {
    it('should render the component with title', () => {
      render(
        <SeverityFilter
          filterConfig={defaultConfig}
          onFilterChange={mockOnFilterChange}
        />
      );

      expect(screen.getByText('Severity Filters')).toBeInTheDocument();
    });

    it('should render all pigmentation severity options', () => {
      render(
        <SeverityFilter
          filterConfig={defaultConfig}
          onFilterChange={mockOnFilterChange}
        />
      );

      expect(screen.getByText('Low Severity')).toBeInTheDocument();
      expect(screen.getByText('Medium Severity')).toBeInTheDocument();
      expect(screen.getByText('High Severity')).toBeInTheDocument();
    });

    it('should render all wrinkle type options', () => {
      render(
        <SeverityFilter
          filterConfig={defaultConfig}
          onFilterChange={mockOnFilterChange}
        />
      );

      expect(screen.getByText(/Micro-wrinkles/)).toBeInTheDocument();
      expect(screen.getByText('Regular Wrinkles')).toBeInTheDocument();
    });

    it('should render "Show All Combined" option', () => {
      render(
        <SeverityFilter
          filterConfig={defaultConfig}
          onFilterChange={mockOnFilterChange}
        />
      );

      expect(screen.getByText('Show All Combined')).toBeInTheDocument();
    });

    it('should render Clear All button', () => {
      render(
        <SeverityFilter
          filterConfig={defaultConfig}
          onFilterChange={mockOnFilterChange}
        />
      );

      expect(screen.getByText('Clear All')).toBeInTheDocument();
    });
  });

  describe('All Combined Toggle', () => {
    it('should call onFilterChange when "Show All Combined" is toggled on', () => {
      const config: SeverityFilterConfig = {
        pigmentation: { low: false, medium: false, high: false },
        wrinkles: { micro: false, regular: false },
        allCombined: false
      };

      render(
        <SeverityFilter
          filterConfig={config}
          onFilterChange={mockOnFilterChange}
        />
      );

      const checkbox = screen.getByRole('checkbox', { name: /Show All Combined/i });
      fireEvent.click(checkbox);

      expect(mockOnFilterChange).toHaveBeenCalledWith({
        pigmentation: { low: true, medium: true, high: true },
        wrinkles: { micro: true, regular: true },
        allCombined: true
      });
    });

    it('should disable all filters when "Show All Combined" is toggled off', () => {
      render(
        <SeverityFilter
          filterConfig={defaultConfig}
          onFilterChange={mockOnFilterChange}
        />
      );

      const checkbox = screen.getByRole('checkbox', { name: /Show All Combined/i });
      fireEvent.click(checkbox);

      expect(mockOnFilterChange).toHaveBeenCalledWith({
        pigmentation: { low: false, medium: false, high: false },
        wrinkles: { micro: false, regular: false },
        allCombined: false
      });
    });

    it('should disable individual filter checkboxes when allCombined is true', () => {
      render(
        <SeverityFilter
          filterConfig={defaultConfig}
          onFilterChange={mockOnFilterChange}
        />
      );

      const lowCheckbox = screen.getByRole('checkbox', { name: /Low Severity/i });
      expect(lowCheckbox).toBeDisabled();
    });
  });

  describe('Pigmentation Filters', () => {
    it('should toggle low severity pigmentation filter', () => {
      const config: SeverityFilterConfig = {
        pigmentation: { low: true, medium: false, high: false },
        wrinkles: { micro: false, regular: false },
        allCombined: false
      };

      render(
        <SeverityFilter
          filterConfig={config}
          onFilterChange={mockOnFilterChange}
        />
      );

      const checkbox = screen.getByRole('checkbox', { name: /Low Severity/i });
      fireEvent.click(checkbox);

      expect(mockOnFilterChange).toHaveBeenCalledWith({
        pigmentation: { low: false, medium: false, high: false },
        wrinkles: { micro: false, regular: false },
        allCombined: false
      });
    });

    it('should toggle medium severity pigmentation filter', () => {
      const config: SeverityFilterConfig = {
        pigmentation: { low: false, medium: false, high: false },
        wrinkles: { micro: false, regular: false },
        allCombined: false
      };

      render(
        <SeverityFilter
          filterConfig={config}
          onFilterChange={mockOnFilterChange}
        />
      );

      const checkbox = screen.getByRole('checkbox', { name: /Medium Severity/i });
      fireEvent.click(checkbox);

      expect(mockOnFilterChange).toHaveBeenCalledWith({
        pigmentation: { low: false, medium: true, high: false },
        wrinkles: { micro: false, regular: false },
        allCombined: false
      });
    });

    it('should toggle high severity pigmentation filter', () => {
      const config: SeverityFilterConfig = {
        pigmentation: { low: false, medium: false, high: false },
        wrinkles: { micro: false, regular: false },
        allCombined: false
      };

      render(
        <SeverityFilter
          filterConfig={config}
          onFilterChange={mockOnFilterChange}
        />
      );

      const checkbox = screen.getByRole('checkbox', { name: /High Severity/i });
      fireEvent.click(checkbox);

      expect(mockOnFilterChange).toHaveBeenCalledWith({
        pigmentation: { low: false, medium: false, high: true },
        wrinkles: { micro: false, regular: false },
        allCombined: false
      });
    });

    it('should enable all pigmentation filters when "All" button is clicked', () => {
      const config: SeverityFilterConfig = {
        pigmentation: { low: false, medium: false, high: false },
        wrinkles: { micro: false, regular: false },
        allCombined: false
      };

      render(
        <SeverityFilter
          filterConfig={config}
          onFilterChange={mockOnFilterChange}
        />
      );

      const buttons = screen.getAllByText('All');
      const pigmentationAllButton = buttons[0]; // First "All" button is for pigmentation
      fireEvent.click(pigmentationAllButton);

      expect(mockOnFilterChange).toHaveBeenCalledWith({
        pigmentation: { low: true, medium: true, high: true },
        wrinkles: { micro: false, regular: false },
        allCombined: false
      });
    });
  });

  describe('Wrinkle Filters', () => {
    it('should toggle micro wrinkle filter', () => {
      const config: SeverityFilterConfig = {
        pigmentation: { low: false, medium: false, high: false },
        wrinkles: { micro: false, regular: false },
        allCombined: false
      };

      render(
        <SeverityFilter
          filterConfig={config}
          onFilterChange={mockOnFilterChange}
        />
      );

      const checkbox = screen.getByRole('checkbox', { name: /Micro-wrinkles/i });
      fireEvent.click(checkbox);

      expect(mockOnFilterChange).toHaveBeenCalledWith({
        pigmentation: { low: false, medium: false, high: false },
        wrinkles: { micro: true, regular: false },
        allCombined: false
      });
    });

    it('should toggle regular wrinkle filter', () => {
      const config: SeverityFilterConfig = {
        pigmentation: { low: false, medium: false, high: false },
        wrinkles: { micro: false, regular: false },
        allCombined: false
      };

      render(
        <SeverityFilter
          filterConfig={config}
          onFilterChange={mockOnFilterChange}
        />
      );

      const checkbox = screen.getByRole('checkbox', { name: /Regular Wrinkles/i });
      fireEvent.click(checkbox);

      expect(mockOnFilterChange).toHaveBeenCalledWith({
        pigmentation: { low: false, medium: false, high: false },
        wrinkles: { micro: false, regular: true },
        allCombined: false
      });
    });

    it('should enable all wrinkle filters when "All" button is clicked', () => {
      const config: SeverityFilterConfig = {
        pigmentation: { low: false, medium: false, high: false },
        wrinkles: { micro: false, regular: false },
        allCombined: false
      };

      render(
        <SeverityFilter
          filterConfig={config}
          onFilterChange={mockOnFilterChange}
        />
      );

      const buttons = screen.getAllByText('All');
      const wrinkleAllButton = buttons[1]; // Second "All" button is for wrinkles
      fireEvent.click(wrinkleAllButton);

      expect(mockOnFilterChange).toHaveBeenCalledWith({
        pigmentation: { low: false, medium: false, high: false },
        wrinkles: { micro: true, regular: true },
        allCombined: false
      });
    });
  });

  describe('Multi-select Filtering', () => {
    it('should support multiple pigmentation severity levels selected', () => {
      const config: SeverityFilterConfig = {
        pigmentation: { low: true, medium: true, high: false },
        wrinkles: { micro: false, regular: false },
        allCombined: false
      };

      render(
        <SeverityFilter
          filterConfig={config}
          onFilterChange={mockOnFilterChange}
        />
      );

      const lowCheckbox = screen.getByRole('checkbox', { name: /Low Severity/i });
      const mediumCheckbox = screen.getByRole('checkbox', { name: /Medium Severity/i });

      expect(lowCheckbox).toBeChecked();
      expect(mediumCheckbox).toBeChecked();
    });

    it('should support multiple wrinkle types selected', () => {
      const config: SeverityFilterConfig = {
        pigmentation: { low: false, medium: false, high: false },
        wrinkles: { micro: true, regular: true },
        allCombined: false
      };

      render(
        <SeverityFilter
          filterConfig={config}
          onFilterChange={mockOnFilterChange}
        />
      );

      const microCheckbox = screen.getByRole('checkbox', { name: /Micro-wrinkles/i });
      const regularCheckbox = screen.getByRole('checkbox', { name: /Regular Wrinkles/i });

      expect(microCheckbox).toBeChecked();
      expect(regularCheckbox).toBeChecked();
    });

    it('should support mixed pigmentation and wrinkle filters', () => {
      const config: SeverityFilterConfig = {
        pigmentation: { low: true, medium: false, high: true },
        wrinkles: { micro: true, regular: false },
        allCombined: false
      };

      render(
        <SeverityFilter
          filterConfig={config}
          onFilterChange={mockOnFilterChange}
        />
      );

      expect(screen.getByRole('checkbox', { name: /Low Severity/i })).toBeChecked();
      expect(screen.getByRole('checkbox', { name: /High Severity/i })).toBeChecked();
      expect(screen.getByRole('checkbox', { name: /Micro-wrinkles/i })).toBeChecked();
    });
  });

  describe('Clear All', () => {
    it('should clear all filters when Clear All button is clicked', () => {
      render(
        <SeverityFilter
          filterConfig={defaultConfig}
          onFilterChange={mockOnFilterChange}
        />
      );

      const clearButton = screen.getByText('Clear All');
      fireEvent.click(clearButton);

      expect(mockOnFilterChange).toHaveBeenCalledWith({
        pigmentation: { low: false, medium: false, high: false },
        wrinkles: { micro: false, regular: false },
        allCombined: false
      });
    });
  });

  describe('Summary Text', () => {
    it('should display "Showing all anomalies" when allCombined is true', () => {
      render(
        <SeverityFilter
          filterConfig={defaultConfig}
          onFilterChange={mockOnFilterChange}
        />
      );

      expect(screen.getByText('Showing all anomalies')).toBeInTheDocument();
    });

    it('should display active pigmentation filters in summary', () => {
      const config: SeverityFilterConfig = {
        pigmentation: { low: true, medium: true, high: false },
        wrinkles: { micro: false, regular: false },
        allCombined: false
      };

      render(
        <SeverityFilter
          filterConfig={config}
          onFilterChange={mockOnFilterChange}
        />
      );

      expect(screen.getByText(/Active: Pigmentation: Low, Medium/)).toBeInTheDocument();
    });

    it('should display active wrinkle filters in summary', () => {
      const config: SeverityFilterConfig = {
        pigmentation: { low: false, medium: false, high: false },
        wrinkles: { micro: true, regular: true },
        allCombined: false
      };

      render(
        <SeverityFilter
          filterConfig={config}
          onFilterChange={mockOnFilterChange}
        />
      );

      expect(screen.getByText(/Active: Wrinkles: Micro, Regular/)).toBeInTheDocument();
    });

    it('should display "No filters active" when all filters are disabled', () => {
      const config: SeverityFilterConfig = {
        pigmentation: { low: false, medium: false, high: false },
        wrinkles: { micro: false, regular: false },
        allCombined: false
      };

      render(
        <SeverityFilter
          filterConfig={config}
          onFilterChange={mockOnFilterChange}
        />
      );

      expect(screen.getByText('No filters active (all anomalies hidden)')).toBeInTheDocument();
    });
  });

  describe('Requirements Validation', () => {
    it('should provide toggle controls for each severity level (Req 5.1)', () => {
      render(
        <SeverityFilter
          filterConfig={defaultConfig}
          onFilterChange={mockOnFilterChange}
        />
      );

      // Verify all severity level controls exist
      expect(screen.getByRole('checkbox', { name: /Low Severity/i })).toBeInTheDocument();
      expect(screen.getByRole('checkbox', { name: /Medium Severity/i })).toBeInTheDocument();
      expect(screen.getByRole('checkbox', { name: /High Severity/i })).toBeInTheDocument();
    });

    it('should display only selected severity levels (Req 5.2)', () => {
      const config: SeverityFilterConfig = {
        pigmentation: { low: true, medium: false, high: false },
        wrinkles: { micro: false, regular: false },
        allCombined: false
      };

      render(
        <SeverityFilter
          filterConfig={config}
          onFilterChange={mockOnFilterChange}
        />
      );

      // Only low severity should be checked
      expect(screen.getByRole('checkbox', { name: /Low Severity/i })).toBeChecked();
      expect(screen.getByRole('checkbox', { name: /Medium Severity/i })).not.toBeChecked();
      expect(screen.getByRole('checkbox', { name: /High Severity/i })).not.toBeChecked();
    });

    it('should hide deselected severity levels (Req 5.3)', () => {
      const config: SeverityFilterConfig = {
        pigmentation: { low: false, medium: true, high: true },
        wrinkles: { micro: false, regular: false },
        allCombined: false
      };

      render(
        <SeverityFilter
          filterConfig={config}
          onFilterChange={mockOnFilterChange}
        />
      );

      // Low severity should not be checked (hidden)
      expect(screen.getByRole('checkbox', { name: /Low Severity/i })).not.toBeChecked();
      expect(screen.getByRole('checkbox', { name: /Medium Severity/i })).toBeChecked();
      expect(screen.getByRole('checkbox', { name: /High Severity/i })).toBeChecked();
    });

    it('should support multi-select filtering (Req 5.4)', () => {
      const config: SeverityFilterConfig = {
        pigmentation: { low: true, medium: true, high: false },
        wrinkles: { micro: true, regular: false },
        allCombined: false
      };

      render(
        <SeverityFilter
          filterConfig={config}
          onFilterChange={mockOnFilterChange}
        />
      );

      // Multiple filters should be checked
      expect(screen.getByRole('checkbox', { name: /Low Severity/i })).toBeChecked();
      expect(screen.getByRole('checkbox', { name: /Medium Severity/i })).toBeChecked();
      expect(screen.getByRole('checkbox', { name: /Micro-wrinkles/i })).toBeChecked();
    });

    it('should display all anomalies when "all combined" is selected (Req 5.5)', () => {
      render(
        <SeverityFilter
          filterConfig={defaultConfig}
          onFilterChange={mockOnFilterChange}
        />
      );

      expect(screen.getByRole('checkbox', { name: /Show All Combined/i })).toBeChecked();
      expect(screen.getByText('Showing all anomalies')).toBeInTheDocument();
    });
  });
});
