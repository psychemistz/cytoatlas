import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { FilterBar, ToggleGroup, SelectFilter, SearchFilter } from '@/components/ui/filter-bar';

describe('FilterBar', () => {
  it('renders children', () => {
    render(
      <FilterBar>
        <span data-testid="child">Filter content</span>
      </FilterBar>,
    );
    expect(screen.getByTestId('child')).toBeInTheDocument();
  });

  it('applies custom className', () => {
    const { container } = render(
      <FilterBar className="extra-class">
        <span>Content</span>
      </FilterBar>,
    );
    expect(container.firstChild).toHaveClass('extra-class');
  });
});

describe('ToggleGroup', () => {
  const options = [
    { value: 'a', label: 'Option A' },
    { value: 'b', label: 'Option B' },
    { value: 'c', label: 'Option C' },
  ];

  it('renders all options', () => {
    render(<ToggleGroup options={options} value="a" onChange={() => {}} />);
    expect(screen.getByText('Option A')).toBeInTheDocument();
    expect(screen.getByText('Option B')).toBeInTheDocument();
    expect(screen.getByText('Option C')).toBeInTheDocument();
  });

  it('renders label when provided', () => {
    render(<ToggleGroup options={options} value="a" onChange={() => {}} label="Filter" />);
    expect(screen.getByText('Filter')).toBeInTheDocument();
  });

  it('calls onChange when option clicked', () => {
    const onChange = vi.fn();
    render(<ToggleGroup options={options} value="a" onChange={onChange} />);
    fireEvent.click(screen.getByText('Option B'));
    expect(onChange).toHaveBeenCalledWith('b');
  });
});

describe('SelectFilter', () => {
  const options = [
    { value: 'all', label: 'All' },
    { value: 'type-a', label: 'Type A' },
    { value: 'type-b', label: 'Type B' },
  ];

  it('renders label and select', () => {
    render(<SelectFilter label="Category" options={options} value="all" onChange={() => {}} />);
    expect(screen.getByText('Category')).toBeInTheDocument();
    expect(screen.getByRole('combobox')).toBeInTheDocument();
  });

  it('calls onChange on selection', () => {
    const onChange = vi.fn();
    render(<SelectFilter label="Category" options={options} value="all" onChange={onChange} />);
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'type-a' } });
    expect(onChange).toHaveBeenCalledWith('type-a');
  });

  it('renders all options', () => {
    render(<SelectFilter label="Test" options={options} value="all" onChange={() => {}} />);
    const optionElements = screen.getAllByRole('option');
    expect(optionElements).toHaveLength(3);
  });
});

describe('SearchFilter', () => {
  it('renders with placeholder', () => {
    render(<SearchFilter value="" onChange={() => {}} placeholder="Search genes..." />);
    expect(screen.getByPlaceholderText('Search genes...')).toBeInTheDocument();
  });

  it('calls onChange on input', () => {
    const onChange = vi.fn();
    render(<SearchFilter value="" onChange={onChange} />);
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'TNF' } });
    expect(onChange).toHaveBeenCalledWith('TNF');
  });

  it('uses default placeholder', () => {
    render(<SearchFilter value="" onChange={() => {}} />);
    expect(screen.getByPlaceholderText('Search...')).toBeInTheDocument();
  });
});
