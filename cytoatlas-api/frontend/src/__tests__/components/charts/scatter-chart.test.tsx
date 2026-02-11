import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ScatterChart } from '@/components/charts/scatter-chart';

describe('ScatterChart', () => {
  const defaultProps = {
    x: [1, 2, 3, 4, 5],
    y: [2, 4, 6, 8, 10],
  };

  it('renders a plotly chart', () => {
    render(<ScatterChart {...defaultProps} />);
    expect(screen.getByTestId('plotly-chart')).toBeInTheDocument();
  });

  it('creates a scatter trace', () => {
    render(<ScatterChart {...defaultProps} />);
    const el = screen.getByTestId('plotly-chart');
    const traces = JSON.parse(el.getAttribute('data-traces') || '[]');
    expect(traces[0].type).toBe('scatter');
    expect(traces[0].mode).toBe('markers');
    expect(traces[0].x).toEqual([1, 2, 3, 4, 5]);
    expect(traces[0].y).toEqual([2, 4, 6, 8, 10]);
  });

  it('adds trend line when showTrendLine is true', () => {
    render(<ScatterChart {...defaultProps} showTrendLine />);
    const el = screen.getByTestId('plotly-chart');
    const traces = JSON.parse(el.getAttribute('data-traces') || '[]');
    expect(traces).toHaveLength(2);
    expect(traces[1].mode).toBe('lines');
  });

  it('does not add trend line with fewer than 2 points', () => {
    render(<ScatterChart x={[1]} y={[2]} showTrendLine />);
    const el = screen.getByTestId('plotly-chart');
    const traces = JSON.parse(el.getAttribute('data-traces') || '[]');
    expect(traces).toHaveLength(1);
  });

  it('sets axis titles in layout', () => {
    render(<ScatterChart {...defaultProps} xTitle="X Axis" yTitle="Y Axis" />);
    const el = screen.getByTestId('plotly-chart');
    const layout = JSON.parse(el.getAttribute('data-layout') || '{}');
    expect(layout.xaxis.title).toEqual({ text: 'X Axis' });
    expect(layout.yaxis.title).toEqual({ text: 'Y Axis' });
  });

  it('includes stats annotation when stats provided', () => {
    render(<ScatterChart {...defaultProps} stats={{ rho: 0.95, p: 0.001 }} />);
    const el = screen.getByTestId('plotly-chart');
    const layout = JSON.parse(el.getAttribute('data-layout') || '{}');
    expect(layout.annotations).toHaveLength(1);
    expect(layout.annotations[0].text).toContain('0.950');
  });

  it('sets chart height', () => {
    render(<ScatterChart {...defaultProps} height={300} />);
    const el = screen.getByTestId('plotly-chart');
    const layout = JSON.parse(el.getAttribute('data-layout') || '{}');
    expect(layout.height).toBe(300);
  });
});
