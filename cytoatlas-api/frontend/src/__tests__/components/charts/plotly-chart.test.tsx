import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { PlotlyChart } from '@/components/charts/plotly-chart';

describe('PlotlyChart', () => {
  it('renders a chart container', () => {
    render(<PlotlyChart data={[{ type: 'scatter', x: [1, 2], y: [3, 4] }]} />);
    expect(screen.getByTestId('plotly-chart')).toBeInTheDocument();
  });

  it('passes data traces to Plot', () => {
    const data = [{ type: 'scatter' as const, x: [1, 2, 3], y: [4, 5, 6] }];
    render(<PlotlyChart data={data} />);
    const el = screen.getByTestId('plotly-chart');
    const traces = JSON.parse(el.getAttribute('data-traces') || '[]');
    expect(traces).toHaveLength(1);
    expect(traces[0].x).toEqual([1, 2, 3]);
  });

  it('merges layout overrides', () => {
    render(
      <PlotlyChart
        data={[]}
        layout={{ title: { text: 'Test' }, height: 300 }}
      />,
    );
    const el = screen.getByTestId('plotly-chart');
    const layout = JSON.parse(el.getAttribute('data-layout') || '{}');
    expect(layout.title).toEqual({ text: 'Test' });
    expect(layout.height).toBe(300);
  });

  it('applies className', () => {
    const { container } = render(
      <PlotlyChart data={[]} className="my-chart" />,
    );
    expect(container.firstChild).toHaveClass('my-chart');
  });
});
