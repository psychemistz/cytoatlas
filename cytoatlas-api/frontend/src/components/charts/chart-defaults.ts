import type { Config, Layout, DataTitle } from 'plotly.js-dist-min';

/** Wrap a string as a Plotly DataTitle object */
export function title(text: string | undefined): Partial<DataTitle> | undefined {
  return text ? { text } : undefined;
}

export const PLOTLY_FONT = {
  family: 'Inter, system-ui, sans-serif',
  size: 12,
};

export const PLOTLY_LAYOUT: Partial<Layout> = {
  font: PLOTLY_FONT,
  margin: { l: 80, r: 40, t: 40, b: 80 },
  paper_bgcolor: 'white',
  plot_bgcolor: 'white',
  hovermode: 'closest' as const,
};

export const PLOTLY_CONFIG: Partial<Config> = {
  responsive: true,
  displayModeBar: true,
  modeBarButtonsToRemove: ['lasso2d', 'select2d'] as never[],
  toImageButtonOptions: {
    format: 'png',
    filename: 'cytoatlas_chart',
    scale: 2,
  },
};

export const HEATMAP_COLORSCALE: [number, string][] = [
  [0, '#2166ac'],
  [0.5, '#f7f7f7'],
  [1, '#b2182b'],
];

export const COLORS = {
  primary: '#2563eb',
  primaryDark: '#1d4ed8',
  red: '#ef4444',
  green: '#10b981',
  amber: '#f59e0b',
  purple: '#8b5cf6',
  gray: '#94a3b8',
  darkSlate: '#1e293b',
  gridline: '#f1f5f9',
  zeroline: '#e2e8f0',
} as const;
