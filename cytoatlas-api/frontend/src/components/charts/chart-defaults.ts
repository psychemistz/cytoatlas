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
  plot_bgcolor: '#fafafa',
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

/** Secondary colorscale for correlation heatmaps (biochemistry, metabolites, severity) */
export const CORRELATION_COLORSCALE: [number, string][] = [
  [0, '#a8d4e6'],
  [0.5, '#f5f5f5'],
  [1, '#f4a6a6'],
];

export const COLORS = {
  primary: '#1f77b4',
  primaryDark: '#1d4ed8',
  red: '#d62728',
  green: '#2ca02c',
  amber: '#ff7f0e',
  purple: '#9467bd',
  gray: '#ccc',
  darkSlate: '#1e293b',
  gridline: '#f1f5f9',
  zeroline: '#e2e8f0',
} as const;
