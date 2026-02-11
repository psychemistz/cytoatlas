import Plotly from 'plotly.js-dist-min';

export function exportCSV(data: Record<string, unknown>[], filename = 'cytoatlas_export.csv') {
  if (!data.length) return;
  const headers = Object.keys(data[0]);
  const rows = data.map((row) => headers.map((h) => String(row[h] ?? '')).join(','));
  const csv = [headers.join(','), ...rows].join('\n');
  downloadBlob(csv, filename, 'text/csv');
}

export function exportHeatmapCSV(
  z: number[][],
  x: string[],
  y: string[],
  filename = 'cytoatlas_heatmap.csv',
) {
  const header = ['', ...x].join(',');
  const rows = z.map((row, i) => [y[i], ...row.map((v) => v.toFixed(4))].join(','));
  const csv = [header, ...rows].join('\n');
  downloadBlob(csv, filename, 'text/csv');
}

export async function exportPlotlyPNG(divId: string, filename = 'cytoatlas_chart') {
  await Plotly.downloadImage(divId, {
    format: 'png',
    filename,
    width: 1200,
    height: 800,
  });
}

function downloadBlob(content: string, filename: string, type: string) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
