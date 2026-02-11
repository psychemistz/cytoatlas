import '@testing-library/jest-dom/vitest';
import { createElement } from 'react';

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: (key: string) => store[key] ?? null,
    setItem: (key: string, value: string) => { store[key] = value; },
    removeItem: (key: string) => { delete store[key]; },
    clear: () => { store = {}; },
    get length() { return Object.keys(store).length; },
    key: (index: number) => Object.keys(store)[index] ?? null,
  };
})();
Object.defineProperty(globalThis, 'localStorage', { value: localStorageMock });

// Mock ResizeObserver
globalThis.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
};

// Mock react-plotly.js (heavy Plotly bundle)
vi.mock('react-plotly.js', () => ({
  default: (props: Record<string, unknown>) => {
    const { data, layout, ...rest } = props;
    return createElement('div', {
      'data-testid': 'plotly-chart',
      'data-traces': JSON.stringify(data),
      'data-layout': JSON.stringify(layout),
      ...rest,
    });
  },
}));
