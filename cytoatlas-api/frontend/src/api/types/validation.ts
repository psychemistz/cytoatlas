export interface ValidationTarget {
  target: string;
  rho: number;
  p_value?: number;
  gene?: string;
  n?: number;
}

export interface ScatterPoint {
  x: number;
  y: number;
  label?: string;
  cell_type?: string;
  color?: string;
}

export interface ScatterData {
  points: ScatterPoint[];
  rho: number;
  p_value: number;
  pearson_r?: number;
  n?: number;
  target?: string;
}

export interface SummaryBoxplotData {
  targets: string[];
  categories: string[];
  rhos: Record<string, Record<string, number[]>>;
}

export interface MethodComparison {
  methods: string[];
  categories: string[];
  rhos: Record<string, Record<string, number[]>>;
}

export interface BulkRnaseqTarget {
  target: string;
  rho: number;
  p_value?: number;
  n?: number;
  tissue?: string;
}

export interface SingleCellSignature {
  /** Mapped from backend "target" field */
  signature: string;
  gene?: string;
  rho?: number;
  n_total?: number;
  n_expressing?: number;
  expressing_fraction?: number;
}

export interface SingleCellCelltypeStat {
  cell_type: string;
  expression: number;
  activity: number;
  n_cells: number;
  rho?: number;
}
