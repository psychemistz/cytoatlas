export interface GeneOverview {
  gene: string;
  description?: string;
  has_expression: boolean;
  has_cytosig: boolean;
  has_secact: boolean;
  cell_type_count?: number;
  atlas_count?: number;
  top_cell_types?: { cell_type: string; atlas: string; mean_activity: number }[];
}

export interface GeneExpressionData {
  cell_type: string;
  atlas: string;
  mean_expression: number;
  pct_expressed?: number;
  n_cells?: number;
  organ?: string;
}

export interface GeneActivityData {
  cell_type: string;
  atlas: string;
  mean_activity: number;
  sd?: number;
  n_cells?: number;
}

export interface GeneDiseaseData {
  disease: string;
  cohort: string;
  activity_diff: number;
  p_value: number;
  fdr?: number;
  mean_disease: number;
  mean_healthy: number;
}

export interface GeneCorrelation {
  variable: string;
  type: string;
  rho: number;
  p_value: number;
  n: number;
}
