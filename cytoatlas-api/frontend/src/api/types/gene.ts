/** Backend GeneOverview schema (from /gene/{signature}) */
export interface GeneOverviewResponse {
  signature: string;
  signature_type: string;
  description?: string;
  atlases: string[];
  summary_stats: {
    n_atlases: number;
    n_cell_types: number;
    n_tissues: number;
    n_diseases: number;
    n_correlations: number;
    top_cell_type?: string;
    top_tissue?: string;
    has_expression: boolean;
  };
}

/** Frontend-friendly overview shape used by components */
export interface GeneOverview {
  gene: string;
  description?: string;
  has_expression: boolean;
  has_cytosig: boolean;
  has_secact: boolean;
  cell_type_count?: number;
  atlas_count?: number;
  atlases: string[];
  top_cell_types?: { cell_type: string; atlas: string; mean_activity: number }[];
}

/** Backend GeneExpressionResponse */
export interface GeneExpressionResponse {
  gene: string;
  data: GeneExpressionData[];
  atlases: string[];
  n_cell_types: number;
  max_expression: number;
  top_cell_type?: string;
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
  std_activity?: number;
  n_cells?: number;
  n_samples?: number;
}

/** Backend GeneDiseaseActivityResponse (wrapped) */
export interface GeneDiseaseActivityResponse {
  signature: string;
  signature_type: string;
  data: GeneDiseaseItem[];
  disease_groups: string[];
  n_diseases: number;
  n_significant: number;
}

export interface GeneDiseaseItem {
  disease: string;
  disease_group: string;
  activity_diff: number;
  pvalue: number;
  qvalue?: number;
  mean_disease: number;
  mean_healthy: number;
  neg_log10_pval?: number;
  is_significant?: boolean;
}

/** Frontend-friendly disease shape used by components */
export interface GeneDiseaseData {
  disease: string;
  cohort: string;
  activity_diff: number;
  p_value: number;
  fdr?: number;
  mean_disease: number;
  mean_healthy: number;
}

/** Backend GeneCorrelations (categorized wrapper) */
export interface GeneCorrelationsResponse {
  signature: string;
  signature_type: string;
  age: GeneCorrelationItem[];
  bmi: GeneCorrelationItem[];
  biochemistry: GeneCorrelationItem[];
  metabolites: GeneCorrelationItem[];
}

export interface GeneCorrelationItem {
  variable: string;
  rho: number;
  pvalue: number;
  qvalue?: number;
  n_samples?: number;
  cell_type?: string;
  category?: string;
}

/** Frontend-friendly correlation shape used by components */
export interface GeneCorrelation {
  variable: string;
  type: string;
  rho: number;
  p_value: number;
  q_value?: number;
  cell_type?: string;
  n: number;
}
