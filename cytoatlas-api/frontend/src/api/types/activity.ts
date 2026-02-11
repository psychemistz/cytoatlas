export interface ActivityData {
  cell_type: string;
  signature: string;
  mean_activity: number;
  std_activity?: number;
  n_cells?: number;
  n_samples?: number;
}

export interface CorrelationData {
  signature: string;
  cell_type?: string;
  variable?: string;
  rho: number;
  p_value: number;
  q_value?: number;
  n: number;
}

export interface DifferentialData {
  signature: string;
  cell_type?: string;
  activity_diff: number;
  p_value: number;
  fdr: number;
  group_a?: string;
  group_b?: string;
  mean_a?: number;
  mean_b?: number;
}

export interface HeatmapData {
  z: number[][];
  x: string[];
  y: string[];
}

export interface PopulationStratification {
  signature: string;
  variable: string;
  effect_size: number;
  p_value: number;
  fdr: number;
}

export interface TreatmentPrediction {
  treatment: string;
  auroc: number;
  fpr: number[];
  tpr: number[];
  features: { name: string; importance: number }[];
}

export interface BiochemCorrelation {
  signature: string;
  marker: string;
  rho: number;
  p_value: number;
  q_value?: number;
  n: number;
}

export interface MetaboliteCorrelation {
  signature: string;
  metabolite: string;
  category: string;
  rho: number;
  p_value: number;
  q_value?: number;
}

export interface OrganSignature {
  organ: string;
  signature: string;
  mean_activity: number;
  std_activity?: number;
  n_cells?: number;
}

export interface ExhaustionDiff {
  signature: string;
  cancer_type: string;
  activity_diff: number;
  p_value: number;
  fdr: number;
}
