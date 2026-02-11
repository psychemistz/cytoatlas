export interface SpatialSummary {
  total_cells: number;
  total_datasets: number;
  technologies: number;
  tissues: number;
}

export interface SpatialDataset {
  dataset_id: string;
  filename?: string;
  technology: string;
  tissue: string;
  n_cells: number;
  n_genes: number;
}

export interface TissueActivity {
  tissue: string;
  signature: string;
  mean_activity: number;
}

export interface TechnologyComparison {
  technology_1: string;
  technology_2: string;
  tissue: string;
  activity_tech1: number;
  activity_tech2: number;
}

export interface GeneCoverage {
  technology: string;
  cytosig_coverage: number;
  secact_coverage: number;
}

export interface SpatialCoordinate {
  x_coord: number;
  y_coord: number;
  activity?: number;
  cell_type?: string;
}
