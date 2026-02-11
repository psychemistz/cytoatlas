export interface Atlas {
  name: string;
  display_name: string;
  description: string;
  n_cells: number;
  n_samples: number;
  n_cell_types: number;
  atlas_type: string;
  status: string;
  has_cytosig: boolean;
  has_secact: boolean;
  species?: string;
  version?: string;
  features?: string[];
  created_at?: string;
  updated_at?: string;
  /** Derived fields used by frontend components */
  source_type?: string;
  validation_grade?: string;
}

export interface AtlasListResponse {
  atlases: Atlas[];
  total: number;
}

export interface AtlasSummary {
  name: string;
  display_name: string;
  n_cells: number;
  n_samples: number;
  n_cell_types: number;
  n_cytosig_signatures: number;
  n_secact_signatures: number;
  cell_types: string[];
  diseases?: string[];
  tissues?: string[];
}

export interface CellType {
  name: string;
  count: number;
  fraction: number;
}

export interface Signature {
  name: string;
  type: 'CytoSig' | 'SecAct';
  description?: string;
}
