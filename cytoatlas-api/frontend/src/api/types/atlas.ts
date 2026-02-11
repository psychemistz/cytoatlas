export interface Atlas {
  name: string;
  display_name: string;
  description: string;
  n_cells: number;
  n_samples: number;
  n_cell_types: number;
  source_type: string;
  validation_grade: string;
  status?: string;
  created_at?: string;
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
