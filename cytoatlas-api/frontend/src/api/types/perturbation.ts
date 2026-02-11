export interface PerturbationSummary {
  parse10m: {
    total_cells: number;
    total_cytokines: number;
    total_cell_types: number;
  };
  tahoe: {
    total_cells: number;
    total_drugs: number;
    total_cell_lines: number;
    total_plates: number;
  };
}

export interface CytokineResponse {
  cytokine: string;
  cell_type: string;
  activity_diff: number;
}

export interface GroundTruth {
  cytokine: string;
  cell_type: string;
  predicted_activity: number;
  actual_response: number;
  is_self_signature: boolean;
}

export interface DrugSensitivity {
  drug: string;
  cell_line: string;
  activity_diff: number;
}

export interface DoseResponse {
  drug: string;
  cell_line: string;
  dose: number;
  activity_diff: number;
}

export interface PathwayActivation {
  drug: string;
  pathway: string;
  activation_score: number;
}
