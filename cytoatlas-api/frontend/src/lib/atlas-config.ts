export interface TabConfig {
  id: string;
  label: string;
  icon?: string;
}

const CIMA_TABS: TabConfig[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'celltypes', label: 'Cell Types' },
  { id: 'age-bmi', label: 'Age & BMI' },
  { id: 'age-bmi-stratified', label: 'Age/BMI Stratified' },
  { id: 'biochemistry', label: 'Biochemistry' },
  { id: 'biochem-scatter', label: 'Biochem Scatter' },
  { id: 'metabolites', label: 'Metabolites' },
  { id: 'differential', label: 'Differential' },
  { id: 'multiomics', label: 'Multi-omics' },
  { id: 'population', label: 'Population' },
  { id: 'eqtl', label: 'eQTL Browser' },
];

const INFLAMMATION_TABS: TabConfig[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'celltypes', label: 'Cell Types' },
  { id: 'age-bmi', label: 'Age & BMI' },
  { id: 'age-bmi-stratified', label: 'Age/BMI Stratified' },
  { id: 'sankey', label: 'Disease Flow' },
  { id: 'disease', label: 'Disease' },
  { id: 'severity', label: 'Severity' },
  { id: 'differential', label: 'Differential' },
  { id: 'treatment', label: 'Treatment Response' },
  { id: 'validation', label: 'Cohort Validation' },
  { id: 'drivers', label: 'Cell Drivers' },
];

const SCATLAS_TABS: TabConfig[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'celltypes', label: 'Cell Types' },
  { id: 'tissue-atlas', label: 'Tissue Atlas' },
  { id: 'differential-analysis', label: 'Differential' },
  { id: 'immune-infiltration', label: 'Immune Infiltration' },
  { id: 'tcell-state', label: 'T Cell State' },
  { id: 'exhaustion', label: 'Exhaustion Diff' },
  { id: 'caf', label: 'CAF Types' },
];

export const ATLAS_TABS: Record<string, TabConfig[]> = {
  cima: CIMA_TABS,
  inflammation: INFLAMMATION_TABS,
  scatlas: SCATLAS_TABS,
};

export function getAtlasTabs(atlasName: string): TabConfig[] {
  return ATLAS_TABS[atlasName] ?? [];
}
