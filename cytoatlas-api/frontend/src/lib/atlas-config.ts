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
  { id: 'eqtl', label: 'eQTL' },
];

const INFLAMMATION_TABS: TabConfig[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'celltypes', label: 'Cell Types' },
  { id: 'age-bmi', label: 'Age & BMI' },
  { id: 'age-bmi-stratified', label: 'Age/BMI Stratified' },
  { id: 'sankey', label: 'Disease Flow' },
  { id: 'disease', label: 'Disease Activity' },
  { id: 'severity', label: 'Severity' },
  { id: 'differential', label: 'Differential' },
  { id: 'treatment', label: 'Treatment' },
  { id: 'validation', label: 'Validation' },
  { id: 'drivers', label: 'Drivers' },
];

const SCATLAS_TABS: TabConfig[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'celltypes', label: 'Cell Types' },
  { id: 'tissue-atlas', label: 'Tissue Atlas' },
  { id: 'differential', label: 'Differential' },
  { id: 'immune-infiltration', label: 'Immune Infiltration' },
  { id: 'tcell-state', label: 'T Cell States' },
  { id: 'exhaustion', label: 'Exhaustion' },
  { id: 'caf', label: 'CAF Subtypes' },
];

export const ATLAS_TABS: Record<string, TabConfig[]> = {
  cima: CIMA_TABS,
  inflammation: INFLAMMATION_TABS,
  scatlas: SCATLAS_TABS,
};

export function getAtlasTabs(atlasName: string): TabConfig[] {
  return ATLAS_TABS[atlasName] ?? [];
}
