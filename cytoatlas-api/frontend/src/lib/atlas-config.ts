export interface TabConfig {
  id: string;
  label: string;
  icon?: string;
}

const CIMA_TABS: TabConfig[] = [
  { id: 'overview', label: 'Overview', icon: '\u{1F3E0}' },
  { id: 'celltypes', label: 'Cell Types', icon: '\u{1F52C}' },
  { id: 'age-bmi', label: 'Age & BMI', icon: '\u{1F4C8}' },
  { id: 'age-bmi-stratified', label: 'Age/BMI Stratified', icon: '\u{1F4CA}' },
  { id: 'biochemistry', label: 'Biochemistry', icon: '\u{1F9EA}' },
  { id: 'biochem-scatter', label: 'Biochem Scatter', icon: '\u{1F4C9}' },
  { id: 'metabolites', label: 'Metabolites', icon: '\u2697' },
  { id: 'differential', label: 'Differential', icon: '\u{1F4D1}' },
  { id: 'multiomics', label: 'Multi-omics', icon: '\u{1F52C}' },
  { id: 'population', label: 'Population', icon: '\u{1F465}' },
  { id: 'eqtl', label: 'eQTL Browser', icon: '\u{1F9EC}' },
];

const INFLAMMATION_TABS: TabConfig[] = [
  { id: 'overview', label: 'Overview', icon: '\u{1F3E0}' },
  { id: 'celltypes', label: 'Cell Types', icon: '\u{1F52C}' },
  { id: 'age-bmi', label: 'Age & BMI', icon: '\u{1F4C8}' },
  { id: 'age-bmi-stratified', label: 'Age/BMI Stratified', icon: '\u{1F4CA}' },
  { id: 'sankey', label: 'Disease Flow', icon: '\u{1F504}' },
  { id: 'disease', label: 'Disease Activity', icon: '\u{1FA7A}' },
  { id: 'severity', label: 'Severity', icon: '\u{1F4C8}' },
  { id: 'differential', label: 'Differential', icon: '\u{1F4D1}' },
  { id: 'treatment', label: 'Treatment Response', icon: '\u{1F489}' },
  { id: 'validation', label: 'Cohort Validation', icon: '\u2705' },
  { id: 'drivers', label: 'Cell Drivers', icon: '\u{1F52E}' },
];

const SCATLAS_TABS: TabConfig[] = [
  { id: 'overview', label: 'Overview', icon: '\u{1F3E0}' },
  { id: 'celltypes', label: 'Cell Types', icon: '\u{1F52C}' },
  { id: 'tissue-atlas', label: 'Tissue Atlas', icon: '\u{1F495}' },
  { id: 'differential-analysis', label: 'Differential', icon: '\u{1F4C9}' },
  { id: 'immune-infiltration', label: 'Immune Infiltration', icon: '\u{1F52C}' },
  { id: 'tcell-state', label: 'T Cell State', icon: '\u{1F9EC}' },
  { id: 'exhaustion', label: 'Exhaustion Diff', icon: '\u{1F622}' },
  { id: 'caf', label: 'CAF Types', icon: '\u{1F52E}' },
];

export const ATLAS_TABS: Record<string, TabConfig[]> = {
  cima: CIMA_TABS,
  inflammation: INFLAMMATION_TABS,
  scatlas: SCATLAS_TABS,
};

export function getAtlasTabs(atlasName: string): TabConfig[] {
  return ATLAS_TABS[atlasName] ?? [];
}
