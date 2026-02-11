import type { Atlas } from '@/api/types/atlas';

export const API_BASE = '/api/v1';

export const ATLAS_CONFIGS = {
  cima: {
    name: 'cima',
    displayName: 'CIMA',
    description: 'Chinese Immune Multi-omics Atlas — healthy adult immune profiling with biochemistry and metabolomics',
    color: '#2563eb',
  },
  inflammation: {
    name: 'inflammation',
    displayName: 'Inflammation Atlas',
    description: 'Pan-disease immune profiling across multiple inflammatory conditions with treatment response data',
    color: '#10b981',
  },
  scatlas: {
    name: 'scatlas',
    displayName: 'scAtlas',
    description: 'Human tissue reference atlas with normal organs and pan-cancer immune profiling',
    color: '#f59e0b',
  },
} as const;

export type AtlasName = keyof typeof ATLAS_CONFIGS;

export const SIGNATURE_TYPES = ['CytoSig', 'SecAct'] as const;
export type SignatureType = (typeof SIGNATURE_TYPES)[number];

export const STATS = [
  { value: '17.8M', label: 'Total Cells' },
  { value: '43', label: 'Cytokines (CytoSig)' },
  { value: '1,170', label: 'Proteins (SecAct)' },
  { value: '12+', label: 'Diseases' },
] as const;

export const EXAMPLE_GENES = ['IFNG', 'TNF', 'IL6', 'IL17A', 'TGFB1'] as const;

export const PLACEHOLDER_ATLASES: Atlas[] = [
  {
    name: 'cima',
    display_name: 'CIMA',
    description: 'Chinese Immune Multi-omics Atlas - Healthy adult immune profiling with biochemistry and metabolomics',
    n_cells: 6484974,
    n_samples: 421,
    n_cell_types: 39,
    atlas_type: 'builtin',
    status: 'active',
    has_cytosig: true,
    has_secact: true,
    source_type: 'builtin',
    validation_grade: 'A',
  },
  {
    name: 'inflammation',
    display_name: 'Inflammation Atlas',
    description: 'Pan-disease immune profiling across multiple inflammatory conditions with treatment response data',
    n_cells: 4900000,
    n_samples: 817,
    n_cell_types: 43,
    atlas_type: 'builtin',
    status: 'active',
    has_cytosig: true,
    has_secact: true,
    source_type: 'builtin',
    validation_grade: 'B',
  },
  {
    name: 'scatlas',
    display_name: 'scAtlas',
    description: 'Human tissue reference atlas with normal organs and pan-cancer immune profiling',
    n_cells: 6400000,
    n_samples: 781,
    n_cell_types: 213,
    atlas_type: 'builtin',
    status: 'active',
    has_cytosig: true,
    has_secact: true,
    source_type: 'builtin',
    validation_grade: 'B',
  },
];
