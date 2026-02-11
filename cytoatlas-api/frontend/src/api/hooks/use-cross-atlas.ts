import { useQuery } from '@tanstack/react-query';
import { get } from '@/api/client';

export function useCrossAtlasSummary() {
  return useQuery({
    queryKey: ['cross-atlas', 'summary'],
    queryFn: () => get<Record<string, unknown>>('/cross-atlas/summary'),
  });
}

export function useCelltypeSankey(level: string, lineage: string) {
  return useQuery({
    queryKey: ['cross-atlas', 'sankey', level, lineage],
    queryFn: () =>
      get<{
        nodes: { label: string; color: string }[];
        links: { source: number; target: number; value: number }[];
        coarse_mapping?: unknown[];
        fine_mapping?: unknown[];
      }>('/cross-atlas/celltype-sankey', { level, lineage }),
  });
}

export function usePairwiseScatter(
  atlas1: string,
  atlas2: string,
  signatureType: string,
  level: string,
) {
  return useQuery({
    queryKey: ['cross-atlas', 'pairwise', atlas1, atlas2, signatureType, level],
    queryFn: () =>
      get<{
        points: { x: number; y: number; label?: string; cell_type?: string }[];
        rho: number;
        p_value: number;
      }>('/cross-atlas/pairwise-scatter', {
        atlas1,
        atlas2,
        signature_type: signatureType,
        level,
      }),
    enabled: !!atlas1 && !!atlas2,
  });
}

export function useConservedSignatures(signatureType: string, minAtlases: number = 2) {
  return useQuery({
    queryKey: ['cross-atlas', 'conserved', signatureType, minAtlases],
    queryFn: () =>
      get<
        {
          signature: string;
          conservation_score: number;
          prevalence: number;
          atlases: string[];
        }[]
      >('/cross-atlas/conserved-signatures', {
        signature_type: signatureType,
        min_atlases: String(minAtlases),
      }),
  });
}

export function useMetaAnalysisForest(
  analysis: string,
  signatureType: string,
  signature?: string,
) {
  return useQuery({
    queryKey: ['cross-atlas', 'forest', analysis, signatureType, signature],
    queryFn: () =>
      get<Record<string, unknown>>('/cross-atlas/meta-analysis/forest', {
        analysis,
        signature_type: signatureType,
        ...(signature ? { signature } : {}),
      }),
    enabled: !!analysis,
  });
}

export function useConsistencyHeatmap(signatureType: string) {
  return useQuery({
    queryKey: ['cross-atlas', 'consistency', signatureType],
    queryFn: () =>
      get<{ z: number[][]; x: string[]; y: string[] }>('/cross-atlas/consistency', {
        signature_type: signatureType,
      }),
  });
}

export function useCorrelationMatrix(signatureType: string) {
  return useQuery({
    queryKey: ['cross-atlas', 'correlation-matrix', signatureType],
    queryFn: () =>
      get<{ z: number[][]; x: string[]; y: string[] }>(
        '/cross-atlas/heatmap/correlation-matrix',
        { signature_type: signatureType },
      ),
  });
}
