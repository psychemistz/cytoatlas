import { useQuery } from '@tanstack/react-query';
import { get } from '@/api/client';
import type {
  GeneOverview,
  GeneOverviewResponse,
  GeneExpressionResponse,
  GeneActivityData,
  GeneDiseaseData,
  GeneDiseaseActivityResponse,
  GeneCorrelation,
  GeneCorrelationsResponse,
} from '@/api/types/gene';

export function useGeneCheck(gene: string) {
  return useQuery({
    queryKey: ['gene', 'check', gene],
    queryFn: () => get<{ has_expression: boolean; has_cytosig: boolean; has_secact: boolean; description?: string }>(`/gene/${encodeURIComponent(gene)}/check`),
    enabled: !!gene,
  });
}

export function useGeneOverview(gene: string, signatureType: string) {
  return useQuery({
    queryKey: ['gene', gene, signatureType],
    queryFn: async () => {
      const res = await get<GeneOverviewResponse>(`/gene/${encodeURIComponent(gene)}`, { signature_type: signatureType });
      return {
        gene: res.signature,
        description: res.description,
        has_expression: res.summary_stats.has_expression,
        has_cytosig: signatureType === 'CytoSig',
        has_secact: signatureType === 'SecAct',
        cell_type_count: res.summary_stats.n_cell_types,
        atlas_count: res.summary_stats.n_atlases,
        atlases: res.atlases,
      } as GeneOverview;
    },
    enabled: !!gene,
  });
}

export function useGeneExpression(gene: string) {
  return useQuery({
    queryKey: ['gene', 'expression', gene],
    queryFn: () => get<GeneExpressionResponse>(`/gene/${encodeURIComponent(gene)}/expression`),
    enabled: !!gene,
  });
}

export function useGeneCellTypes(gene: string, signatureType: string, atlas?: string) {
  return useQuery({
    queryKey: ['gene', 'cell-types', gene, signatureType, atlas],
    queryFn: () => get<GeneActivityData[]>(`/gene/${encodeURIComponent(gene)}/cell-types`, {
      signature_type: signatureType,
      ...(atlas ? { atlas } : {}),
    }),
    enabled: !!gene,
  });
}

export function useGeneDiseases(gene: string, signatureType: string) {
  return useQuery({
    queryKey: ['gene', 'diseases', gene, signatureType],
    queryFn: async () => {
      const res = await get<GeneDiseaseActivityResponse>(`/gene/${encodeURIComponent(gene)}/diseases`, { signature_type: signatureType });
      return res.data.map((d) => ({
        disease: d.disease,
        cohort: d.disease_group,
        activity_diff: d.activity_diff,
        p_value: d.pvalue,
        fdr: d.qvalue,
        mean_disease: d.mean_disease,
        mean_healthy: d.mean_healthy,
      })) as GeneDiseaseData[];
    },
    enabled: !!gene,
  });
}

export function useGeneCorrelations(gene: string, signatureType: string) {
  return useQuery({
    queryKey: ['gene', 'correlations', gene, signatureType],
    queryFn: async () => {
      const res = await get<GeneCorrelationsResponse>(`/gene/${encodeURIComponent(gene)}/correlations`, { signature_type: signatureType });
      const flat: GeneCorrelation[] = [];
      for (const [category, items] of Object.entries({
        age: res.age,
        bmi: res.bmi,
        biochemistry: res.biochemistry,
        metabolites: res.metabolites,
      })) {
        for (const item of items) {
          flat.push({
            variable: item.variable,
            type: item.category ?? category,
            rho: item.rho,
            p_value: item.pvalue,
            q_value: item.qvalue,
            cell_type: item.cell_type,
            n: item.n_samples ?? 0,
          });
        }
      }
      return flat;
    },
    enabled: !!gene,
  });
}
