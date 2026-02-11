import { useQuery } from '@tanstack/react-query';
import { get } from '@/api/client';
import type {
  GeneOverview,
  GeneExpressionData,
  GeneActivityData,
  GeneDiseaseData,
  GeneCorrelation,
} from '@/api/types/gene';

export function useGeneCheck(gene: string) {
  return useQuery({
    queryKey: ['gene', 'check', gene],
    queryFn: () => get<{ has_expression: boolean; has_cytosig: boolean; has_secact: boolean }>(`/gene/${encodeURIComponent(gene)}/check`),
    enabled: !!gene,
  });
}

export function useGeneOverview(gene: string, signatureType: string) {
  return useQuery({
    queryKey: ['gene', gene, signatureType],
    queryFn: () => get<GeneOverview>(`/gene/${encodeURIComponent(gene)}`, { signature_type: signatureType }),
    enabled: !!gene,
  });
}

export function useGeneExpression(gene: string) {
  return useQuery({
    queryKey: ['gene', 'expression', gene],
    queryFn: () => get<{ data: GeneExpressionData[]; expression_boxplot?: { data: GeneExpressionData[] } }>(`/gene/${encodeURIComponent(gene)}/expression`),
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
    queryFn: () => get<GeneDiseaseData[]>(`/gene/${encodeURIComponent(gene)}/diseases`, { signature_type: signatureType }),
    enabled: !!gene,
  });
}

export function useGeneCorrelations(gene: string, signatureType: string) {
  return useQuery({
    queryKey: ['gene', 'correlations', gene, signatureType],
    queryFn: () => get<GeneCorrelation[]>(`/gene/${encodeURIComponent(gene)}/correlations`, { signature_type: signatureType }),
    enabled: !!gene,
  });
}
