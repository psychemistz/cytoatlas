import { useQuery } from '@tanstack/react-query';
import { get } from '@/api/client';
import type { ActivityData, CorrelationData, BiochemCorrelation, MetaboliteCorrelation, HeatmapData, PopulationStratification } from '@/api/types/activity';

export function useCimaActivity(signatureType: string) {
  return useQuery({
    queryKey: ['cima', 'activity', signatureType],
    queryFn: () => get<ActivityData[]>('/atlases/cima/activity', { signature_type: signatureType }),
  });
}

export function useCimaHeatmap(signatureType: string) {
  return useQuery({
    queryKey: ['cima', 'heatmap', signatureType],
    queryFn: () => get<HeatmapData>('/atlases/cima/heatmap/activity', { signature_type: signatureType }),
  });
}

export function useCimaCorrelation(variable: string, signatureType: string) {
  return useQuery({
    queryKey: ['cima', 'correlation', variable, signatureType],
    queryFn: () => get<CorrelationData[]>(`/atlases/cima/correlations/${variable}`, { signature_type: signatureType }),
    enabled: !!variable,
  });
}

export function useCimaBiochemistry(signatureType: string) {
  return useQuery({
    queryKey: ['cima', 'biochemistry', signatureType],
    queryFn: () => get<BiochemCorrelation[]>('/atlases/cima/correlations/biochemistry', { signature_type: signatureType }),
  });
}

export function useCimaMetabolites(signatureType: string) {
  return useQuery({
    queryKey: ['cima', 'metabolites', signatureType],
    queryFn: () => get<MetaboliteCorrelation[]>('/atlases/cima/correlations/metabolites', { signature_type: signatureType, limit: '500' }),
  });
}

export function useCimaPopulation(signatureType: string, variable: string) {
  return useQuery({
    queryKey: ['cima', 'population', signatureType, variable],
    queryFn: () => get<PopulationStratification[]>('/atlases/cima/population-stratification', { signature_type: signatureType, variable }),
    enabled: !!variable,
  });
}

export function useCimaSignatures(signatureType: string) {
  return useQuery({
    queryKey: ['cima', 'signatures', signatureType],
    queryFn: () => get<string[]>('/atlases/cima/signatures', { signature_type: signatureType }),
  });
}
