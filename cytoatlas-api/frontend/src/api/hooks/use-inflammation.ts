import { useQuery } from '@tanstack/react-query';
import { get } from '@/api/client';
import type { ActivityData, CorrelationData, DifferentialData, HeatmapData, TreatmentPrediction } from '@/api/types/activity';

export function useInflamActivity(signatureType: string) {
  return useQuery({
    queryKey: ['inflammation', 'activity', signatureType],
    queryFn: () => get<ActivityData[]>('/atlases/inflammation/activity', { signature_type: signatureType }),
  });
}

export function useInflamHeatmap(signatureType: string) {
  return useQuery({
    queryKey: ['inflammation', 'heatmap', signatureType],
    queryFn: () => get<HeatmapData>('/atlases/inflammation/heatmap/activity', { signature_type: signatureType }),
  });
}

export function useInflamCorrelation(variable: string, signatureType: string) {
  return useQuery({
    queryKey: ['inflammation', 'correlation', variable, signatureType],
    queryFn: () => get<CorrelationData[]>(`/atlases/inflammation/correlations/${variable}`, { signature_type: signatureType }),
    enabled: !!variable,
  });
}

export function useInflamDifferential(signatureType: string, disease?: string) {
  return useQuery({
    queryKey: ['inflammation', 'differential', signatureType, disease],
    queryFn: () => get<DifferentialData[]>('/atlases/inflammation/differential', {
      signature_type: signatureType,
      ...(disease ? { disease } : {}),
    }),
  });
}

export function useInflamDiseaseActivity(signatureType: string) {
  return useQuery({
    queryKey: ['inflammation', 'disease-activity', signatureType],
    queryFn: () => get<ActivityData[]>('/atlases/inflammation/disease-activity', { signature_type: signatureType }),
  });
}

export function useInflamTreatment(signatureType: string) {
  return useQuery({
    queryKey: ['inflammation', 'treatment', signatureType],
    queryFn: () => get<TreatmentPrediction[]>('/atlases/inflammation/treatment-predictions', { signature_type: signatureType }),
  });
}
