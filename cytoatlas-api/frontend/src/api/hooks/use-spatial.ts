import { useQuery } from '@tanstack/react-query';
import { get } from '@/api/client';
import type {
  SpatialSummary,
  SpatialDataset,
  TissueActivity,
  TechnologyComparison,
  GeneCoverage,
  SpatialCoordinate,
} from '@/api/types/spatial';

export function useSpatialSummary() {
  return useQuery({
    queryKey: ['spatial', 'summary'],
    queryFn: () => get<SpatialSummary>('/spatial/summary'),
  });
}

export function useSpatialDatasets(technology?: string) {
  return useQuery({
    queryKey: ['spatial', 'datasets', technology],
    queryFn: () => {
      const params: Record<string, string> = {};
      if (technology) params.technology = technology;
      return get<SpatialDataset[]>('/spatial/datasets', params);
    },
  });
}

export function useTissueActivity(signatureType: string) {
  return useQuery({
    queryKey: ['spatial', 'tissue-activity', signatureType],
    queryFn: () =>
      get<TissueActivity[]>('/spatial/tissue-summary', {
        signature_type: signatureType,
      }),
  });
}

export function useTechnologyComparison(signatureType: string) {
  return useQuery({
    queryKey: ['spatial', 'technology-comparison', signatureType],
    queryFn: () =>
      get<TechnologyComparison[]>('/spatial/technology-comparison', {
        signature_type: signatureType,
      }),
  });
}

export function useGeneCoverage(technology?: string) {
  return useQuery({
    queryKey: ['spatial', 'gene-coverage', technology],
    queryFn: () => {
      const params: Record<string, string> = {};
      if (technology) params.technology = technology;
      return get<GeneCoverage[]>('/spatial/gene-coverage', params);
    },
  });
}

export function useSpatialCoordinates(datasetId: string, signatureType: string) {
  return useQuery({
    queryKey: ['spatial', 'coordinates', datasetId, signatureType],
    queryFn: () =>
      get<SpatialCoordinate[]>(`/spatial/coordinates/${datasetId}/activity`, {
        signature_type: signatureType,
      }),
    enabled: !!datasetId,
  });
}
