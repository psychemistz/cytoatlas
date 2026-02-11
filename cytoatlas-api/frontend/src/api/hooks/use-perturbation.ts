import { useQuery } from '@tanstack/react-query';
import { get } from '@/api/client';
import type {
  PerturbationSummary,
  CytokineResponse,
  GroundTruth,
  DrugSensitivity,
  DoseResponse,
  PathwayActivation,
} from '@/api/types/perturbation';

export function usePerturbationSummary() {
  return useQuery({
    queryKey: ['perturbation', 'summary'],
    queryFn: () => get<PerturbationSummary>('/perturbation/summary'),
  });
}

export function useCytokineHeatmap(signatureType: string) {
  return useQuery({
    queryKey: ['perturbation', 'cytokine-heatmap', signatureType],
    queryFn: () =>
      get<CytokineResponse[]>('/perturbation/parse10m/heatmap', {
        signature_type: signatureType,
      }),
  });
}

export function useGroundTruth(signatureType: string) {
  return useQuery({
    queryKey: ['perturbation', 'ground-truth', signatureType],
    queryFn: () =>
      get<GroundTruth[]>('/perturbation/parse10m/ground-truth', {
        signature_type: signatureType,
      }),
  });
}

export function useSensitivityMatrix(signatureType: string) {
  return useQuery({
    queryKey: ['perturbation', 'sensitivity-matrix', signatureType],
    queryFn: () =>
      get<DrugSensitivity[]>('/perturbation/tahoe/sensitivity-matrix', {
        signature_type: signatureType,
      }),
  });
}

export function useDoseResponse(drug?: string) {
  return useQuery({
    queryKey: ['perturbation', 'dose-response', drug],
    queryFn: () =>
      get<DoseResponse[]>('/perturbation/tahoe/dose-response', {
        drug: drug!,
      }),
    enabled: !!drug,
  });
}

export function useDrugList() {
  return useQuery({
    queryKey: ['perturbation', 'drug-list'],
    queryFn: () => get<string[]>('/perturbation/tahoe/drugs'),
  });
}

export function usePathwayActivation() {
  return useQuery({
    queryKey: ['perturbation', 'pathway-activation'],
    queryFn: () =>
      get<PathwayActivation[]>('/perturbation/tahoe/pathway-activation'),
  });
}
