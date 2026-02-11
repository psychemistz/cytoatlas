import { useQuery } from '@tanstack/react-query';
import { get } from '@/api/client';
import type { ActivityData, DifferentialData, HeatmapData, OrganSignature, ExhaustionDiff } from '@/api/types/activity';

export function useScatlasActivity(signatureType: string) {
  return useQuery({
    queryKey: ['scatlas', 'activity', signatureType],
    queryFn: () => get<ActivityData[]>('/atlases/scatlas/activity', { signature_type: signatureType }),
  });
}

export function useScatlasHeatmap(signatureType: string) {
  return useQuery({
    queryKey: ['scatlas', 'heatmap', signatureType],
    queryFn: () => get<HeatmapData>('/atlases/scatlas/heatmap/activity', { signature_type: signatureType }),
  });
}

export function useScatlasOrganSignatures(signatureType: string) {
  return useQuery({
    queryKey: ['scatlas', 'organs', signatureType],
    queryFn: () => get<OrganSignature[]>('/atlases/scatlas/organ-signatures', { signature_type: signatureType }),
  });
}

export function useScatlasCancerSignatures(signatureType: string) {
  return useQuery({
    queryKey: ['scatlas', 'cancer', signatureType],
    queryFn: () => get<OrganSignature[]>('/atlases/scatlas/cancer-types-signatures', { signature_type: signatureType }),
  });
}

export function useScatlasDifferential(signatureType: string) {
  return useQuery({
    queryKey: ['scatlas', 'differential', signatureType],
    queryFn: () => get<DifferentialData[]>('/atlases/scatlas/differential', { signature_type: signatureType }),
  });
}

export function useScatlasExhaustion(signatureType: string) {
  return useQuery({
    queryKey: ['scatlas', 'exhaustion', signatureType],
    queryFn: () => get<ExhaustionDiff[]>('/atlases/scatlas/exhaustion', { signature_type: signatureType }),
  });
}
