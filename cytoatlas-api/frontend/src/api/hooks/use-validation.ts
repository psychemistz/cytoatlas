import { useQuery } from '@tanstack/react-query';
import { get } from '@/api/client';
import type {
  ValidationTarget,
  ScatterData,
  SummaryBoxplotData,
  MethodComparison,
  BulkRnaseqTarget,
  SingleCellSignature,
  SingleCellCelltypeStat,
} from '@/api/types/validation';

export function useValidationAtlases() {
  return useQuery({
    queryKey: ['validation', 'atlases'],
    queryFn: () => get<string[]>('/validation/donor/atlases'),
  });
}

export function useSummaryBoxplot(sigtype: string) {
  return useQuery({
    queryKey: ['validation', 'summary-boxplot', sigtype],
    queryFn: () => get<SummaryBoxplotData>('/validation/summary-boxplot', { sigtype }),
  });
}

export function useMethodComparison() {
  return useQuery({
    queryKey: ['validation', 'method-comparison'],
    queryFn: () => get<MethodComparison>('/validation/method-comparison'),
  });
}

export function useBulkRnaseqDatasets() {
  return useQuery({
    queryKey: ['validation', 'bulk-datasets'],
    queryFn: () => get<string[]>('/validation/bulk-rnaseq/datasets'),
  });
}

export function useBulkRnaseqTargets(dataset: string, sigtype: string) {
  return useQuery({
    queryKey: ['validation', 'bulk-targets', dataset, sigtype],
    queryFn: () => get<BulkRnaseqTarget[]>(`/validation/bulk-rnaseq/${dataset}/targets`, { sigtype }),
    enabled: !!dataset,
  });
}

export function useBulkRnaseqScatter(dataset: string, target: string, sigtype: string) {
  return useQuery({
    queryKey: ['validation', 'bulk-scatter', dataset, target, sigtype],
    queryFn: () =>
      get<ScatterData>(
        `/validation/bulk-rnaseq/${dataset}/scatter/${encodeURIComponent(target)}`,
        { sigtype },
      ),
    enabled: !!dataset && !!target,
  });
}

export function useDonorTargets(atlas: string, sigtype: string) {
  return useQuery({
    queryKey: ['validation', 'donor-targets', atlas, sigtype],
    queryFn: () => get<ValidationTarget[]>(`/validation/donor/${atlas}/targets`, { sigtype }),
    enabled: !!atlas,
  });
}

export function useDonorScatter(atlas: string, target: string, sigtype: string) {
  return useQuery({
    queryKey: ['validation', 'donor-scatter', atlas, target, sigtype],
    queryFn: () =>
      get<ScatterData>(
        `/validation/donor/${atlas}/scatter/${encodeURIComponent(target)}`,
        { sigtype },
      ),
    enabled: !!atlas && !!target,
  });
}

export function useCelltypeLevels(atlas: string) {
  return useQuery({
    queryKey: ['validation', 'celltype-levels', atlas],
    queryFn: () => get<string[]>(`/validation/celltype/${atlas}/levels`),
    enabled: !!atlas,
  });
}

export function useCelltypeTargets(atlas: string, sigtype: string, level: string) {
  return useQuery({
    queryKey: ['validation', 'celltype-targets', atlas, sigtype, level],
    queryFn: () =>
      get<ValidationTarget[]>(`/validation/celltype/${atlas}/targets`, { sigtype, level }),
    enabled: !!atlas && !!level,
  });
}

export function useCelltypeScatter(
  atlas: string,
  target: string,
  sigtype: string,
  level: string,
) {
  return useQuery({
    queryKey: ['validation', 'celltype-scatter', atlas, target, sigtype, level],
    queryFn: () =>
      get<ScatterData>(
        `/validation/celltype/${atlas}/scatter/${encodeURIComponent(target)}`,
        { sigtype, level },
      ),
    enabled: !!atlas && !!target && !!level,
  });
}

export function useSingleCellSignatures(atlas: string, sigtype: string) {
  return useQuery({
    queryKey: ['validation', 'sc-signatures', atlas, sigtype],
    queryFn: () =>
      get<SingleCellSignature[]>(`/validation/singlecell-full/${atlas}/signatures`, { sigtype }),
    enabled: !!atlas,
  });
}

export function useSingleCellScatter(atlas: string, target: string, sigtype: string) {
  return useQuery({
    queryKey: ['validation', 'sc-scatter', atlas, target, sigtype],
    queryFn: () =>
      get<ScatterData>(
        `/validation/singlecell-full/${atlas}/scatter/${encodeURIComponent(target)}`,
        { sigtype },
      ),
    enabled: !!atlas && !!target,
  });
}

export function useSingleCellCelltypes(atlas: string, target: string, sigtype: string) {
  return useQuery({
    queryKey: ['validation', 'sc-celltypes', atlas, target, sigtype],
    queryFn: () =>
      get<SingleCellCelltypeStat[]>(
        `/validation/singlecell-full/${atlas}/celltypes/${encodeURIComponent(target)}`,
        { sigtype },
      ),
    enabled: !!atlas && !!target,
  });
}
