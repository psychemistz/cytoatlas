import { useQuery } from '@tanstack/react-query';
import { get } from '@/api/client';
import type {
  ValidationTarget,
  ScatterData,
  ScatterPoint,
  SummaryBoxplotData,
  MethodComparison,
  BulkRnaseqTarget,
  SingleCellSignature,
  SingleCellCelltypeStat,
} from '@/api/types/validation';

/**
 * Transform raw scatter response from backend.
 *
 * Backend returns compact format:
 *   { rho, pval, points: [[x, y], ...] or [[x, y, ct_idx], ...], celltypes?: [...] }
 * Frontend expects:
 *   { rho, p_value, points: [{x, y, cell_type?, label?}, ...] }
 */
function transformScatter(raw: Record<string, unknown>): ScatterData {
  const rawPoints = (raw.points ?? []) as number[][];
  const celltypes = (raw.celltypes ?? []) as string[];

  const points: ScatterPoint[] = rawPoints.map((pt) => {
    const point: ScatterPoint = { x: pt[0], y: pt[1] };
    if (pt.length > 2 && celltypes.length > 0) {
      const ctIdx = pt[2];
      point.cell_type = celltypes[ctIdx] ?? '';
      point.label = point.cell_type;
    }
    return point;
  });

  return {
    points,
    rho: raw.rho as number,
    p_value: (raw.pval ?? raw.p_value) as number,
    pearson_r: raw.pearson_r as number | undefined,
    n: raw.n as number | undefined,
    target: raw.target as string | undefined,
  };
}

/**
 * Transform singlecell-full scatter response.
 *
 * Backend returns nested format:
 *   { rho, pval, sampled: { celltypes: [...], points: [[expr, act, ct_idx, 0, is_expr], ...] } }
 */
function transformSingleCellScatter(raw: Record<string, unknown>): ScatterData {
  const sampled = (raw.sampled ?? {}) as { celltypes?: string[]; points?: number[][] };
  const rawPoints = sampled.points ?? [];
  const celltypes = sampled.celltypes ?? [];

  const points: ScatterPoint[] = rawPoints.map((pt) => {
    const point: ScatterPoint = { x: pt[0], y: pt[1] };
    if (pt.length > 2 && celltypes.length > 0) {
      const ctIdx = pt[2];
      point.cell_type = celltypes[ctIdx] ?? '';
      point.label = point.cell_type;
    }
    return point;
  });

  return {
    points,
    rho: raw.rho as number,
    p_value: (raw.pval ?? raw.p_value) as number,
    n: raw.n_total as number | undefined,
    target: raw.target as string | undefined,
  };
}

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
    queryFn: async () => {
      const raw = await get<Record<string, unknown>>(
        `/validation/bulk-rnaseq/${dataset}/scatter/${encodeURIComponent(target)}`,
        { sigtype },
      );
      return transformScatter(raw);
    },
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
    queryFn: async () => {
      const raw = await get<Record<string, unknown>>(
        `/validation/donor/${atlas}/scatter/${encodeURIComponent(target)}`,
        { sigtype },
      );
      return transformScatter(raw);
    },
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
    queryFn: async () => {
      const raw = await get<Record<string, unknown>>(
        `/validation/celltype/${atlas}/scatter/${encodeURIComponent(target)}`,
        { sigtype, level },
      );
      return transformScatter(raw);
    },
    enabled: !!atlas && !!target && !!level,
  });
}

export function useSingleCellSignatures(atlas: string, sigtype: string) {
  return useQuery({
    queryKey: ['validation', 'sc-signatures', atlas, sigtype],
    queryFn: async () => {
      const raw = await get<Record<string, unknown>[]>(
        `/validation/singlecell-full/${atlas}/signatures`,
        { sigtype },
      );
      return raw.map((s) => ({
        signature: (s.target ?? s.signature) as string,
        gene: s.gene as string | undefined,
        rho: s.rho as number | undefined,
        n_total: s.n_total as number | undefined,
        n_expressing: s.n_expressing as number | undefined,
        expressing_fraction: s.expressing_fraction as number | undefined,
      })) as SingleCellSignature[];
    },
    enabled: !!atlas,
  });
}

export function useSingleCellScatter(atlas: string, target: string, sigtype: string) {
  return useQuery({
    queryKey: ['validation', 'sc-scatter', atlas, target, sigtype],
    queryFn: async () => {
      const raw = await get<Record<string, unknown>>(
        `/validation/singlecell-full/${atlas}/scatter/${encodeURIComponent(target)}`,
        { sigtype },
      );
      return transformSingleCellScatter(raw);
    },
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
