import { useQuery } from '@tanstack/react-query';
import { get } from '@/api/client';
import type { Atlas, AtlasListResponse, AtlasSummary } from '@/api/types/atlas';

export function useAtlases() {
  return useQuery({
    queryKey: ['atlases'],
    queryFn: async () => {
      const res = await get<AtlasListResponse>('/atlases');
      return res.atlases.map((a) => ({
        ...a,
        source_type: a.source_type ?? a.atlas_type,
        validation_grade: a.validation_grade ?? 'A',
      }));
    },
  });
}

export function useAtlas(name: string) {
  return useQuery({
    queryKey: ['atlas', name],
    queryFn: () => get<Atlas>(`/atlases/${name}`),
    enabled: !!name,
  });
}

export function useAtlasSummary(name: string) {
  return useQuery({
    queryKey: ['atlas', name, 'summary'],
    queryFn: () => get<AtlasSummary>(`/atlases/${name}/summary`),
    enabled: !!name,
  });
}
