import { useQuery } from '@tanstack/react-query';
import { get } from '@/api/client';
import type { Atlas, AtlasSummary } from '@/api/types/atlas';

export function useAtlases() {
  return useQuery({
    queryKey: ['atlases'],
    queryFn: () => get<Atlas[]>('/atlases'),
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
