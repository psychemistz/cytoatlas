import { useParams } from 'react-router';
import { useAtlasSummary } from '@/api/hooks/use-atlas';
import { useAppStore } from '@/stores/app-store';
import { AtlasHeader } from '@/routes/atlas/atlas-header';
import { AtlasTabRouter } from '@/routes/atlas/atlas-tab-router';
import { Spinner } from '@/components/ui/loading-skeleton';
import { ATLAS_CONFIGS } from '@/lib/constants';

export default function AtlasDetail() {
  const { name = '' } = useParams<{ name: string }>();
  const signatureType = useAppStore((s) => s.signatureType);
  const { data: summary, isLoading } = useAtlasSummary(name);

  const config = ATLAS_CONFIGS[name as keyof typeof ATLAS_CONFIGS];
  const displayName = summary?.display_name ?? config?.displayName ?? name;

  if (isLoading) {
    return (
      <div className="mx-auto max-w-7xl px-4 py-12">
        <Spinner message={`Loading ${displayName}...`} />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-7xl px-4 py-8">
      <AtlasHeader
        atlasName={name}
        displayName={displayName}
        summary={summary}
        signatureType={signatureType}
      />
      <AtlasTabRouter atlasName={name} signatureType={signatureType} />
    </div>
  );
}
