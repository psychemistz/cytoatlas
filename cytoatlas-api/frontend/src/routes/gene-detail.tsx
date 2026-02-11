import { lazy, Suspense, useMemo } from 'react';
import { useParams } from 'react-router';
import { useAppStore } from '@/stores/app-store';
import { useGeneCheck, useGeneOverview } from '@/api/hooks/use-gene';
import { TabPanel } from '@/components/ui/tab-panel';
import { SignatureToggle } from '@/components/ui/signature-toggle';
import { Spinner } from '@/components/ui/loading-skeleton';
import { ErrorBoundary } from '@/components/ui/error-boundary';

const ExpressionTab = lazy(() => import('@/components/gene/expression-tab'));
const CytosigTab = lazy(() => import('@/components/gene/cytosig-tab'));
const SecactTab = lazy(() => import('@/components/gene/secact-tab'));
const DiseasesTab = lazy(() => import('@/components/gene/diseases-tab'));
const CorrelationsTab = lazy(() => import('@/components/gene/correlations-tab'));

export default function GeneDetail() {
  const { symbol = '' } = useParams<{ symbol: string }>();
  const signatureType = useAppStore((s) => s.signatureType);

  const { data: check, isLoading: checkLoading } = useGeneCheck(symbol);
  const { data: overview, isLoading: overviewLoading } = useGeneOverview(symbol, signatureType);

  const tabs = useMemo(() => {
    const t = [];
    if (!check || check.has_expression) t.push({ id: 'expression', label: 'Expression' });
    if (!check || check.has_cytosig) t.push({ id: 'cytosig', label: 'CytoSig' });
    if (!check || check.has_secact) t.push({ id: 'secact', label: 'SecAct' });
    t.push({ id: 'diseases', label: 'Diseases' });
    t.push({ id: 'correlations', label: 'Correlations' });
    return t;
  }, [check]);

  if (checkLoading || overviewLoading) {
    return (
      <div className="mx-auto max-w-[1400px] px-4 py-12">
        <Spinner message={`Loading ${symbol}...`} />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-[1400px] px-4 py-8">
      <div className="mb-6">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold">{symbol}</h1>
            {overview?.description && (
              <p className="mt-1 text-sm text-text-secondary">{overview.description}</p>
            )}
            <div className="mt-2 flex gap-3">
              {check?.has_expression && (
                <span className="rounded-full bg-green-100 px-2.5 py-0.5 text-xs font-medium text-green-700">
                  Expression
                </span>
              )}
              {check?.has_cytosig && (
                <span className="rounded-full bg-blue-100 px-2.5 py-0.5 text-xs font-medium text-blue-700">
                  CytoSig
                </span>
              )}
              {check?.has_secact && (
                <span className="rounded-full bg-purple-100 px-2.5 py-0.5 text-xs font-medium text-purple-700">
                  SecAct
                </span>
              )}
            </div>
          </div>
          <SignatureToggle />
        </div>
      </div>

      {tabs.length === 0 ? (
        <p className="py-12 text-center text-text-muted">
          No data available for {symbol}
        </p>
      ) : (
        <TabPanel tabs={tabs} defaultTab={tabs[0].id}>
          {(tabId) => (
            <ErrorBoundary>
              <Suspense fallback={<Spinner message="Loading tab..." />}>
                {tabId === 'expression' && <ExpressionTab gene={symbol} />}
                {tabId === 'cytosig' && (
                  <CytosigTab gene={symbol} signatureType="CytoSig" />
                )}
                {tabId === 'secact' && (
                  <SecactTab gene={symbol} signatureType="SecAct" />
                )}
                {tabId === 'diseases' && (
                  <DiseasesTab gene={symbol} signatureType={signatureType} />
                )}
                {tabId === 'correlations' && (
                  <CorrelationsTab gene={symbol} signatureType={signatureType} />
                )}
              </Suspense>
            </ErrorBoundary>
          )}
        </TabPanel>
      )}
    </div>
  );
}
