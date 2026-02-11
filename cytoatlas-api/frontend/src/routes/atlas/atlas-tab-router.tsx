import { lazy, Suspense } from 'react';
import { useAtlasStore } from '@/stores/atlas-store';
import { getAtlasTabs } from '@/lib/atlas-config';
import { TabPanel } from '@/components/ui/tab-panel';
import { Spinner } from '@/components/ui/loading-skeleton';
import { ErrorBoundary } from '@/components/ui/error-boundary';

// Shared panels
const OverviewPanel = lazy(() => import('@/components/atlas/shared/overview-panel'));
const CelltypePanel = lazy(() => import('@/components/atlas/shared/celltype-panel'));
const AgeBmiPanel = lazy(() => import('@/components/atlas/shared/age-bmi-panel'));
const AgeBmiStratified = lazy(() => import('@/components/atlas/shared/age-bmi-stratified'));
const DifferentialPanel = lazy(() => import('@/components/atlas/shared/differential-panel'));

// CIMA panels
const BiochemistryPanel = lazy(() => import('@/components/atlas/cima/biochemistry-panel'));
const BiochemScatterPanel = lazy(() => import('@/components/atlas/cima/biochem-scatter-panel'));
const MetabolitesPanel = lazy(() => import('@/components/atlas/cima/metabolites-panel'));
const MultiomicsPanel = lazy(() => import('@/components/atlas/cima/multiomics-panel'));
const PopulationPanel = lazy(() => import('@/components/atlas/cima/population-panel'));
const EqtlPanel = lazy(() => import('@/components/atlas/cima/eqtl-panel'));

// Inflammation panels
const DiseasePanel = lazy(() => import('@/components/atlas/inflammation/disease-panel'));
const SeverityPanel = lazy(() => import('@/components/atlas/inflammation/severity-panel'));
const TreatmentPanel = lazy(() => import('@/components/atlas/inflammation/treatment-panel'));
const SankeyPanel = lazy(() => import('@/components/atlas/inflammation/sankey-panel'));
const InflamValidationPanel = lazy(() => import('@/components/atlas/inflammation/validation-panel'));
const DriversPanel = lazy(() => import('@/components/atlas/inflammation/drivers-panel'));

// scAtlas panels
const TissueAtlasPanel = lazy(() => import('@/components/atlas/scatlas/tissue-atlas-panel'));
const ImmuneInfiltration = lazy(() => import('@/components/atlas/scatlas/immune-infiltration'));
const TcellStatePanel = lazy(() => import('@/components/atlas/scatlas/tcell-state-panel'));
const ExhaustionPanel = lazy(() => import('@/components/atlas/scatlas/exhaustion-panel'));
const CafPanel = lazy(() => import('@/components/atlas/scatlas/caf-panel'));

interface AtlasTabRouterProps {
  atlasName: string;
  signatureType: string;
}

export function AtlasTabRouter({ atlasName, signatureType }: AtlasTabRouterProps) {
  const tabs = getAtlasTabs(atlasName);
  const activeTab = useAtlasStore((s) => s.activeTab);
  const setActiveTab = useAtlasStore((s) => s.setActiveTab);

  if (!tabs.length) {
    return <div className="py-12 text-center text-text-muted">No tabs configured for this atlas</div>;
  }

  return (
    <TabPanel
      tabs={tabs}
      defaultTab={activeTab}
      onTabChange={setActiveTab}
      variant="pill"
    >
      {(tabId) => (
        <ErrorBoundary>
          <Suspense fallback={<Spinner message="Loading panel..." />}>
            {renderPanel(atlasName, tabId, signatureType)}
          </Suspense>
        </ErrorBoundary>
      )}
    </TabPanel>
  );
}

function renderPanel(atlas: string, tabId: string, signatureType: string) {
  const props = { signatureType, atlasName: atlas };

  // Shared panels
  switch (tabId) {
    case 'overview':
      return <OverviewPanel {...props} />;
    case 'celltypes':
      return <CelltypePanel {...props} />;
    case 'age-bmi':
      return <AgeBmiPanel {...props} />;
    case 'age-bmi-stratified':
      return <AgeBmiStratified {...props} />;
  }

  // Atlas-specific panels
  if (atlas === 'cima') {
    switch (tabId) {
      case 'differential': return <DifferentialPanel {...props} context="population" />;
      case 'biochemistry': return <BiochemistryPanel signatureType={signatureType} />;
      case 'biochem-scatter': return <BiochemScatterPanel signatureType={signatureType} />;
      case 'metabolites': return <MetabolitesPanel signatureType={signatureType} />;
      case 'multiomics': return <MultiomicsPanel signatureType={signatureType} />;
      case 'population': return <PopulationPanel signatureType={signatureType} />;
      case 'eqtl': return <EqtlPanel signatureType={signatureType} />;
    }
  }

  if (atlas === 'inflammation') {
    switch (tabId) {
      case 'differential': return <DifferentialPanel {...props} context="disease" />;
      case 'disease': return <DiseasePanel signatureType={signatureType} />;
      case 'severity': return <SeverityPanel signatureType={signatureType} />;
      case 'treatment': return <TreatmentPanel signatureType={signatureType} />;
      case 'sankey': return <SankeyPanel />;
      case 'validation': return <InflamValidationPanel signatureType={signatureType} />;
      case 'drivers': return <DriversPanel signatureType={signatureType} />;
    }
  }

  if (atlas === 'scatlas') {
    switch (tabId) {
      case 'differential-analysis': return <DifferentialPanel {...props} context="cancer" />;
      case 'tissue-atlas': return <TissueAtlasPanel signatureType={signatureType} />;
      case 'immune-infiltration': return <ImmuneInfiltration signatureType={signatureType} />;
      case 'tcell-state': return <TcellStatePanel signatureType={signatureType} />;
      case 'exhaustion': return <ExhaustionPanel signatureType={signatureType} />;
      case 'caf': return <CafPanel signatureType={signatureType} />;
    }
  }

  return <div className="py-8 text-center text-text-muted">Panel not yet implemented</div>;
}
