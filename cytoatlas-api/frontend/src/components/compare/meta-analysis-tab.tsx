import { useState, useMemo } from 'react';
import { useMetaAnalysisForest, useConservedSignatures } from '@/api/hooks/use-cross-atlas';
import { Spinner } from '@/components/ui/loading-skeleton';
import { FilterBar, SelectFilter } from '@/components/ui/filter-bar';
import { ForestPlot } from '@/components/charts/forest-plot';
import { BarChart } from '@/components/charts/bar-chart';
import { HeatmapChart } from '@/components/charts/heatmap-chart';

interface MetaAnalysisTabProps {
  signatureType: string;
}

const ANALYSIS_OPTIONS = [
  { value: 'age', label: 'Age' },
  { value: 'bmi', label: 'BMI' },
  { value: 'sex', label: 'Sex' },
];

export default function MetaAnalysisTab({ signatureType }: MetaAnalysisTabProps) {
  const [analysis, setAnalysis] = useState('age');
  const [signature, setSignature] = useState('');

  const { data: conserved } = useConservedSignatures(signatureType, 2);
  const { data: forest, isLoading } = useMetaAnalysisForest(
    analysis,
    signatureType,
    signature || undefined,
  );

  const sigOptions = useMemo(() => {
    if (!conserved) return [];
    return [
      { value: '', label: 'All signatures' },
      ...conserved.slice(0, 50).map((s) => ({ value: s.signature, label: s.signature })),
    ];
  }, [conserved]);

  const forestData = forest as
    | {
        items?: {
          signature: string;
          individual_effects: { atlas: string; effect: number; se: number; n: number }[];
          pooled_effect: number;
          ci_low: number;
          ci_high: number;
          I2: number;
        }[];
        effects?: { atlas: string; effect: number; ci_low: number; ci_high: number }[];
        pooled?: { effect: number; ci_low: number; ci_high: number };
      }
    | undefined;

  const effectBarData = useMemo(() => {
    if (!forestData?.effects) return null;
    return {
      categories: forestData.effects.map((e) => e.atlas),
      values: forestData.effects.map((e) => e.effect),
    };
  }, [forestData]);

  const summaryStats = useMemo(() => {
    if (!forestData?.items || forestData.items.length === 0) return null;
    const items = forestData.items;
    const nSignificant = items.filter(i => {
      const ci = [i.ci_low, i.ci_high];
      return ci[0] > 0 || ci[1] < 0;
    }).length;
    const nConsistent = items.filter(i => i.I2 < 50).length;
    const nReplicated = items.filter(i => {
      const ci = [i.ci_low, i.ci_high];
      return (ci[0] > 0 || ci[1] < 0) && i.I2 < 50;
    }).length;
    return { nSignificant, nConsistent, nReplicated, total: items.length };
  }, [forestData]);

  const heterogeneityData = useMemo(() => {
    if (!forestData?.items || forestData.items.length === 0) return null;
    const sorted = [...forestData.items].sort((a, b) => b.I2 - a.I2).slice(0, 30);
    return {
      categories: sorted.map(i => i.signature),
      values: sorted.map(i => i.I2),
      colors: sorted.map(i => i.I2 < 25 ? '#22c55e' : i.I2 < 50 ? '#f59e0b' : i.I2 < 75 ? '#f97316' : '#ef4444'),
    };
  }, [forestData]);

  if (isLoading) return <Spinner message="Loading meta-analysis..." />;

  return (
    <div className="space-y-6">
      <FilterBar>
        <SelectFilter
          label="Analysis"
          options={ANALYSIS_OPTIONS}
          value={analysis}
          onChange={(v) => {
            setAnalysis(v);
            setSignature('');
          }}
        />
        {sigOptions.length > 1 && (
          <SelectFilter
            label="Signature"
            options={sigOptions}
            value={signature}
            onChange={setSignature}
          />
        )}
      </FilterBar>

      {summaryStats && (
        <div className="grid grid-cols-4 gap-4">
          <div className="rounded-md border border-border-light bg-bg-secondary p-3 text-center">
            <div className="text-2xl font-bold text-green-600">{summaryStats.nReplicated}</div>
            <div className="text-xs text-text-muted">Replicated</div>
          </div>
          <div className="rounded-md border border-border-light bg-bg-secondary p-3 text-center">
            <div className="text-2xl font-bold text-blue-600">{summaryStats.nConsistent}</div>
            <div className="text-xs text-text-muted">Consistent (I&sup2; &lt; 50%)</div>
          </div>
          <div className="rounded-md border border-border-light bg-bg-secondary p-3 text-center">
            <div className="text-2xl font-bold text-purple-600">{summaryStats.nSignificant}</div>
            <div className="text-xs text-text-muted">Significant</div>
          </div>
          <div className="rounded-md border border-border-light bg-bg-secondary p-3 text-center">
            <div className="text-2xl font-bold text-text-primary">{summaryStats.total}</div>
            <div className="text-xs text-text-muted">Total Signatures</div>
          </div>
        </div>
      )}

      {forestData?.items && forestData.items.length > 0 && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Forest Plot: {analysis}
            {signature ? ` \u2014 ${signature}` : ''}
          </h3>
          <p className="mb-2 text-xs text-text-muted">
            Effect sizes across atlases with 95% confidence intervals.
          </p>
          <ForestPlot
            items={forestData.items}
            title={`Meta-Analysis: ${analysis}${signature ? ` (${signature})` : ''}`}
          />
        </div>
      )}

      {effectBarData && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Effect Sizes by Atlas
          </h3>
          <BarChart
            categories={effectBarData.categories}
            values={effectBarData.values}
            title={`Effect Size: ${analysis}`}
            yTitle="Effect Size"
          />
        </div>
      )}

      {heterogeneityData && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Heterogeneity (I&sup2;) by Signature
          </h3>
          <p className="mb-2 text-xs text-text-muted">
            I&sup2; measures inconsistency across atlases. Green: low (&lt;25%), Yellow: moderate (25-50%), Orange: substantial (50-75%), Red: high (&gt;75%).
          </p>
          <BarChart
            categories={heterogeneityData.categories}
            values={heterogeneityData.values}
            orientation="h"
            xTitle="I² (%)"
            yTitle="Signature"
            title="Cross-Atlas Heterogeneity"
            colors={heterogeneityData.colors}
            height={Math.max(400, heterogeneityData.categories.length * 24 + 150)}
          />
        </div>
      )}

      {!forestData?.items?.length && !forestData?.effects?.length && (
        <p className="py-8 text-center text-text-muted">
          No meta-analysis data available for {analysis}
        </p>
      )}
    </div>
  );
}
