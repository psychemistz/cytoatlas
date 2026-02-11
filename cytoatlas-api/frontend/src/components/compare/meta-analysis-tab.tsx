import { useState, useMemo } from 'react';
import { useMetaAnalysisForest, useConservedSignatures } from '@/api/hooks/use-cross-atlas';
import { Spinner } from '@/components/ui/loading-skeleton';
import { FilterBar, SelectFilter } from '@/components/ui/filter-bar';
import { ForestPlot } from '@/components/charts/forest-plot';
import { BarChart } from '@/components/charts/bar-chart';

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

      {!forestData?.items?.length && !forestData?.effects?.length && (
        <p className="py-8 text-center text-text-muted">
          No meta-analysis data available for {analysis}
        </p>
      )}
    </div>
  );
}
