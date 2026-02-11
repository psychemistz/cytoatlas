import { useState, useMemo } from 'react';
import { useCimaPopulation } from '@/api/hooks/use-cima';
import type { PopulationStratification } from '@/api/types/activity';
import { Spinner } from '@/components/ui/loading-skeleton';
import { BarChart } from '@/components/charts/bar-chart';
import { FilterBar, SelectFilter } from '@/components/ui/filter-bar';

interface PopulationPanelProps {
  signatureType: string;
}

const VARIABLE_OPTIONS = [
  { value: 'sex', label: 'Sex' },
  { value: 'age', label: 'Age' },
  { value: 'bmi', label: 'BMI' },
  { value: 'blood_type', label: 'Blood Type' },
  { value: 'smoking', label: 'Smoking' },
];

function buildBarData(rows: PopulationStratification[], topN: number) {
  const sorted = [...rows].sort((a, b) => Math.abs(b.effect_size) - Math.abs(a.effect_size));
  const top = sorted.slice(0, topN);

  const categories = top.map((d) => d.signature);
  const values = top.map((d) => d.effect_size);
  const colors = top.map((d) =>
    d.effect_size >= 0 ? '#ef4444' : '#3b82f6',
  );

  return { categories, values, colors };
}

export default function PopulationPanel({ signatureType }: PopulationPanelProps) {
  const [variable, setVariable] = useState('sex');

  const { data, isLoading, error } = useCimaPopulation(signatureType, variable);

  const barData = useMemo(() => {
    if (!data || data.length === 0) return null;
    return buildBarData(data, 20);
  }, [data]);

  const sigCount = useMemo(() => {
    if (!data) return 0;
    return data.filter((d) => d.fdr < 0.05).length;
  }, [data]);

  if (isLoading) return <Spinner message="Loading population stratification..." />;

  if (error) {
    return (
      <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        Failed to load population data: {(error as Error).message}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div>
        <h3 className="mb-1 text-sm font-semibold text-text-secondary">
          Population Stratification
        </h3>
        <p className="mb-3 text-xs text-text-muted">
          Top 20 signatures by |effect size| for the selected variable.
          {data && data.length > 0 && ` ${sigCount} of ${data.length} signatures significant at FDR < 0.05.`}
        </p>
      </div>

      <FilterBar>
        <SelectFilter
          label="Variable"
          options={VARIABLE_OPTIONS}
          value={variable}
          onChange={setVariable}
        />
      </FilterBar>

      {barData && barData.categories.length > 0 ? (
        <div>
          <p className="mb-2 text-xs text-text-muted">
            Red = positive effect, Blue = negative effect ({'\u0394'} Activity between groups)
          </p>
          <BarChart
            categories={barData.categories}
            values={barData.values}
            orientation="h"
            xTitle={`\u0394 Activity (Effect Size)`}
            yTitle="Signature"
            title={`Population Stratification by ${VARIABLE_OPTIONS.find((o) => o.value === variable)?.label ?? variable}`}
            colors={barData.colors}
            height={Math.max(500, barData.categories.length * 24 + 150)}
          />
        </div>
      ) : (
        <p className="py-8 text-center text-text-muted">
          No population stratification data available for this variable
        </p>
      )}
    </div>
  );
}
