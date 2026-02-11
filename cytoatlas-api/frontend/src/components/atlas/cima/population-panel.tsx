import { useState, useMemo } from 'react';
import { useCimaPopulation } from '@/api/hooks/use-cima';
import type { PopulationStratification } from '@/api/types/activity';
import { Spinner } from '@/components/ui/loading-skeleton';
import { BarChart } from '@/components/charts/bar-chart';
import { PlotlyChart } from '@/components/charts/plotly-chart';
import { FilterBar, SelectFilter } from '@/components/ui/filter-bar';
import type { Data, Layout } from 'plotly.js-dist-min';

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

/** Maps variable → group label/column pairs for extracting sample counts from the data rows. */
const VARIABLE_GROUP_COLUMNS: Record<string, { label: string; column: string }[]> = {
  sex: [
    { label: 'Male', column: 'n_male' },
    { label: 'Female', column: 'n_female' },
  ],
  age: [
    { label: 'Young (<40)', column: 'n_young' },
    { label: 'Older (>60)', column: 'n_older' },
  ],
  bmi: [
    { label: 'Normal', column: 'n_normal' },
    { label: 'Obese', column: 'n_obese' },
  ],
  blood_type: [
    { label: 'Type O', column: 'n_type_o' },
    { label: 'Type A', column: 'n_type_a' },
  ],
  smoking: [
    { label: 'Never', column: 'n_never' },
    { label: 'Current', column: 'n_current' },
  ],
};

function buildPieData(rows: Record<string, unknown>[], variable: string): { data: Data[]; layout: Partial<Layout> } | null {
  if (!rows || rows.length === 0) return null;

  const groupCols = VARIABLE_GROUP_COLUMNS[variable];
  if (!groupCols) return null;

  // Use the first row to extract sample counts (same across all signatures)
  const firstRow = rows[0] as Record<string, unknown>;
  const labels: string[] = [];
  const values: number[] = [];

  for (const { label, column } of groupCols) {
    const n = Number(firstRow[column]);
    if (!isNaN(n) && n > 0) {
      labels.push(label);
      values.push(n);
    }
  }

  if (labels.length === 0) return null;

  const traces: Data[] = [
    {
      type: 'pie',
      labels,
      values,
      textinfo: 'label+value+percent',
      hovertemplate: '%{label}: %{value} samples (%{percent})<extra></extra>',
    },
  ];

  const layout: Partial<Layout> = {
    showlegend: true,
    margin: { t: 30, b: 30, l: 30, r: 30 },
    height: 300,
  };

  return { data: traces, layout };
}

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

  const pieData = useMemo(() => {
    if (!data || data.length === 0) return null;
    return buildPieData(data as unknown as Record<string, unknown>[], variable);
  }, [data, variable]);

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

      {pieData && (
        <div>
          <h4 className="mb-2 text-sm font-semibold text-text-secondary">
            Sample Distribution by {VARIABLE_OPTIONS.find((o) => o.value === variable)?.label ?? variable}
          </h4>
          <PlotlyChart data={pieData.data} layout={pieData.layout} />
        </div>
      )}

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
