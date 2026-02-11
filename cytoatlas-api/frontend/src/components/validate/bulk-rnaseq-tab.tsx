import { useState, useMemo, useEffect } from 'react';
import {
  useBulkRnaseqDatasets,
  useBulkRnaseqTargets,
  useBulkRnaseqScatter,
} from '@/api/hooks/use-validation';
import { Spinner } from '@/components/ui/loading-skeleton';
import { FilterBar, SelectFilter, CheckboxFilter } from '@/components/ui/filter-bar';
import { SearchableSelect } from '@/components/ui/searchable-select';
import { ScatterChart } from '@/components/charts/scatter-chart';
import { BarChart } from '@/components/charts/bar-chart';
import { BoxplotChart } from '@/components/charts/boxplot-chart';

const SIG_OPTIONS = [
  { value: 'cytosig', label: 'CytoSig (43)' },
  { value: 'secact', label: 'SecAct (1,249)' },
];

function spearmanRho(x: number[], y: number[]): number {
  const n = x.length;
  if (n < 3) return 0;
  const rank = (arr: number[]) => {
    const sorted = [...arr].map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
    const ranks = new Array(n);
    sorted.forEach((s, r) => {
      ranks[s.i] = r + 1;
    });
    return ranks;
  };
  const rx = rank(x),
    ry = rank(y);
  const mx = rx.reduce((a, b) => a + b) / n,
    my = ry.reduce((a, b) => a + b) / n;
  let num = 0,
    dx = 0,
    dy = 0;
  for (let i = 0; i < n; i++) {
    num += (rx[i] - mx) * (ry[i] - my);
    dx += (rx[i] - mx) ** 2;
    dy += (ry[i] - my) ** 2;
  }
  return dx * dy === 0 ? 0 : num / Math.sqrt(dx * dy);
}

export default function BulkRnaseqTab() {
  const [sigtype, setSigtype] = useState('cytosig');
  const [dataset, setDataset] = useState('');
  const [target, setTarget] = useState('');
  const [group, setGroup] = useState('');
  const [hideNonExpr, setHideNonExpr] = useState(false);

  const { data: datasets, isLoading: dsLoading } = useBulkRnaseqDatasets();
  const activeDataset = dataset || (datasets?.[0] ?? '');
  const { data: targets, isLoading: tgLoading } = useBulkRnaseqTargets(activeDataset, sigtype);
  const { data: scatter, isLoading: scLoading } = useBulkRnaseqScatter(
    activeDataset,
    target,
    sigtype,
  );

  const dsOptions = (datasets || []).map((d) => ({ value: d, label: d.toUpperCase() }));

  // Auto-select first target when targets load
  useEffect(() => {
    if (!targets || targets.length === 0) return;
    if (!target) {
      setTarget(targets[0].target);
    } else {
      const available = new Set(targets.map((t) => t.target));
      if (!available.has(target)) {
        const match = targets.find((t) => t.target.toLowerCase() === target.toLowerCase());
        setTarget(match ? match.target : targets[0].target);
        setGroup('');
      }
    }
  }, [targets]); // eslint-disable-line react-hooks/exhaustive-deps

  const targetOptions = useMemo(() => {
    if (!targets) return [];
    return targets
      .sort((a, b) => Math.abs(b.rho) - Math.abs(a.rho))
      .slice(0, 100)
      .map((t) => ({
        value: t.target,
        label: `${t.target} (rho=${t.rho.toFixed(3)})`,
      }));
  }, [targets]);

  const barData = useMemo(() => {
    if (!targets) return null;
    const sorted = [...targets].sort((a, b) => b.rho - a.rho).slice(0, 30);
    return {
      categories: sorted.map((t) => t.target),
      values: sorted.map((t) => t.rho),
    };
  }, [targets]);

  const groupOptions = useMemo(() => {
    if (!scatter?.points) return [];
    const seen = new Set<string>();
    for (const p of scatter.points) {
      const g = p.cell_type || p.label || '';
      if (g) seen.add(g);
    }
    const sorted = [...seen].sort();
    return [
      { value: '', label: 'All' },
      ...sorted.map((g) => ({ value: g, label: g })),
    ];
  }, [scatter]);

  const filteredPoints = useMemo(() => {
    if (!scatter?.points) return [];
    return scatter.points.filter((p) => {
      if (group && (p.cell_type || p.label || '') !== group) return false;
      if (hideNonExpr && p.x <= 0) return false;
      return true;
    });
  }, [scatter, group, hideNonExpr]);

  const groupCorrelationData = useMemo(() => {
    if (!scatter?.points || groupOptions.length <= 1) return null;
    const byGroup = new Map<string, { xs: number[]; ys: number[] }>();
    for (const p of scatter.points) {
      const g = p.cell_type || p.label || '';
      if (!g) continue;
      if (!byGroup.has(g)) byGroup.set(g, { xs: [], ys: [] });
      const entry = byGroup.get(g)!;
      entry.xs.push(p.x);
      entry.ys.push(p.y);
    }
    const groups: string[] = [];
    const rhos: number[] = [];
    for (const [g, { xs, ys }] of byGroup) {
      if (xs.length < 5) continue;
      groups.push(g);
      rhos.push(spearmanRho(xs, ys));
    }
    if (groups.length === 0) return null;
    const indices = groups.map((_, i) => i).sort((a, b) => rhos[b] - rhos[a]);
    return {
      categories: indices.map((i) => groups[i]),
      values: indices.map((i) => rhos[i]),
    };
  }, [scatter, groupOptions]);

  const groupBoxData = useMemo(() => {
    if (!scatter?.points || groupOptions.length <= 1) return null;
    const byGroup = new Map<string, number[]>();
    for (const p of scatter.points) {
      const g = p.cell_type || p.label || '';
      if (!g) continue;
      if (!byGroup.has(g)) byGroup.set(g, []);
      byGroup.get(g)!.push(p.y);
    }
    const groups = [...byGroup.keys()].sort();
    const values = groups.map((g) => byGroup.get(g)!);
    if (groups.length === 0) return null;
    return { groups, values };
  }, [scatter, groupOptions]);

  if (dsLoading) return <Spinner message="Loading datasets..." />;

  return (
    <div className="space-y-6">
      <FilterBar>
        <SelectFilter
          label="Signature"
          options={SIG_OPTIONS}
          value={sigtype}
          onChange={(v) => { setSigtype(v); }}
        />
        <SelectFilter
          label="Dataset"
          options={dsOptions}
          value={activeDataset}
          onChange={(v) => {
            setDataset(v);
            setTarget('');
            setGroup('');
          }}
        />
        {targetOptions.length > 0 && (
          <SearchableSelect
            label="Target"
            options={targetOptions}
            value={target}
            onChange={(v) => {
              setTarget(v);
              setGroup('');
            }}
            placeholder="Search targets..."
          />
        )}
        {target && groupOptions.length > 1 && (
          <SelectFilter
            label={activeDataset === 'gtex' ? 'Tissue' : 'Cancer Type'}
            options={groupOptions}
            value={group}
            onChange={setGroup}
          />
        )}
        {target && scatter && (
          <CheckboxFilter
            label="Hide non-expressing"
            checked={hideNonExpr}
            onChange={setHideNonExpr}
          />
        )}
      </FilterBar>

      {tgLoading && <Spinner message="Loading targets..." />}

      {barData && !target && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Top 30 Targets by Correlation ({activeDataset.toUpperCase()})
          </h3>
          <BarChart
            categories={barData.categories}
            values={barData.values}
            orientation="h"
            xTitle="Spearman rho"
            yTitle="Target"
            title={`Validation Correlations: ${activeDataset.toUpperCase()}`}
            height={Math.max(400, barData.categories.length * 24 + 150)}
          />
        </div>
      )}

      {target && scLoading && <Spinner message="Loading scatter..." />}

      {scatter && target && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            {target} — {activeDataset.toUpperCase()}
            {group && ` — ${group}`}
          </h3>
          <p className="mb-2 text-xs text-text-muted">
            Each point = one {activeDataset === 'gtex' ? 'tissue' : 'cancer type'} sample ({filteredPoints.length} points)
          </p>
          <ScatterChart
            x={filteredPoints.map((p) => p.x)}
            y={filteredPoints.map((p) => p.y)}
            groups={!group ? filteredPoints.map((p) => p.cell_type || p.label || '') : undefined}
            labels={filteredPoints.map((p) => p.label || p.cell_type || '')}
            xTitle="Signature Gene Expression"
            yTitle="Predicted Activity"
            title={`${target}: Expression vs Activity`}
            showTrendLine
            stats={{ rho: scatter.rho, p: scatter.p_value, n: filteredPoints.length }}
            height={500}
          />
        </div>
      )}

      {scatter && target && groupCorrelationData && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Per-Group Spearman Correlation — {target}
          </h3>
          <BarChart
            categories={groupCorrelationData.categories}
            values={groupCorrelationData.values}
            orientation="h"
            xTitle="Spearman rho"
            yTitle="Group"
            title={`Per-Group Correlation: ${target}`}
            height={Math.max(350, groupCorrelationData.categories.length * 24 + 150)}
          />
        </div>
      )}

      {scatter && target && groupBoxData && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Activity Distribution by Group — {target}
          </h3>
          <BoxplotChart
            groups={groupBoxData.groups}
            values={groupBoxData.values}
            yTitle="Predicted Activity"
            title={`Activity by Group: ${target}`}
            height={Math.max(400, groupBoxData.groups.length * 20 + 200)}
          />
        </div>
      )}

      {!target && !tgLoading && targets && targets.length === 0 && (
        <p className="py-8 text-center text-text-muted">
          No validation targets available for {activeDataset.toUpperCase()}
        </p>
      )}
    </div>
  );
}
