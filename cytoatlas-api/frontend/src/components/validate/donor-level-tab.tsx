import { useState, useMemo, useEffect } from 'react';
import { useDonorTargets, useDonorScatter } from '@/api/hooks/use-validation';
import { Spinner } from '@/components/ui/loading-skeleton';
import { FilterBar, SelectFilter, CheckboxFilter } from '@/components/ui/filter-bar';
import { SearchableSelect } from '@/components/ui/searchable-select';
import { ScatterChart } from '@/components/charts/scatter-chart';
import { BarChart } from '@/components/charts/bar-chart';
import { BoxplotChart } from '@/components/charts/boxplot-chart';

const SIG_OPTIONS = [
  { value: 'cytosig', label: 'CytoSig (43)' },
  { value: 'lincytosig', label: 'LinCytoSig' },
  { value: 'secact', label: 'SecAct (1,249)' },
];

function spearmanRho(x: number[], y: number[]): number {
  const n = x.length;
  if (n < 3) return 0;
  const rank = (arr: number[]) => {
    const sorted = [...arr].map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
    const ranks = new Array(n);
    sorted.forEach((s, r) => { ranks[s.i] = r + 1; });
    return ranks;
  };
  const rx = rank(x), ry = rank(y);
  const mx = rx.reduce((a, b) => a + b) / n, my = ry.reduce((a, b) => a + b) / n;
  let num = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) {
    num += (rx[i] - mx) * (ry[i] - my);
    dx += (rx[i] - mx) ** 2;
    dy += (ry[i] - my) ** 2;
  }
  return dx * dy === 0 ? 0 : num / Math.sqrt(dx * dy);
}

interface DonorLevelTabProps {
  atlas: string;
}

export default function DonorLevelTab({ atlas }: DonorLevelTabProps) {
  const [sigtype, setSigtype] = useState('cytosig');
  const [target, setTarget] = useState('');
  const [group, setGroup] = useState('');
  const [hideNonExpr, setHideNonExpr] = useState(false);
  const { data: targets, isLoading: tgLoading } = useDonorTargets(atlas, sigtype);
  const { data: scatter, isLoading: scLoading } = useDonorScatter(atlas, target, sigtype);

  // Auto-select first target when targets load
  useEffect(() => {
    if (!targets || targets.length === 0) return;
    if (!target) {
      setTarget(targets[0].target);
    } else {
      // Validate target against current list (handles atlas/sigtype changes)
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
    if (!scatter) return [];
    const unique = new Set<string>();
    for (const p of scatter.points) {
      const g = p.cell_type || p.label || '';
      if (g) unique.add(g);
    }
    return [
      { value: '', label: 'All' },
      ...[...unique].sort().map((g) => ({ value: g, label: g })),
    ];
  }, [scatter]);

  const filteredPoints = useMemo(() => {
    if (!scatter) return [];
    let pts = scatter.points;
    if (group) {
      pts = pts.filter((p) => (p.cell_type || p.label || '') === group);
    }
    if (hideNonExpr) {
      pts = pts.filter((p) => p.x > 0);
    }
    return pts;
  }, [scatter, group, hideNonExpr]);

  const perGroupCorr = useMemo(() => {
    if (!scatter) return null;
    const groups = new Map<string, { xs: number[]; ys: number[] }>();
    for (const p of scatter.points) {
      const g = p.cell_type || p.label || 'Unknown';
      if (!groups.has(g)) groups.set(g, { xs: [], ys: [] });
      const entry = groups.get(g)!;
      entry.xs.push(p.x);
      entry.ys.push(p.y);
    }
    const entries: { group: string; rho: number }[] = [];
    for (const [g, { xs, ys }] of groups) {
      entries.push({ group: g, rho: spearmanRho(xs, ys) });
    }
    entries.sort((a, b) => b.rho - a.rho);
    return {
      categories: entries.map((e) => e.group),
      values: entries.map((e) => e.rho),
    };
  }, [scatter]);

  const perGroupActivity = useMemo(() => {
    if (!scatter) return null;
    const groups = new Map<string, number[]>();
    for (const p of scatter.points) {
      const g = p.cell_type || p.label || 'Unknown';
      if (!groups.has(g)) groups.set(g, []);
      groups.get(g)!.push(p.y);
    }
    const sorted = [...groups.entries()].sort((a, b) => {
      const medA = a[1].sort((x, y) => x - y)[Math.floor(a[1].length / 2)];
      const medB = b[1].sort((x, y) => x - y)[Math.floor(b[1].length / 2)];
      return medB - medA;
    });
    return {
      groups: sorted.map(([g]) => g),
      values: sorted.map(([, v]) => v),
    };
  }, [scatter]);

  if (!atlas) {
    return (
      <p className="py-8 text-center text-text-muted">
        Select an atlas above to view donor-level validation.
      </p>
    );
  }

  return (
    <div className="space-y-6">
      <FilterBar>
        <SelectFilter
          label="Signature"
          options={SIG_OPTIONS}
          value={sigtype}
          onChange={(v) => { setSigtype(v); }}
        />
        {targetOptions.length > 0 && (
          <SearchableSelect
            label="Target"
            options={targetOptions}
            value={target}
            onChange={setTarget}
          />
        )}
        {scatter && groupOptions.length > 1 && (
          <SelectFilter
            label="Group"
            options={groupOptions}
            value={group}
            onChange={setGroup}
          />
        )}
        {scatter && target && (
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
            Top 30 Donor-Level Correlations ({atlas})
          </h3>
          <BarChart
            categories={barData.categories}
            values={barData.values}
            orientation="h"
            xTitle="Spearman rho"
            yTitle="Target"
            title={`Donor-Level Validation: ${atlas}`}
            height={Math.max(400, barData.categories.length * 24 + 150)}
          />
        </div>
      )}

      {target && scLoading && <Spinner message="Loading scatter..." />}

      {scatter && target && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            {target} — Donor Level ({atlas})
            {group && <span className="ml-2 text-text-muted">| {group}</span>}
          </h3>
          <p className="mb-2 text-xs text-text-muted">
            Each point = one donor pseudobulk sample ({filteredPoints.length} points)
          </p>
          <ScatterChart
            x={filteredPoints.map((p) => p.x)}
            y={filteredPoints.map((p) => p.y)}
            groups={!group ? filteredPoints.map((p) => p.cell_type || p.label || '') : undefined}
            labels={filteredPoints.map((p) => p.label || p.cell_type || '')}
            xTitle="Signature Gene Expression"
            yTitle="Predicted Activity"
            title={`${target}: Donor-Level Validation`}
            showTrendLine
            stats={{ rho: scatter.rho, p: scatter.p_value, n: filteredPoints.length }}
            height={500}
          />
        </div>
      )}

      {scatter && target && perGroupCorr && perGroupCorr.categories.length > 1 && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Per-Group Spearman Correlation — {target}
          </h3>
          <BarChart
            categories={perGroupCorr.categories}
            values={perGroupCorr.values}
            orientation="h"
            xTitle="Spearman rho"
            yTitle="Group"
            title={`Per-Group Correlation: ${target}`}
            height={Math.max(400, perGroupCorr.categories.length * 24 + 150)}
          />
        </div>
      )}

      {scatter && target && perGroupActivity && perGroupActivity.groups.length > 1 && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Per-Group Activity Distribution — {target}
          </h3>
          <BoxplotChart
            groups={perGroupActivity.groups}
            values={perGroupActivity.values}
            xTitle="Group"
            yTitle="Predicted Activity"
            title={`Activity by Group: ${target}`}
            height={Math.max(400, perGroupActivity.groups.length * 40 + 150)}
          />
        </div>
      )}
    </div>
  );
}
