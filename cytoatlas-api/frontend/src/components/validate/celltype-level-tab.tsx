import { useState, useMemo, useEffect } from 'react';
import {
  useCelltypeLevels,
  useCelltypeTargets,
  useCelltypeScatter,
} from '@/api/hooks/use-validation';
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

interface CelltypeLevelTabProps {
  atlas: string;
}

export default function CelltypeLevelTab({ atlas }: CelltypeLevelTabProps) {
  const [sigtype, setSigtype] = useState('cytosig');
  const [level, setLevel] = useState('');
  const [target, setTarget] = useState('');
  const [group, setGroup] = useState('');
  const [hideNonExpr, setHideNonExpr] = useState(false);

  const { data: levels } = useCelltypeLevels(atlas);
  const { data: targets, isLoading: tgLoading } = useCelltypeTargets(atlas, sigtype, level);
  const { data: scatter, isLoading: scLoading } = useCelltypeScatter(
    atlas,
    target,
    sigtype,
    level,
  );

  useEffect(() => {
    if (levels && levels.length > 0 && !level) {
      setLevel(levels[0]);
    }
  }, [levels, level]);

  const levelOptions = (levels || []).map((l) => ({ value: l, label: l }));

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
    if (!scatter) return [];
    const unique = Array.from(
      new Set(scatter.points.map((p) => p.cell_type || p.label || '').filter(Boolean)),
    ).sort();
    return [
      { value: '', label: 'All groups' },
      ...unique.map((g) => ({ value: g, label: g })),
    ];
  }, [scatter]);

  const filteredPoints = useMemo(() => {
    if (!scatter) return [];
    let pts = scatter.points;
    if (group) {
      pts = pts.filter((p) => (p.cell_type || p.label || '') === group);
    }
    if (hideNonExpr) {
      pts = pts.filter((p) => p.x !== 0);
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

  const perGroupBox = useMemo(() => {
    if (!scatter) return null;
    const groups = new Map<string, number[]>();
    for (const p of scatter.points) {
      const g = p.cell_type || p.label || 'Unknown';
      if (!groups.has(g)) groups.set(g, []);
      groups.get(g)!.push(p.y);
    }
    const entries = Array.from(groups.entries())
      .map(([g, vals]) => {
        const sorted = [...vals].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        const median = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
        return { group: g, vals, median };
      })
      .sort((a, b) => b.median - a.median);
    return {
      groups: entries.map((e) => e.group),
      values: entries.map((e) => e.vals),
    };
  }, [scatter]);

  if (!atlas) {
    return (
      <p className="py-8 text-center text-text-muted">
        Select an atlas above to view cell-type level validation.
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
        {levelOptions.length > 0 && (
          <SelectFilter
            label="Level"
            options={levelOptions}
            value={level}
            onChange={(v) => {
              setLevel(v);
              setTarget('');
              setGroup('');
            }}
          />
        )}
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
        {scatter && target && groupOptions.length > 1 && (
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
            Top 30 Cell-Type Level Correlations ({atlas}, {level})
          </h3>
          <BarChart
            categories={barData.categories}
            values={barData.values}
            orientation="h"
            xTitle="Spearman rho"
            yTitle="Target"
            title={`Cell-Type Validation: ${atlas} (${level})`}
            height={Math.max(400, barData.categories.length * 24 + 150)}
          />
        </div>
      )}

      {target && scLoading && <Spinner message="Loading scatter..." />}

      {scatter && target && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            {target} — Cell-Type Level ({atlas}, {level})
            {group && ` — ${group}`}
          </h3>
          <p className="mb-2 text-xs text-text-muted">
            Each point = one cell type (all donors pooled, {filteredPoints.length} points)
          </p>
          <ScatterChart
            x={filteredPoints.map((p) => p.x)}
            y={filteredPoints.map((p) => p.y)}
            labels={filteredPoints.map((p) => p.label || p.cell_type || '')}
            xTitle="Signature Gene Expression"
            yTitle="Predicted Activity"
            title={`${target}: Cell-Type Level (${level})${group ? ` — ${group}` : ''}`}
            showTrendLine
            stats={{ rho: scatter.rho, p: scatter.p_value, n: filteredPoints.length }}
            height={500}
          />
        </div>
      )}

      {scatter && target && perGroupCorr && perGroupCorr.categories.length > 1 && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Per-Group Spearman Correlation for {target}
          </h3>
          <BarChart
            categories={perGroupCorr.categories}
            values={perGroupCorr.values}
            orientation="h"
            xTitle="Spearman rho"
            yTitle="Group"
            title={`Per-Group Correlation: ${target} (${level})`}
            height={Math.max(400, perGroupCorr.categories.length * 24 + 150)}
          />
        </div>
      )}

      {scatter && target && perGroupBox && perGroupBox.groups.length > 1 && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-text-secondary">
            Per-Group Activity Distribution for {target}
          </h3>
          <BoxplotChart
            groups={perGroupBox.groups}
            values={perGroupBox.values}
            xTitle="Group"
            yTitle="Predicted Activity"
            title={`Activity by Group: ${target} (${level})`}
            height={Math.max(400, perGroupBox.groups.length * 30 + 150)}
          />
        </div>
      )}
    </div>
  );
}
