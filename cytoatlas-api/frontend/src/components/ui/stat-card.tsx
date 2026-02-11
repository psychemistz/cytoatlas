import { cn } from '@/lib/utils';

interface StatCardProps {
  value: string | number;
  label: string;
  className?: string;
}

export function StatCard({ value, label, className }: StatCardProps) {
  return (
    <div className={cn('text-center', className)}>
      <div className="text-3xl font-bold text-primary">{value}</div>
      <div className="mt-1 text-sm text-text-secondary">{label}</div>
    </div>
  );
}
