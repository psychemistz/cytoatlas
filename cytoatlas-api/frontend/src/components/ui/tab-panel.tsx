import { useState, type ReactNode } from 'react';
import { cn } from '@/lib/utils';

interface Tab {
  id: string;
  label: string;
  icon?: string;
  disabled?: boolean;
}

interface TabPanelProps {
  tabs: Tab[];
  defaultTab?: string;
  onTabChange?: (tabId: string) => void;
  children: (activeTab: string) => ReactNode;
  className?: string;
  orientation?: 'horizontal' | 'vertical';
  variant?: 'underline' | 'pill';
}

export function TabPanel({
  tabs,
  defaultTab,
  onTabChange,
  children,
  className,
  orientation = 'horizontal',
  variant = 'underline',
}: TabPanelProps) {
  const [activeTab, setActiveTab] = useState(defaultTab ?? tabs[0]?.id ?? '');

  function handleTabClick(tabId: string) {
    setActiveTab(tabId);
    onTabChange?.(tabId);
  }

  const isVertical = orientation === 'vertical';
  const isPill = variant === 'pill';

  return (
    <div className={cn(isVertical && 'flex gap-4', className)}>
      <div
        className={cn(
          'flex gap-1',
          isVertical
            ? 'flex-col border-r border-border-light pr-4'
            : isPill
              ? 'p-1 bg-bg-tertiary rounded-lg overflow-x-auto mb-4'
              : 'border-b border-border-light pb-1',
        )}
        role="tablist"
      >
        {tabs.map((tab) => (
          <button
            key={tab.id}
            role="tab"
            aria-selected={activeTab === tab.id}
            disabled={tab.disabled}
            onClick={() => handleTabClick(tab.id)}
            className={cn(
              'flex items-center gap-2 whitespace-nowrap px-4 py-2 text-sm font-medium transition-colors',
              isPill
                ? cn(
                    'rounded-md',
                    activeTab === tab.id
                      ? 'bg-bg-primary text-primary shadow-sm'
                      : 'text-text-secondary hover:text-text-primary hover:bg-bg-primary',
                  )
                : cn(
                    'rounded-t-md',
                    activeTab === tab.id
                      ? 'border-b-2 border-primary bg-bg-primary text-primary'
                      : 'text-text-secondary hover:bg-bg-tertiary hover:text-text-primary',
                  ),
              tab.disabled && 'cursor-not-allowed opacity-50',
            )}
          >
            {tab.icon && <span>{tab.icon}</span>}
            {tab.label}
          </button>
        ))}
      </div>
      <div
        className={cn(
          'flex-1 pt-4',
          isPill && 'rounded-lg border border-border-light bg-bg-primary p-8 shadow-sm min-h-[500px]',
        )}
        role="tabpanel"
      >
        {children(activeTab)}
      </div>
    </div>
  );
}
