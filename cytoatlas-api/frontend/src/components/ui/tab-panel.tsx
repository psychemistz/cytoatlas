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
}

export function TabPanel({
  tabs,
  defaultTab,
  onTabChange,
  children,
  className,
  orientation = 'horizontal',
}: TabPanelProps) {
  const [activeTab, setActiveTab] = useState(defaultTab ?? tabs[0]?.id ?? '');

  function handleTabClick(tabId: string) {
    setActiveTab(tabId);
    onTabChange?.(tabId);
  }

  const isVertical = orientation === 'vertical';

  return (
    <div className={cn(isVertical && 'flex gap-4', className)}>
      <div
        className={cn(
          'flex gap-1',
          isVertical ? 'flex-col border-r border-border-light pr-4' : 'border-b border-border-light pb-1',
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
              'flex items-center gap-2 whitespace-nowrap rounded-t-md px-4 py-2 text-sm font-medium transition-colors',
              activeTab === tab.id
                ? 'border-b-2 border-primary bg-bg-primary text-primary'
                : 'text-text-secondary hover:bg-bg-tertiary hover:text-text-primary',
              tab.disabled && 'cursor-not-allowed opacity-50',
            )}
          >
            {tab.icon && <span>{tab.icon}</span>}
            {tab.label}
          </button>
        ))}
      </div>
      <div className="flex-1 pt-4" role="tabpanel">
        {children(activeTab)}
      </div>
    </div>
  );
}
