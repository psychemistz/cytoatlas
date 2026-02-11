import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { TabPanel } from '@/components/ui/tab-panel';

const tabs = [
  { id: 'overview', label: 'Overview' },
  { id: 'details', label: 'Details' },
  { id: 'settings', label: 'Settings' },
];

describe('TabPanel', () => {
  it('renders all tabs', () => {
    render(
      <TabPanel tabs={tabs}>{(id) => <div>Content: {id}</div>}</TabPanel>,
    );
    expect(screen.getByRole('tab', { name: 'Overview' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Details' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Settings' })).toBeInTheDocument();
  });

  it('shows first tab by default', () => {
    render(
      <TabPanel tabs={tabs}>{(id) => <div>Content: {id}</div>}</TabPanel>,
    );
    expect(screen.getByText('Content: overview')).toBeInTheDocument();
  });

  it('respects defaultTab', () => {
    render(
      <TabPanel tabs={tabs} defaultTab="settings">
        {(id) => <div>Content: {id}</div>}
      </TabPanel>,
    );
    expect(screen.getByText('Content: settings')).toBeInTheDocument();
  });

  it('switches tabs on click', () => {
    render(
      <TabPanel tabs={tabs}>{(id) => <div>Content: {id}</div>}</TabPanel>,
    );
    fireEvent.click(screen.getByRole('tab', { name: 'Details' }));
    expect(screen.getByText('Content: details')).toBeInTheDocument();
  });

  it('calls onTabChange when tab is clicked', () => {
    const onChange = vi.fn();
    render(
      <TabPanel tabs={tabs} onTabChange={onChange}>
        {(id) => <div>Content: {id}</div>}
      </TabPanel>,
    );
    fireEvent.click(screen.getByRole('tab', { name: 'Settings' }));
    expect(onChange).toHaveBeenCalledWith('settings');
  });

  it('marks active tab with aria-selected', () => {
    render(
      <TabPanel tabs={tabs}>{(id) => <div>Content: {id}</div>}</TabPanel>,
    );
    expect(screen.getByRole('tab', { name: 'Overview' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('tab', { name: 'Details' })).toHaveAttribute('aria-selected', 'false');
  });

  it('handles disabled tabs', () => {
    const disabledTabs = [
      { id: 'a', label: 'A' },
      { id: 'b', label: 'B', disabled: true },
    ];
    render(
      <TabPanel tabs={disabledTabs}>{(id) => <div>{id}</div>}</TabPanel>,
    );
    expect(screen.getByRole('tab', { name: 'B' })).toBeDisabled();
  });

  it('renders tablist and tabpanel roles', () => {
    render(
      <TabPanel tabs={tabs}>{(id) => <div>{id}</div>}</TabPanel>,
    );
    expect(screen.getByRole('tablist')).toBeInTheDocument();
    expect(screen.getByRole('tabpanel')).toBeInTheDocument();
  });
});
