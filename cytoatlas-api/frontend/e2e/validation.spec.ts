import { test, expect } from '@playwright/test';

test.describe('Validation page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/validate');
  });

  test('renders with tabs', async ({ page }) => {
    await expect(page.getByRole('tablist')).toBeVisible();
  });

  test('has signature toggle', async ({ page }) => {
    await expect(page.getByText('CytoSig')).toBeVisible();
    await expect(page.getByText('SecAct')).toBeVisible();
  });

  test('accepts atlas query parameter', async ({ page }) => {
    await page.goto('/validate?atlas=cima');
    await expect(page.getByRole('tablist')).toBeVisible();
  });

  test('summary tab loads by default', async ({ page }) => {
    // The first tab should be active by default
    const firstTab = page.getByRole('tab').first();
    await expect(firstTab).toHaveAttribute('aria-selected', 'true');
  });

  test('can switch between tabs', async ({ page }) => {
    const tabs = page.getByRole('tab');
    const tabCount = await tabs.count();
    expect(tabCount).toBeGreaterThanOrEqual(2);

    // Click second tab
    await tabs.nth(1).click();
    await expect(tabs.nth(1)).toHaveAttribute('aria-selected', 'true');
  });
});
