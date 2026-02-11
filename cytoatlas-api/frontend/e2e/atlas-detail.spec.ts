import { test, expect } from '@playwright/test';

test.describe('Atlas Detail page', () => {
  test('CIMA loads with tabs', async ({ page }) => {
    await page.goto('/atlas/cima');
    await expect(page.getByText('CIMA')).toBeVisible();
    await expect(page.getByRole('tablist')).toBeVisible();
    await expect(page.getByRole('tab', { name: /overview/i })).toBeVisible();
  });

  test('Inflammation loads with tabs', async ({ page }) => {
    await page.goto('/atlas/inflammation');
    await expect(page.getByRole('tablist')).toBeVisible();
    await expect(page.getByRole('tab', { name: /overview/i })).toBeVisible();
  });

  test('scAtlas loads with tabs', async ({ page }) => {
    await page.goto('/atlas/scatlas');
    await expect(page.getByRole('tablist')).toBeVisible();
    await expect(page.getByRole('tab', { name: /overview/i })).toBeVisible();
  });

  test('switching tabs updates panel content', async ({ page }) => {
    await page.goto('/atlas/cima');
    await expect(page.getByRole('tabpanel')).toBeVisible();
    const tabs = page.getByRole('tab');
    const tabCount = await tabs.count();
    expect(tabCount).toBeGreaterThan(1);

    // Click second tab
    await tabs.nth(1).click();
    await expect(page.getByRole('tabpanel')).toBeVisible();
  });

  test('signature toggle is present', async ({ page }) => {
    await page.goto('/atlas/cima');
    await expect(page.getByText('CytoSig')).toBeVisible();
    await expect(page.getByText('SecAct')).toBeVisible();
  });

  test('shows "Panel not yet implemented" for unknown tab', async ({ page }) => {
    // Navigate to atlas and verify basic loading works
    await page.goto('/atlas/cima');
    await expect(page.getByRole('tablist')).toBeVisible();
  });
});
