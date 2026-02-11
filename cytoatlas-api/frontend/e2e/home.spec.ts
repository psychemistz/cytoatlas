import { test, expect } from '@playwright/test';

test.describe('Home page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('renders hero section', async ({ page }) => {
    await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
    await expect(page.getByText('CytoAtlas')).toBeVisible();
  });

  test('displays atlas cards', async ({ page }) => {
    await expect(page.getByText('CIMA')).toBeVisible();
    await expect(page.getByText('Inflammation Atlas')).toBeVisible();
    await expect(page.getByText('scAtlas')).toBeVisible();
  });

  test('displays statistics', async ({ page }) => {
    await expect(page.getByText('Total Cells')).toBeVisible();
    await expect(page.getByText('Cytokines')).toBeVisible();
  });

  test('navigates to atlas detail on card click', async ({ page }) => {
    await page.getByText('CIMA').click();
    await expect(page).toHaveURL(/atlas\/cima/);
  });

  test('has navigation links', async ({ page }) => {
    await expect(page.getByRole('link', { name: /explore/i })).toBeVisible();
    await expect(page.getByRole('link', { name: /compare/i })).toBeVisible();
    await expect(page.getByRole('link', { name: /validate/i })).toBeVisible();
  });
});
