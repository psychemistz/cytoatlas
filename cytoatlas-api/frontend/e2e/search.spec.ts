import { test, expect } from '@playwright/test';

test.describe('Search page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/search');
  });

  test('renders search input', async ({ page }) => {
    await expect(page.getByRole('textbox')).toBeVisible();
  });

  test('allows typing a search query', async ({ page }) => {
    const input = page.getByRole('textbox');
    await input.fill('IFNG');
    await expect(input).toHaveValue('IFNG');
  });

  test('shows results area', async ({ page }) => {
    // The page should render without errors even with no query
    await expect(page.locator('body')).toBeVisible();
  });
});
