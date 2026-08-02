import { expect, test } from "@playwright/test";

// Requires the seeded stack from `python e2e/seed_stack.py` (admin password
// "test-admin-password", one processed synthetic STA/LTA event).
test("admin logs in, reviews the seeded event, and saves the human label", async ({ page }) => {
  await page.goto("/login");
  await page.getByLabel("管理员密码").fill("test-admin-password");
  await page.getByRole("button", { name: "登录" }).click();

  await page.getByRole("link", { name: "事件复核" }).click();
  // Queue rows are keyed by occurrence time ("YYYY-MM-DD HH:MM:SS").
  await page.getByText(/\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}/).first().click();

  await page.getByRole("tab", { name: "瀑布图", exact: true }).click();
  await expect(page.getByAltText("事件瀑布图")).toBeVisible();
  await page.getByLabel("人工标签").selectOption("earthquake");
  await page.getByLabel("备注").fill("e2e 合成事件");
  await page.getByRole("button", { name: "保存复核" }).click();
  await expect(page.getByText("已保存")).toBeVisible();

  await page.getByRole("link", { name: "模型" }).click();
  await expect(page.getByRole("heading", { name: /模型/ })).toBeVisible();
});
