import { expect, test, type Page } from "@playwright/test";

function formatQueueTime(iso: string, timeZone: string): string {
  const parts = new Intl.DateTimeFormat("zh-CN", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hourCycle: "h23",
  }).formatToParts(new Date(iso));
  const value = Object.fromEntries(parts.map((part) => [part.type, part.value]));
  return `${value.year}-${value.month}-${value.day} ${value.hour}:${value.minute}:${value.second}`;
}

async function login(page: Page): Promise<void> {
  await page.goto("/login");
  await page.locator('input[type="password"]').fill("test-admin-password");
  await page.locator('button[type="submit"]').click();
  await expect(page).toHaveURL(/\/dashboard$/);
}

async function assertFullscreenContained(page: Page): Promise<void> {
  const fullscreenTrigger = page.getByRole("button", { name: "全屏查看" });
  await fullscreenTrigger.click();
  const dialog = page.getByRole("dialog");
  const imageBox = await dialog.locator("img").boundingBox();
  const close = page.getByRole("button", { name: "关闭全屏" });
  const closeBox = await close.boundingBox();
  expect(imageBox).not.toBeNull();
  expect(closeBox).not.toBeNull();
  expect(closeBox!.y + closeBox!.height).toBeLessThanOrEqual(imageBox!.y);
  expect(imageBox!.y).toBeGreaterThanOrEqual(0);
  expect(imageBox!.y + imageBox!.height).toBeLessThanOrEqual(
    page.viewportSize()!.height,
  );
  await expect(close).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(dialog).toHaveCount(0);
  await expect(fullscreenTrigger).toBeFocused();
}

async function artifactHref(page: Page, tabName: string): Promise<string> {
  await page.getByRole("tab", { name: tabName, exact: true }).click();
  const href = await page.getByRole("link", { name: "下载" }).getAttribute("href");
  expect(href).not.toBeNull();
  return href!;
}

// Requires the seeded stack from `python e2e/seed_stack.py` (admin password
// "test-admin-password", two processed synthetic STA/LTA events).
test("admin logs in, reviews the seeded event, and saves the human label", async ({ page }) => {
  await login(page);
  await page.goto("/events");
  await page.locator(".queue-item").first().click();

  const firstTab = page.getByRole("tab").first();
  const secondTab = page.getByRole("tab").nth(1);
  await firstTab.focus();
  await firstTab.press("ArrowRight");
  await expect(secondTab).toHaveAttribute("aria-selected", "true");
  await expect(page.locator(".artifact-viewer img")).toBeVisible();
  await assertFullscreenContained(page);
  await page.locator(".review-form select").selectOption("earthquake");
  await page.locator(".review-form textarea").fill("e2e synthetic event");
  await page.locator('.review-form button[type="submit"]').click();
  await expect(page.locator('.review-form [role="status"]')).toBeVisible();

  await page.goto("/models");
  await expect(page.locator("h1")).toBeVisible();
});

test("seeded event previews both waterfall distance ranges", async ({ page }) => {
  // Catches a missing accessible range form or a preview that replaces only one waterfall.
  await login(page);
  await page.goto("/events");
  const options = await page.evaluate(async () => {
    const [event] = (await fetch("/api/v1/events?limit=1").then((response) => response.json())) as Array<{
      id: string;
    }>;
    return fetch(`/api/v1/events/${event.id}/waterfall-options`).then((response) => response.json());
  });
  expect(options).toMatchObject({
    automatic_distance_start_m: 8,
    automatic_distance_stop_m: 36,
    primary_detector: "stalta",
    primary_detector_display: "STA/LTA",
    filter_low_hz: 2,
    filter_high_hz: 40,
  });
  await page.locator(".queue-item").first().click();

  await expect(page.getByText("可用 8–36 m")).toBeVisible();
  await expect(page.getByLabel("起始距离（m）")).toHaveValue("8");
  await expect(page.getByLabel("结束距离（m）")).toHaveValue("36");
  await expect(page.getByText("滤波瀑布图（2–40 Hz）")).toBeVisible();
  await expect(page.getByText("主检测器：STA/LTA")).toBeVisible();

  const originalRaw = await artifactHref(page, "瀑布图");
  const originalFiltered = await artifactHref(page, "滤波瀑布图");
  const originalWaveform = await artifactHref(page, "波形");
  const originalPsd = await artifactHref(page, "PSD");
  const originalStft = await artifactHref(page, "STFT");
  await page.getByRole("tab", { name: "瀑布图", exact: true }).click();

  await page.getByLabel("起始距离（m）").fill("10");
  await page.getByLabel("结束距离（m）").fill("22");
  await page.getByRole("button", { name: "生成临时预览" }).click();
  await expect(page.getByRole("link", { name: "下载" })).toHaveAttribute(
    "href",
    /^blob:/,
  );
  const previewRaw = await artifactHref(page, "瀑布图");
  expect(previewRaw).toMatch(/^blob:/);
  expect(previewRaw).not.toBe(originalRaw);
  await page.getByRole("tab", { name: "滤波瀑布图", exact: true }).click();
  await expect(page.getByRole("link", { name: "下载" })).toHaveAttribute(
    "href",
    /^blob:/,
  );
  const previewFiltered = await artifactHref(page, "滤波瀑布图");
  expect(previewFiltered).toMatch(/^blob:/);
  expect(previewFiltered).not.toBe(originalFiltered);
  expect(previewFiltered).not.toBe(previewRaw);
  await expect(page.getByAltText("事件滤波瀑布图")).toHaveAttribute("src", /^blob:/);
  expect(await artifactHref(page, "波形")).toBe(originalWaveform);
  expect(await artifactHref(page, "PSD")).toBe(originalPsd);
  expect(await artifactHref(page, "STFT")).toBe(originalStft);

  await page.getByRole("tab", { name: "瀑布图", exact: true }).click();
  await assertFullscreenContained(page);
  await page.getByRole("button", { name: "重置临时预览" }).click();

  await expect(page.getByLabel("起始距离（m）")).toHaveValue("8");
  await expect(page.getByLabel("结束距离（m）")).toHaveValue("36");
  expect(await artifactHref(page, "瀑布图")).toBe(originalRaw);
  expect(await artifactHref(page, "滤波瀑布图")).toBe(originalFiltered);
});

test("artifact integrity failure stays visible and retryable", async ({ page }) => {
  await login(page);
  await page.route("**/api/v1/artifacts/*", (route) =>
    route.fulfill({
      status: 409,
      contentType: "application/json",
      body: JSON.stringify({ detail: "Artifact integrity check failed" }),
    }),
  );
  await page.goto("/events");
  await page.locator(".queue-item").first().click();

  await expect(page.getByText("图件完整性校验失败，请联系管理员")).toBeVisible();
  await expect(page.getByRole("button", { name: "重试图件" })).toBeVisible();
  await expect(page.getByText("规则判断")).toBeVisible();
});

test("late detail response cannot replace the latest selection", async ({ page }) => {
  await login(page);
  await page.goto("/events");
  const events = (await page.evaluate(async () =>
    fetch("/api/v1/events?limit=100").then((response) => response.json()),
  )) as Array<{ id: string; occurred_at: string; site_timezone: string }>;
  expect(events.length).toBeGreaterThanOrEqual(2);
  const [first, second] = events;
  await page.route(`**/api/v1/events/${first.id}`, async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 500));
    await route.continue();
  });
  await page.getByText(formatQueueTime(first.occurred_at, first.site_timezone), { exact: true }).click();
  await page.getByText(formatQueueTime(second.occurred_at, second.site_timezone), { exact: true }).click();
  await expect(
    page.getByRole("heading", { name: formatQueueTime(second.occurred_at, second.site_timezone) }),
  ).toBeVisible();
  await page.waitForTimeout(700);
  await expect(
    page.getByRole("heading", { name: formatQueueTime(second.occurred_at, second.site_timezone) }),
  ).toBeVisible();
});

test("replaying one review request id appends one audit row", async ({ page }) => {
  await login(page);
  const result = await page.evaluate(async () => {
    const session = await fetch("/api/v1/session").then((response) => response.json());
    const [event] = await fetch("/api/v1/events?limit=1").then((response) => response.json());
    const before = await fetch(`/api/v1/events/${event.id}`).then((response) => response.json());
    const requestId = crypto.randomUUID();
    const submit = () =>
      fetch(`/api/v1/events/${event.id}/reviews`, {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json", "X-CSRF-Token": session.csrf_token },
        body: JSON.stringify({
          label: "earthquake",
          note: "idempotent e2e",
          request_id: requestId,
        }),
      });
    const responses = await Promise.all([submit(), submit()]);
    const after = await fetch(`/api/v1/events/${event.id}`).then((response) => response.json());
    return {
      statuses: responses.map((response) => response.status).sort(),
      added: after.review_history.length - before.review_history.length,
    };
  });
  expect(result.statuses).toEqual([200, 201]);
  expect(result.added).toBe(1);
});
