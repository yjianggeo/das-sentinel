import { expect, test, type Page } from "@playwright/test";

function formatQueueTime(iso: string): string {
  return iso.replace("T", " ").replace(/(\.\d+)?Z$/, "");
}

async function login(page: Page): Promise<void> {
  await page.goto("/login");
  await page.locator('input[type="password"]').fill("test-admin-password");
  await page.locator('button[type="submit"]').click();
  await expect(page).toHaveURL(/\/dashboard$/);
}

// Requires the seeded stack from `python e2e/seed_stack.py` (admin password
// "test-admin-password", two processed synthetic STA/LTA events).
test("admin logs in, reviews the seeded event, and saves the human label", async ({ page }) => {
  await login(page);
  await page.goto("/events");
  await page.locator(".queue-item").first().click();

  await page.getByRole("tab").first().click();
  await expect(page.locator(".artifact-viewer img")).toBeVisible();
  await page.locator(".review-form select").selectOption("earthquake");
  await page.locator(".review-form textarea").fill("e2e synthetic event");
  await page.locator('.review-form button[type="submit"]').click();
  await expect(page.locator('.review-form [role="status"]')).toBeVisible();

  await page.goto("/models");
  await expect(page.locator("h1")).toBeVisible();
});

test("late detail response cannot replace the latest selection", async ({ page }) => {
  await login(page);
  await page.goto("/events");
  const events = (await page.evaluate(async () =>
    fetch("/api/v1/events?limit=100").then((response) => response.json()),
  )) as Array<{ id: string; occurred_at: string }>;
  expect(events.length).toBeGreaterThanOrEqual(2);
  const [first, second] = events;
  await page.route(`**/api/v1/events/${first.id}`, async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 500));
    await route.continue();
  });
  await page.getByText(formatQueueTime(first.occurred_at), { exact: true }).click();
  await page.getByText(formatQueueTime(second.occurred_at), { exact: true }).click();
  await expect(
    page.getByRole("heading", { name: formatQueueTime(second.occurred_at) }),
  ).toBeVisible();
  await page.waitForTimeout(700);
  await expect(
    page.getByRole("heading", { name: formatQueueTime(second.occurred_at) }),
  ).toBeVisible();
});

test("replaying one review request id appends one audit row", async ({ page }) => {
  await login(page);
  const result = await page.evaluate(async () => {
    const session = await fetch("/api/v1/session").then((response) => response.json());
    const [event] = await fetch("/api/v1/events?limit=1").then((response) => response.json());
    const before = await fetch(`/api/v1/events/${event.id}`).then((response) => response.json());
    const requestId = "33333333-3333-4333-8333-333333333333";
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
