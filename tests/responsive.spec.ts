import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.setItem("uprightly-tutorial-seen", "true");
  });
});

test("mobile dashboard stays reachable without horizontal overflow", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/", { waitUntil: "domcontentloaded" });

  await expect(
    page.getByRole("button", { name: "Start Session" }),
  ).toBeVisible();
  await expect(page.getByText("Posture Score").last()).toBeVisible();
  await expect(page.getByText("Shoulder Tilt").last()).toBeVisible();

  const viewportMetrics = await page.evaluate(() => ({
    innerWidth: window.innerWidth,
    scrollWidth: document.documentElement.scrollWidth,
    scrollHeight: document.documentElement.scrollHeight,
  }));
  expect(viewportMetrics.scrollWidth).toBeLessThanOrEqual(
    viewportMetrics.innerWidth,
  );
  expect(viewportMetrics.scrollHeight).toBeGreaterThan(600);
});

test("idle session log is compact and exposes a gear settings control", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/", { waitUntil: "domcontentloaded" });
  await page.getByRole("button", { name: "Open Session Log" }).click();

  const sessionLog = page.locator('section[aria-label="Session Log"]');
  const settingsButton = page.getByRole("button", { name: "Open Settings" });

  await expect(sessionLog).toBeVisible();
  await expect(page.getByText("Ready when you are")).toBeVisible();
  await expect(settingsButton.locator(".lucide-settings")).toBeVisible();

  const [sheetBox, settingsBox] = await Promise.all([
    sessionLog.boundingBox(),
    settingsButton.boundingBox(),
  ]);

  expect(sheetBox?.height ?? Number.POSITIVE_INFINITY).toBeLessThanOrEqual(355);
  expect(
    sheetBox
      ? sheetBox.y + sheetBox.height
      : Number.POSITIVE_INFINITY,
  ).toBeLessThanOrEqual(844);
  expect(settingsBox?.width ?? 0).toBeGreaterThanOrEqual(44);
  expect(settingsBox?.height ?? 0).toBeGreaterThanOrEqual(44);
});

test("desktop camera remains the dominant canvas", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto("/", { waitUntil: "domcontentloaded" });

  const cameraStage = page.locator('[data-tour="camera-stage"]');
  await expect(cameraStage).toBeVisible();
  const stageBox = await cameraStage.boundingBox();
  expect(stageBox?.width ?? 0).toBeGreaterThan(800);
  await expect(
    page.getByRole("button", { name: "Show Session Log" }),
  ).toBeVisible();
});

test("short desktop view keeps the complete rail inside the viewport", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 760 });
  await page.goto("/", { waitUntil: "domcontentloaded" });

  const rail = page.getByTestId("desktop-rail");
  await expect(rail).toBeVisible();
  await expect(page.getByText("Shoulder Tilt").first()).toBeVisible();

  const fit = await page.evaluate(() => {
    const railElement = document.querySelector<HTMLElement>(
      '[data-testid="desktop-rail"]',
    );
    const railRect = railElement?.getBoundingClientRect();
    return {
      railBottom: railRect?.bottom ?? Number.POSITIVE_INFINITY,
      viewportHeight: window.innerHeight,
      documentHeight: document.documentElement.scrollHeight,
    };
  });

  expect(fit.railBottom).toBeLessThanOrEqual(fit.viewportHeight);
  expect(fit.documentHeight).toBeLessThanOrEqual(fit.viewportHeight);
});

test("settings uses an inset internal scrollbar", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 760 });
  await page.goto("/", { waitUntil: "domcontentloaded" });
  await page.getByRole("button", { name: "Toggle Settings" }).click();

  const scrollArea = page.getByTestId("settings-scroll-area");
  await expect(scrollArea).toBeVisible();

  const scrollState = await scrollArea.evaluate((element) => {
    const styles = window.getComputedStyle(element);
    return {
      clientHeight: element.clientHeight,
      scrollHeight: element.scrollHeight,
      overflowY: styles.overflowY,
      scrollbarWidth: styles.scrollbarWidth,
      pageHeight: document.documentElement.scrollHeight,
      viewportHeight: window.innerHeight,
    };
  });

  expect(scrollState.overflowY).toBe("auto");
  expect(scrollState.scrollbarWidth).toBe("thin");
  expect(scrollState.scrollHeight).toBeGreaterThan(scrollState.clientHeight);
  expect(scrollState.pageHeight).toBeLessThanOrEqual(scrollState.viewportHeight);
});

test("dashboard has no serious or critical accessibility violations", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto("/", { waitUntil: "domcontentloaded" });

  const results = await new AxeBuilder({ page }).analyze();
  const severeViolations = results.violations.filter((violation) =>
    violation.nodes.some(
      (node) => node.impact === "serious" || node.impact === "critical",
    ),
  );
  expect(severeViolations).toEqual([]);
});
