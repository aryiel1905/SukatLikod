import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

test.beforeEach(async ({ page }, testInfo) => {
  const acknowledgePrivacy = !testInfo.title.startsWith("first visit");
  await page.addInitScript((shouldAcknowledgePrivacy) => {
    window.localStorage.setItem("uprightly-tutorial-seen", "true");
    if (shouldAcknowledgePrivacy) {
      window.localStorage.setItem("uprightly-privacy-notice", "2");
    } else {
      window.localStorage.removeItem("uprightly-privacy-notice");
    }
  }, acknowledgePrivacy);
});

test("first visit explains camera privacy and remembers an explicit choice", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/", { waitUntil: "domcontentloaded" });

  const purpose = page.getByRole("dialog", { name: "Welcome to Uprightly" });
  await expect(purpose).toBeVisible();
  await expect(
    purpose.getByText("Designed to work alongside you"),
  ).toBeVisible();
  await expect(
    purpose.getByText("Posture guidance only—not medical advice."),
  ).toBeVisible();
  await purpose.getByRole("button", { name: "Continue to privacy" }).click();

  const notice = page.getByRole("dialog", {
    name: "Your camera stays private",
  });
  await expect(notice).toBeVisible();
  await expect(
    notice.getByText("It does not record, upload, or save video clips."),
  ).toBeVisible();
  await expect(notice.locator(".lucide-shield-check")).toHaveCount(0);
  await expect(
    notice.getByRole("button", { name: "Continue to Uprightly" }),
  ).toBeFocused();

  await notice
    .getByRole("checkbox", {
      name: "Don't show these opening screens again",
    })
    .check();
  await notice.getByRole("button", { name: "Continue to Uprightly" }).click();
  await expect(notice).toBeHidden();
  await expect
    .poll(() =>
      page.evaluate(() =>
        window.localStorage.getItem("uprightly-privacy-notice"),
      ),
    )
    .toBe("2");
});

test("privacy policy can be revisited from settings", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/", { waitUntil: "domcontentloaded" });
  await page.getByRole("button", { name: "Open settings" }).click();
  await page.getByRole("button", { name: "View Privacy & Data Use" }).click();

  const policy = page.getByRole("dialog", { name: "Privacy & Data Use" });
  await expect(policy).toBeVisible();
  await expect(
    policy.getByRole("heading", { name: "Video and image handling" }),
  ).toBeVisible();
  await expect(
    policy.getByText("Uprightly does not currently create cookies."),
  ).toBeVisible();

  await policy.getByRole("button", { name: "Close privacy policy" }).click();
  await expect(policy).toBeHidden();
});

test("mobile shows only the computer compatibility notice", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/", { waitUntil: "domcontentloaded" });

  await expect(
    page.getByRole("heading", { name: "Designed for computers" }),
  ).toBeVisible();
  await expect(
    page.getByText(
      "Open Uprightly on a laptop or desktop with a camera to begin posture monitoring.",
    ),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Start Session" }),
  ).toHaveCount(0);
  await expect(page.locator("video, canvas")).toHaveCount(0);

  const viewportMetrics = await page.evaluate(() => ({
    innerWidth: window.innerWidth,
    scrollWidth: document.documentElement.scrollWidth,
    scrollHeight: document.documentElement.scrollHeight,
  }));
  expect(viewportMetrics.scrollWidth).toBeLessThanOrEqual(
    viewportMetrics.innerWidth,
  );
  expect(viewportMetrics.scrollHeight).toBeLessThanOrEqual(844);
});

test("application mounts only at the desktop breakpoint", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1023, height: 768 });
  await page.goto("/", { waitUntil: "domcontentloaded" });
  const notice = page.getByRole("heading", { name: "Designed for computers" });
  await expect(notice).toBeVisible();

  await page.setViewportSize({ width: 1024, height: 768 });
  await expect(
    page.getByRole("button", { name: "Start Session" }),
  ).toBeVisible();
  await expect(notice).toBeHidden();

  await page.setViewportSize({ width: 1023, height: 768 });
  await expect(notice).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Start Session" }),
  ).toHaveCount(0);
});

test("desktop camera remains the dominant canvas", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto("/", { waitUntil: "domcontentloaded" });

  const cameraStage = page.locator('[data-tour="camera-stage"]');
  await expect(cameraStage).toBeVisible();
  const stageBox = await cameraStage.boundingBox();
  expect(stageBox?.width ?? 0).toBeGreaterThan(800);
  await expect(
    page.getByRole("button", { name: "Open activity" }),
  ).toBeVisible();
  const workspacePanels = page.getByRole("navigation", {
    name: "Workspace panels",
  });
  await expect(workspacePanels).toBeVisible();
  await expect(
    workspacePanels
      .getByRole("button", { name: "Open activity" })
      .getByText("Activity", { exact: true }),
  ).toBeVisible();
  await expect(
    page
      .getByRole("button", { name: "Open settings" })
      .getByText("Settings", { exact: true }),
  ).toBeVisible();
});

test("activity and settings use one mutually exclusive utility panel", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/", { waitUntil: "domcontentloaded" });
  const cameraStage = page.locator('[data-tour="camera-stage"]');
  const cameraBoxBefore = await cameraStage.boundingBox();
  await page.getByRole("button", { name: "Open activity" }).click();

  const activityPanel = page.locator('section[aria-label="Activity"]');
  await expect(activityPanel).toBeVisible();
  await expect
    .poll(async () => (await activityPanel.boundingBox())?.width ?? 0)
    .toBeGreaterThan(380);
  const activityBox = await activityPanel.boundingBox();
  await page.getByRole("tab", { name: "Settings" }).click();

  const settingsPanel = page.locator('[data-tour="settings-panel"]');
  const settingsTab = page.getByRole("tab", { name: "Settings" });
  await expect(activityPanel).toBeHidden();
  await expect(settingsPanel).toBeVisible();
  await expect(page.getByTestId("settings-scroll-area")).toBeVisible();
  await expect(settingsTab).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await expect(settingsTab.locator(".lucide-settings")).toBeVisible();
  await expect(page.getByRole("tablist", { name: "Utility panel" })).toHaveCount(1);
  const panelControls = page.getByRole("group", { name: "Panel controls" });
  await expect(panelControls).toBeVisible();
  await expect(panelControls.getByRole("tab")).toHaveCount(2);
  await expect(
    panelControls.getByRole("button", { name: "Close utility panel" }),
  ).toBeVisible();
  const [activityTabBox, settingsTabBox] = await Promise.all([
    panelControls.getByRole("tab", { name: "Activity" }).boundingBox(),
    panelControls.getByRole("tab", { name: "Settings" }).boundingBox(),
  ]);
  expect(
    Math.abs((activityTabBox?.width ?? 0) - (settingsTabBox?.width ?? 0)),
  ).toBeLessThanOrEqual(1);

  const [settingsBox, cameraBoxAfter] = await Promise.all([
    settingsPanel.boundingBox(),
    cameraStage.boundingBox(),
  ]);
  expect(Math.abs((settingsBox?.x ?? 0) - (activityBox?.x ?? 0))).toBeLessThanOrEqual(1);
  expect(
    Math.abs((settingsBox?.width ?? 0) - (activityBox?.width ?? 0)),
  ).toBeLessThanOrEqual(1);
  expect(cameraBoxAfter?.width).toBeCloseTo(cameraBoxBefore?.width ?? 0, 0);

  await page.getByRole("tab", { name: "Activity" }).click();
  await expect(activityPanel).toBeVisible();
  const returnedActivityBox = await activityPanel.boundingBox();
  expect(returnedActivityBox?.width ?? 0).toBeGreaterThan(360);
  expect(
    Math.abs((returnedActivityBox?.x ?? 0) - (settingsBox?.x ?? 0)),
  ).toBeLessThanOrEqual(1);
});

test("desktop brand title stays fully inside the rail", async ({ page }) => {
  await page.setViewportSize({ width: 1024, height: 760 });
  await page.goto("/", { waitUntil: "domcontentloaded" });

  const rail = page.getByTestId("desktop-rail");
  const title = page.getByRole("heading", { name: "Uprightly" });
  const [railBox, titleBox, titleStyles] = await Promise.all([
    rail.boundingBox(),
    title.boundingBox(),
    title.evaluate((element) => {
      const styles = window.getComputedStyle(element);
      return {
        color: styles.color,
        backgroundClip: styles.backgroundClip,
        clientWidth: element.clientWidth,
        scrollWidth: element.scrollWidth,
      };
    }),
  ]);

  expect(titleBox?.x ?? -1).toBeGreaterThanOrEqual(railBox?.x ?? 0);
  expect((titleBox?.x ?? 0) + (titleBox?.width ?? 0)).toBeLessThanOrEqual(
    (railBox?.x ?? 0) + (railBox?.width ?? 0),
  );
  expect(titleStyles.color).not.toBe("rgba(0, 0, 0, 0)");
  expect(titleStyles.backgroundClip).not.toBe("text");
  expect(titleStyles.scrollWidth).toBeLessThanOrEqual(titleStyles.clientWidth);
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
  await page.getByRole("button", { name: "Open settings" }).click();

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

test("tutorial keyboard controls take precedence over the session shortcut", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/", { waitUntil: "domcontentloaded" });
  const tutorialLauncher = page.getByRole("button", { name: "Open Tutorial" });
  await tutorialLauncher.click();

  const dialog = page.getByRole("dialog", { name: "Begin when you're ready" });
  await expect(dialog).toBeVisible();
  await expect(page.getByRole("button", { name: "Next" })).toBeFocused();

  await page.keyboard.press("Space");
  await expect(
    page.getByRole("dialog", { name: "Frame your upper body" }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Start Session" }).first(),
  ).toBeVisible();

  await page.keyboard.press("ArrowRight");
  await expect(
    page.getByRole("dialog", { name: "Read the overall signal" }),
  ).toBeVisible();

  await page.keyboard.press("ArrowLeft");
  await expect(
    page.getByRole("dialog", { name: "Frame your upper body" }),
  ).toBeVisible();

  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
  await expect(tutorialLauncher).toBeFocused();
});

test("tutorial remains within constrained viewports", async ({ page }) => {
  const viewports = [
    { width: 1024, height: 640 },
    { width: 1280, height: 720 },
    { width: 1440, height: 1000 },
  ];

  for (const viewport of viewports) {
    await page.setViewportSize(viewport);
    await page.goto("/", { waitUntil: "domcontentloaded" });

    await page.getByRole("button", { name: "Open Tutorial" }).click();

    const dialog = page.getByRole("dialog");
    await expect(dialog).toBeVisible();
    const box = await dialog.boundingBox();
    expect(box?.x ?? -1).toBeGreaterThanOrEqual(0);
    expect(box?.y ?? -1).toBeGreaterThanOrEqual(0);
    expect((box?.x ?? 0) + (box?.width ?? 0)).toBeLessThanOrEqual(
      viewport.width,
    );
    expect((box?.y ?? 0) + (box?.height ?? 0)).toBeLessThanOrEqual(
      viewport.height,
    );
    await page.keyboard.press("Escape");
  }
});

test("tutorial leaves editable settings controls usable", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/", { waitUntil: "domcontentloaded" });
  await page.getByRole("button", { name: "Open Tutorial" }).click();
  await page.getByRole("button", { name: "Go to tutorial step 5" }).click();

  const settingsSelect = page.getByRole("combobox");
  await settingsSelect.focus();
  await page.keyboard.press("Space");
  await expect(
    page.getByRole("dialog", { name: "Tune the experience" }),
  ).toBeVisible();

  await page.evaluate(() => (document.activeElement as HTMLElement)?.blur());
  await page.keyboard.press("ArrowRight");
  await expect(page.getByRole("dialog")).toBeHidden();
});

test("theme accents use neutral white in dark mode and approved navy in light mode", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/", { waitUntil: "domcontentloaded" });

  const primaryAction = page
    .getByRole("button", { name: "Start Session" })
    .first();
  await expect(primaryAction).toHaveCSS(
    "background-color",
    "rgb(232, 231, 226)",
  );
  await expect(primaryAction).toHaveCSS("color", "rgb(23, 22, 18)");

  await page.getByRole("button", { name: "Open settings" }).click();
  const voiceSwitch = page.getByRole("switch", { name: "Voice prompts" });
  const [trackBox, thumbBox] = await Promise.all([
    voiceSwitch.boundingBox(),
    voiceSwitch.locator("span").boundingBox(),
  ]);
  expect(trackBox).not.toBeNull();
  expect(thumbBox).not.toBeNull();
  expect(thumbBox?.x ?? 0).toBeGreaterThanOrEqual(trackBox?.x ?? 0);
  expect((thumbBox?.x ?? 0) + (thumbBox?.width ?? 0)).toBeLessThanOrEqual(
    (trackBox?.x ?? 0) + (trackBox?.width ?? 0),
  );
  await expect(voiceSwitch).toHaveCSS("overflow", "hidden");

  await page.getByRole("button", { name: "Light", exact: true }).click();

  await expect(primaryAction).toHaveCSS("background-color", "rgb(10, 58, 114)");
  await expect(primaryAction).toHaveCSS("color", "rgb(255, 255, 255)");
  await expect(page.locator('[data-tour="camera-stage"]')).toHaveCSS(
    "background-color",
    "rgb(238, 244, 250)",
  );
});
