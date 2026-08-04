import { expect, test, type Page } from "@playwright/test";

async function useProceduralVisualFallback(page: Page) {
  // Keep the contract test independent of optional multi-megabyte visual downloads.
  await page.addInitScript(() => {
    const nativeRequestAnimationFrame = window.requestAnimationFrame.bind(window);
    let frameCount = 0;
    window.requestAnimationFrame = (callback) => {
      if (frameCount++ > 8) return 0;
      return nativeRequestAnimationFrame(callback);
    };
  });
  await page.route("**/assets/models/*.glb", (route) => route.abort());
  await page.route("**/assets/models/**/*.gltf", (route) => route.fulfill({
    status: 200,
    contentType: "model/gltf+json",
    body: JSON.stringify({ asset: { version: "2.0" }, scene: 0, scenes: [{ nodes: [] }] }),
  }));
}

test.describe("public product and operator console", () => {
  test("landing world renders and the primary route opens the console", async ({ page }) => {
    const pageErrors: string[] = [];
    page.on("pageerror", (error) => pageErrors.push(error.message));
    await useProceduralVisualFallback(page);
    await page.goto("/", { waitUntil: "domcontentloaded" });

    await expect(page).toHaveTitle("Muscle Memory | One robot. Many worlds.");
    await expect(page.getByRole("heading", { name: /Teach agents/ })).toBeVisible();
    const canvas = page.locator("canvas");
    await expect(canvas).toHaveCount(1);
    await expect(page.locator(".mm-world-loader")).toBeHidden({ timeout: 30_000 });
    await expect.poll(async () => {
      const box = await canvas.boundingBox();
      return box ? box.width * box.height : 0;
    }, { timeout: 30_000 }).toBeGreaterThan(10_000);
    const canvasPixels = await canvas.screenshot();
    expect(canvasPixels.length).toBeGreaterThan(1_000);
    expect(pageErrors).toEqual([]);

    await page.getByRole("link", { name: /Live console/ }).click();
    await expect(page).toHaveURL(/\/console$/);
    await expect(page.getByRole("button", { name: "Open workspace settings" })).toBeVisible();
  });

  test("landing exposes the evidence, trust, FAQ, and legal surfaces", async ({ page }) => {
    await useProceduralVisualFallback(page);
    await page.goto("/", { waitUntil: "domcontentloaded" });
    await page.locator("#system").scrollIntoViewIfNeeded();
    await expect(page.getByRole("heading", { name: /A physical-data loop/ })).toBeVisible();
    await page.locator(".mm-faq").scrollIntoViewIfNeeded();
    await expect(page.getByText("Questions teams ask first")).toBeVisible();
    await page.getByRole("link", { name: "Privacy" }).click();
    await expect(page).toHaveURL(/\/privacy$/);
    await expect(page.getByRole("heading", { name: /Privacy, kept narrow/ })).toBeVisible();
  });

  test("operator workspace keeps its navigation and simulation boundary reversible", async ({ page }) => {
    const pageErrors: string[] = [];
    page.on("pageerror", (error) => pageErrors.push(error.message));
    await useProceduralVisualFallback(page);
    await page.goto("/console", { waitUntil: "domcontentloaded" });
    await expect(page.getByRole("heading", { name: "World" })).toBeVisible();

    await page.getByRole("button", { name: "Open workspace settings" }).click();
    await expect(page.getByRole("button", { name: "Memory Graph" })).toBeVisible();
    await page.getByRole("button", { name: "Memory Graph" }).click();
    await expect(page.getByRole("heading", { name: "Live memory graph" })).toBeVisible();

    await page.getByRole("button", { name: "Open workspace settings" }).click();
    await page.getByRole("button", { name: "Episode Review" }).click();
    await expect(page.getByRole("heading", { name: "Episode review" })).toBeVisible();
    await expect(page.getByText("No persisted operational episodes are available yet.")).toBeVisible();
    await expect(page.getByText("Choose an episode to open its review workspace.")).toBeVisible();

    await page.getByRole("button", { name: "Open workspace settings" }).click();
    await page.getByRole("button", { name: "System Settings" }).click();
    await expect(page.getByRole("heading", { name: "System settings" })).toBeVisible();

    await page.getByRole("button", { name: "Open workspace settings" }).click();
    await page.getByRole("button", { name: "Operations" }).click();
    await page.getByRole("button", { name: "Run demo loop" }).click();
    await expect(page.getByRole("button", { name: "Return to live data" })).toBeVisible();
    await page.getByRole("button", { name: "Return to live data" }).click();
    await expect(page.getByRole("button", { name: "Run demo loop" })).toBeVisible();
    expect(pageErrors).toEqual([]);
  });

  test("mobile console remains within the viewport", async ({ page }) => {
    await useProceduralVisualFallback(page);
    await page.goto("/console", { waitUntil: "domcontentloaded" });
    await expect(page.getByRole("heading", { name: "World" })).toBeVisible();
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    expect(overflow).toBeLessThanOrEqual(1);
  });
});
