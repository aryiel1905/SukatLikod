import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 60_000,
  expect: { timeout: 5_000 },
  use: {
    baseURL: "http://127.0.0.1:43817",
    colorScheme: "dark",
    trace: "retain-on-failure",
    launchOptions: {
      args: [
        "--use-fake-ui-for-media-stream",
        "--use-fake-device-for-media-stream",
      ],
    },
  },
  webServer: {
    command: "npm run dev -- --host 127.0.0.1 --port 43817 --strictPort",
    url: "http://127.0.0.1:43817",
    reuseExistingServer: false,
  },
});
