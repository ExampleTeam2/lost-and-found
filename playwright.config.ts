import { defineConfig, devices } from '@playwright/test';

import { chromium } from 'playwright-extra';
import stealth from 'puppeteer-extra-plugin-stealth';
import fs from 'fs';
import { DATA_PATH, TEMP_PATH, getTimestampString } from './playwright_base_config';

const timestamp = getTimestampString();

chromium.use(stealth());

const setUpTmp = () => {
  console.log('Setting up environment...');
  if (!fs.existsSync(DATA_PATH)){
    fs.mkdirSync(DATA_PATH);
  }
  if (!fs.existsSync(TEMP_PATH)){
    fs.mkdirSync(TEMP_PATH);
  }
  fs.appendFileSync(TEMP_PATH + 'games', '');
  fs.writeFileSync(TEMP_PATH + 'initial', 'true');
  fs.appendFileSync(TEMP_PATH + 'rate-limits', timestamp + '\n');
  fs.appendFileSync(TEMP_PATH + 'double-joins', timestamp + '\n');
  fs.appendFileSync(TEMP_PATH + 'starts', timestamp + '\n');
  fs.writeFileSync(TEMP_PATH + 'stop', 'false');
};

setUpTmp();

/**
 * Read environment variables from file.
 * https://github.com/motdotla/dotenv
 */
// require('dotenv').config();

/**
 * See https://playwright.dev/docs/test-configuration.
 */
export default defineConfig({
  testDir: './scraping',
  /* Run tests in files in parallel */
  fullyParallel: true,
  /* Fail the build on CI if you accidentally left test.only in the source code. */
  forbidOnly: !!process.env.CI,
  /* Retry on CI only */
  retries: process.env.CI ? 1000 : 0,
  /* Opt out of parallel tests on CI. */
  workers: process.env.CI ? 5 : undefined,
  /* Reporter to use. See https://playwright.dev/docs/test-reporters */
  reporter: process.env.CI ? [['json', { outputFile: `./scraping/reports/results_${timestamp}.json` }]] : 'html',
  /* Shared settings for all the projects below. See https://playwright.dev/docs/api/class-testoptions. */
  use: {
    /* Base URL to use in actions like `await page.goto('/')`. */
    // baseURL: 'http://127.0.0.1:3000',

    /* Collect trace when retrying the failed test. See https://playwright.dev/docs/trace-viewer */
    trace: process.env.CI ? 'off' : 'retain-on-failure',
    screenshot: 'only-on-failure'
  },

  /* Configure projects for major browsers */
  projects: [
    // No idea if this acutally works
    { 
      name: 'chromium-stealth',
      use: { ...devices['Desktop Chrome'], ...chromium }
    },
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },

    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },

    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },

    /* Test against mobile viewports. */
    // {
    //   name: 'Mobile Chrome',
    //   use: { ...devices['Pixel 5'] },
    // },
    // {
    //   name: 'Mobile Safari',
    //   use: { ...devices['iPhone 12'] },
    // },

    /* Test against branded browsers. */
    // {
    //   name: 'Microsoft Edge',
    //   use: { ...devices['Desktop Edge'], channel: 'msedge' },
    // },
    // {
    //   name: 'Google Chrome',
    //   use: { ...devices['Desktop Chrome'], channel: 'chrome' },
    // },
  ],

  /* Run your local dev server before starting the tests */
  // webServer: {
  //   command: 'npm run start',
  //   url: 'http://127.0.0.1:3000',
  //   reuseExistingServer: !process.env.CI,
  // },
});
