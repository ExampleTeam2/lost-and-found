import { test, expect } from '@playwright/test';
import fs from 'fs';
import 'dotenv/config';
import { Page } from 'playwright-core';

const DATA_PATH = 'scraping/data/';

const getButtonWithText = (page: Page, text: string) => {
  return page.locator('button, a').getByText(text);
}

const clickButtonWithText = async (page: Page, text: string, wait = 0) => {
  const button = getButtonWithText(page, text);
  debugger;
  if (wait) {
    await button.waitFor({ state: 'visible', timeout: wait !== -1 ? wait : undefined });
  }
  return await button.click();
}

const clickButtonIfFound = async (page: Page, text: string) => {
  const button = getButtonWithText(page, text);
  if (await button.count() > 0) {
    await button.click();
  }
}

const setCookies = async (page: Page) => {
  // File could be undefined, so check if it exists, but keep synchronous
  if (fs.existsSync(DATA_PATH + 'cookies.json')) {
    const cookies = JSON.parse(fs.readFileSync(DATA_PATH + 'cookies.json', 'utf8'));
    await page.context().addCookies(cookies);
  }
}

const checkIfLoggedIn = async (page: Page) => {
  return await page.getByText('Log in').count() === 0;
}

const getCookieBanner = (page: Page) => {
  return page.locator('#onetrust-consent-sdk');
}

// Create function which checks for a cookie banner and removes it
const removeCookieBanner = async (page: Page) => {
  // Check if element with id "onetrust-consent-sdk" exists
  const cookieBanner = getCookieBanner(page);
  await cookieBanner.waitFor({ state: 'attached', timeout: 15000 });
  await clickButtonWithText(page, 'Accept');
  await cookieBanner.evaluate((el) => el.remove());
}

const logIn = async (page: Page) => {
  // Fail the test if the environment variables are not set
  expect(process.env.GEOGUESSR_EMAIL).toBeTruthy();
  expect(process.env.GEOGUESSR_PASSWORD).toBeTruthy();
  // Will never reach this point if the environment variables are not set, but for type checking.
  if (process.env.GEOGUESSR_EMAIL === undefined || process.env.GEOGUESSR_PASSWORD === undefined) return;
  await page.click('text=Log in');
  await page.waitForTimeout(1000);
  // Wait for page change
  const email = page.locator('input[name="email"]');
  await email.waitFor({ state: 'visible' });
  await email.fill(process.env.GEOGUESSR_EMAIL);
  const password = page.locator('input[name="password"]');
  await password.waitFor({ state: 'visible' });
  await password.fill(process.env.GEOGUESSR_PASSWORD);
  await page.waitForTimeout(1000);
  await clickButtonWithText(page, 'Log in', -1);
  // Wait for logged in page to load
  await getButtonWithText(page, 'Multiplayer').waitFor({ state: 'visible' });
  // If logged in, save the cookies
  const cookies = await page.context().cookies();
  fs.writeFile(DATA_PATH + 'cookies.json', JSON.stringify(cookies), (err) => {
    if (err) console.log(err);
  });
};

// Go to "geoguessr.com", log in, play a game, take a screenshot of the viewer and save the game result into a file.
test('geoguessr.com', async ({ page }) => {
  // Total test timeout is 10 minutes
  test.setTimeout(600000);
  await setCookies(page);
  await page.goto('https://www.geoguessr.com', { timeout: 60000 });
  page.setDefaultTimeout(10000);
  // Wait for any button to be visible
  await page.locator('button, a').first().waitFor({ state: 'visible' });
  if (!(await checkIfLoggedIn(page))) {
    await removeCookieBanner(page);
    await logIn(page);
  }
  page.setDefaultTimeout(5000);
  await clickButtonWithText(page, 'Multiplayer', -1);
  await clickButtonIfFound(page, 'Got it');
  await clickButtonWithText(page, 'Unranked', -1);
  await clickButtonWithText(page, 'Countries', -1);
  // Wait for the street view to load
  const viewer = page.locator('.mapsConsumerUiSceneCoreScene__canvas');
  await viewer.waitFor({ state: 'visible', timeout: 60000 });
  await page.waitForTimeout(5000);
  await viewer?.screenshot({ path: DATA_PATH + 'location.png' });
  const result = page.getByText(' was ');
  await result.waitFor({ state: 'visible', timeout: 200000 });
  fs.writeFile(DATA_PATH + 'geoguessr.json', JSON.stringify(await result.textContent()), (err) => {
    if (err) console.log(err);
  });
});
