import { test, expect } from '@playwright/test';
import fs from 'fs';
import 'dotenv/config';
import { Page } from 'playwright-core';

const DATA_PATH = 'scraping/data/';

// Go to "hslu.ch", navigate to "Informatik", print the text, look for a field of study containing "Artificial Intelligence" and take a screenshot of that field of study.
test.skip('hslu.ch', async ({ page }) => {
  await page.goto('https://www.hslu.ch');
  await page.click('text=Informatik');
  const text = await page.textContent('body');
  // Format into json of sub sections and write into a file async (log if error)
  fs.writeFile(DATA_PATH + 'hslu.json', JSON.stringify(text), (err) => {
    if (err) console.log(err);
  });
  const ai = await page.locator('text("Artificial Intelligence")');
  await expect(ai).toBeVisible();
  await ai?.screenshot({ path: DATA_PATH + 'ai.png' });
});

const getButtonWithText = async (page: Page, text: string) => {
  return await page.locator('button, a', { hasText: text });
}

// Create function which checks for a cookie banner and removes it
const removeCookies = async (page: Page) => {
  // Check if element with id "onetrust-consent-sdk" exists
  const cookieBanner = await page.locator('#onetrust-consent-sdk');
  await cookieBanner.waitFor({ state: 'attached', timeout: 15000 });
  const acceptButton = await getButtonWithText(page, 'Accept');
  await acceptButton?.click();
  await cookieBanner.evaluate((el) => el.remove());
}

const logIn = async (page: Page) => {
  // Fail the test if the environment variables are not set
  expect(process.env.GEOGUESSR_EMAIL).toBeTruthy();
  expect(process.env.GEOGUESSR_PASSWORD).toBeTruthy();
  // Will never reach this point if the environment variables are not set, but for type checking.
  if (process.env.GEOGUESSR_EMAIL === undefined || process.env.GEOGUESSR_PASSWORD === undefined) return;
  await page.click('text=Log in');
  // Wait for page change
  await page.waitForSelector('input[name="email"]', { timeout: 10000 });
  await page.fill('input[name="email"]', process.env.GEOGUESSR_EMAIL);
  await page.waitForSelector('input[name="password"]', { timeout: 10000 });
  await page.fill('input[name="password"]', process.env.GEOGUESSR_PASSWORD);
  const logInButton = await getButtonWithText(page, 'Log in');
  await expect(logInButton).toBeVisible();
  await logInButton?.click();
  // Wait for logged in page to load
  await page.waitForSelector('text=Multiplayer', { timeout: 10000 });
};

// Go to "geoguessr.com", log in, play a game, take a screenshot of the viewer and save the game result into a file.
test('geoguessr.com', async ({ page }) => {
  await page.goto('https://www.geoguessr.com');
  await removeCookies(page);
  test.setTimeout(5000);
  await logIn(page);
  await page.click('text=Multiplayer');
  // Click unranked and set the timeout to 60 seconds
  await page.click('text=Unranked');
  // Wait for the map to load
  await page.waitForSelector('div.leaflet-container', { timeout: 30000 });
  const viewer = await page.locator('.mapsConsumerUiSceneCoreScene__canvas');
  await expect(viewer).toBeVisible();

  await viewer?.screenshot({ path: DATA_PATH + 'map.png' });
  const result = page.locator('text("It was")');
  await result.waitFor({ state: 'visible', timeout: 200000 });
  fs.writeFile(DATA_PATH + 'geoguessr.json', JSON.stringify(await result.textContent()), (err) => {
    if (err) console.log(err);
  });
});
