import { test, expect, ElementHandle } from '@playwright/test';
import fs from 'fs';
import 'dotenv/config';
import { Page } from 'playwright-core';

const DATA_PATH = 'scraping/data/';
const LOCATION_FILE = 'geoguessr_location_';
const LOCATION_FILE_EXTENSION = '.png';
const RESULT_FILE = 'geoguessr_result_';
const RESULT_FILE_EXTENSION = '.json';
const MAX_ROUNDS = 10;
const MAX_GAMES = 5;
const MAX_MINUTES = 10;

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
    return true;
  }
  return false;
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

// Inject css with a way to remove it again
const injectCss = async (page: Page, css: string) => {
  return await page.addStyleTag({ content: css });
}

const removeElement = async (element: ElementHandle<Node>) => {
  return await element.evaluate((el) => el.parentNode?.removeChild(el));
}

// Random 10 character UUID
const randomUUID = () => {
  return 'xxxxxxxxxx'.replace(/x/g, () => Math.floor(Math.random() * 16).toString(16));
};

const viewerSelector = '.mapsConsumerUiSceneCoreScene__canvas';
// Use :has() to also exclude other parent elements
const hideEverythingElseCss = `
*:not(:has(${viewerSelector})) {
  display: none !important;
}
${viewerSelector} {
  display: block !important;
}
`;

const round = async(page: Page) => {
  const roundId = randomUUID();
  await page.waitForTimeout(1000);
  // Wait for the street view to load
  const viewer = page.locator('.mapsConsumerUiSceneCoreScene__canvas').first();
  await viewer.waitFor({ state: 'visible', timeout: 60000 });
  await page.waitForTimeout(10000);
  const css = await injectCss(page, hideEverythingElseCss);
  // Move the mouse to the top right corner to hide the UI (not top left), get the page size dynamically
  const pageWidth = await page.evaluate(() => window.innerWidth);
  await page.mouse.move(pageWidth - 1, 1);
  await page.waitForTimeout(1000);
  await viewer?.screenshot({ path: DATA_PATH + LOCATION_FILE + roundId + LOCATION_FILE_EXTENSION });
  await page.waitForTimeout(1000);
  await removeElement(css);
  const result = page.getByText('right answer was');
  await result.waitFor({ state: 'visible', timeout: 200000 });
  const resultText = await result.textContent();
  // The sentence is like "[.]?[...] right answer was [in | indeed | actually | ...] [country].[...][.]?", parse the country.
  const country = resultText?.split('right answer was')[1].split('.')[0].trim();
  fs.writeFile(DATA_PATH + RESULT_FILE + roundId + RESULT_FILE_EXTENSION, JSON.stringify(country), (err) => {
    if (err) console.log(err);
  });
};

const game = async (page: Page) => {
  let rounds = 1;
  await round(page);
  await page.waitForTimeout(1000);
  if (await clickButtonIfFound(page, 'Spectate')) {
    await round(page);
    rounds++;
    while (rounds < MAX_ROUNDS && await page.getByText('Next round starts').count() > 0) {
      await page.waitForTimeout(10000);
      await round(page);
      rounds++;
      await page.waitForTimeout(1000);
    }
  }
}

// Go to "geoguessr.com", log in, play a game, take a screenshot of the viewer and save the game result into a file.
test('play country battle royale', async ({ page }) => {
  // Total test timeout is 10 minutes
  test.setTimeout(60000 * MAX_MINUTES);
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
  await getButtonWithText(page, 'Got it').or(getButtonWithText(page, 'Multiplayer')).nth(0).waitFor({ state: 'visible' });
  await page.waitForTimeout(1000);
  await clickButtonIfFound(page, 'Got it');
  await getButtonWithText(page, 'Got it').waitFor({ state: 'hidden' });
  await clickButtonWithText(page, 'Unranked', -1);
  await clickButtonWithText(page, 'Countries', -1);
  let games = 1;
  await game(page);
  await page.waitForTimeout(3000);
  while (games < MAX_GAMES && await clickButtonIfFound(page, 'Play again')) {
    await game(page);
    games++;
    await page.waitForTimeout(3000);
  }
});
