import { test, expect, ElementHandle } from '@playwright/test';
import fs from 'fs';
import 'dotenv/config';
import { Page } from 'playwright-core';
import { describe } from 'node:test';

const DATA_PATH = 'scraping/data/';
const LOCATION_FILE = 'geoguessr_location_';
const LOCATION_FILE_EXTENSION = '.png';
const RESULT_FILE = 'geoguessr_result_';
const RESULT_FILE_EXTENSION = '.json';
const MAX_ROUNDS = 15;
const MAX_GAMES = 100;
const MAX_MINUTES = 1440;
const NUMBER_OF_INSTANCES = process.env.CI ? 5 : 1;
const STAGGER_INSTANCES = 30000;

let logProgressInterval: ReturnType<typeof setInterval> | undefined;

// Log a message, print a dot (on the same line) every 10 seconds in a background process to get progress indication until another message is logged
const log = (message: string, i?: string) => {
  console.log((i ? i + ' - ' : '') + message);
  if (!i) {
    if (logProgressInterval) {
      clearInterval(logProgressInterval);
    }
    logProgressInterval = setInterval(() => process.stdout.write('.'), 10000);
  }
}

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

const logIn = async (page: Page, identifier?: string) => {
  log('Log in needed', identifier);
  // Fail the test if the environment variables are not set
  expect(process.env.GEOGUESSR_EMAIL).toBeTruthy();
  expect(process.env.GEOGUESSR_PASSWORD).toBeTruthy();
  // Will never reach this point if the environment variables are not set, but for type checking.
  if (process.env.GEOGUESSR_EMAIL === undefined || process.env.GEOGUESSR_PASSWORD === undefined) return;
  log('Navigating to login page', identifier);
  await page.getByText('Log in').click();
  await page.waitForTimeout(1000);
  log('Entering credentials', identifier);
  // Wait for page change
  const email = page.locator('input[name="email"]');
  await email.waitFor({ state: 'visible' });
  await email.fill(process.env.GEOGUESSR_EMAIL);
  const password = page.locator('input[name="password"]');
  await password.waitFor({ state: 'visible' });
  await password.fill(process.env.GEOGUESSR_PASSWORD);
  await page.waitForTimeout(1000);
  log('Attempting log in', identifier);
  await clickButtonWithText(page, 'Log in', -1);
  // Wait for logged in page to load
  await getButtonWithText(page, 'Multiplayer').waitFor({ state: 'visible' });
  log('Logged in successfully', identifier);
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

const collectGuesses = (page: Page) => {
  let intervalId: ReturnType<typeof setInterval> | undefined;

  const data: Record<number, { incorrect: string[] }> = {};

  let index = 0;

  const task = async () => {
    try {
      // Get element 'Already made guesses' if it exists (without waiting), then get its parent and look for the alt of all img contained somewhere within it (can be nested deeper)
      const incorrectGuessesHeading = (await page.locator('text="Already made guesses"').all())?.[0];
      if (incorrectGuessesHeading) {
        const incorrect = (await incorrectGuessesHeading.evaluate((el): string[] => {
          if (!el.parentElement) {
            return [];
          }
          return Array.from(el.parentElement.querySelectorAll('img')).map(img => img.getAttribute('alt')).filter(text => text) as string[];
        }));
        index++;
        data[index] = { incorrect };
      }
    } catch (e) {
      console.error(e);
    }
  };

  intervalId = setInterval(task, 1000);

  return () => {
    clearInterval(intervalId);
    return data;
  };
}

const getUsers = (page: Page) => {
  // Get links with URL like /user/...
  return page.locator('a[href^="/user/"]');
};

const getCoordinates = async (page: Page): Promise<[number, number] | undefined> => {
  // Get link with URL like https://maps.google.com/maps?ll=<lat>,<lon>&... (only if it exists, without waiting)
  // Make sure the coordinates are not 0,0, if yes then try another link
  const links = await page.locator('a[href^="https://maps.google.com/maps?ll="]').all().then(result => Promise.all(result.map(link => link.getAttribute('href'))));
  for (const link of links) {
    const coordinates = link?.match(/ll=([\d.-]+),([\d.-]+)/);
    if (coordinates) {
      const lat = parseFloat(coordinates[1]);
      const lon = parseFloat(coordinates[2]);
      if (lat !== 0 || lon !== 0) {
        return [lat, lon];
      }
    }
  }
  return undefined;
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

const round = async(page: Page, gameId: string, roundNumber: number, identifier?: string) => {
  log('Starting round - ' + roundNumber, identifier);
  // Wait for the street view to load
  const viewer = page.locator('.mapsConsumerUiSceneCoreScene__canvas').first();
  await viewer.waitFor({ state: 'visible' });
  const startTime = Date.now();
  const people = await getUsers(page).count();
  const stopCollectingGuesses = collectGuesses(page);
  await page.waitForTimeout(5000);
  const css = await injectCss(page, hideEverythingElseCss);
  // Move the mouse to the top right corner to hide the UI (not top left), get the page size dynamically
  const pageWidth = await page.evaluate(() => window.innerWidth);
  await page.mouse.move(pageWidth - 1, 1);
  await viewer?.screenshot({ path: DATA_PATH + LOCATION_FILE + gameId + '_' + roundNumber + LOCATION_FILE_EXTENSION });
  await removeElement(css);
  const result = page.getByText('right answer was');
  await result.waitFor({ state: 'visible', timeout: 200000 });
  const duration = (Date.now() - startTime) / 1000;
  const guesses = await stopCollectingGuesses();
  const coordinates = await getCoordinates(page);
  const resultText = await result.textContent();
  // The sentence is like "[.]?[...] right answer was [in | indeed | actually | ...] [country].[...][.]?", parse the country.
  const country = resultText?.split('right answer was')[1].split('.')[0].trim();
  log('It was ' + country, identifier);
  const resultJson = {
    country,
    coordinates,
    guesses,
    people,
    duration
  }
  fs.writeFile(DATA_PATH + RESULT_FILE + gameId + '_' + roundNumber + RESULT_FILE_EXTENSION, JSON.stringify(resultJson), (err) => {
    if (err) console.log(err);
  });
};

const gameIds: string[] = [];

const game = async (page: Page, identifier?: string) => {
  await page.getByText('3 Lives').waitFor({ state: 'visible' });
  await page.getByText('Game starting in').waitFor({ state: 'visible', timeout: 60000 });
  await page.getByText('Game starting in').waitFor({ state: 'hidden', timeout: 60000 });
  const gameId = page.url().split('/').pop() ?? 'no_id_' + randomUUID();
  if (gameIds.includes(gameId)) {
    return true; 
  }
  gameIds.push(gameId);
  log('Starting game - ' + gameId, identifier);
  // Get the game ID from the URL (https://www.geoguessr.com/de/battle-royale/<ID>)
  let rounds = 0;
  await round(page, gameId, rounds, identifier);
  await page.waitForTimeout(1000);
  if (await clickButtonIfFound(page, 'Spectate')) {
    rounds++;
    await page.getByText('Next round starts in').waitFor({ state: 'visible' });
    await page.getByText('Next round starts in').waitFor({ state: 'hidden', timeout: 20000 });
    await round(page, gameId, rounds, identifier);
    rounds++;
    // Remove footer to improve vision and avoid second "Play again" button
    await page.locator('footer').evaluate((el) => el.remove());
    while (rounds < MAX_ROUNDS && await page.getByText('Next round starts').count() > 0) {
      await page.getByText('Next round starts in').waitFor({ state: 'visible' });
      await page.getByText('Next round starts in').waitFor({ state: 'hidden', timeout: 20000 });
      await round(page, gameId, rounds, identifier);
      rounds++;
      await page.waitForTimeout(1000);
    }
  }
}

const play = async (page: Page, identifier: string, i: number, wait: number = i) => {
  test.setTimeout(60000 * MAX_MINUTES);
  await page.waitForTimeout(STAGGER_INSTANCES * wait);
  log('Starting geoguessr', identifier);
  await setCookies(page);
  await page.goto('https://www.geoguessr.com', { timeout: 60000 });
  page.setDefaultTimeout(10000);
  // Wait for any button to be visible
  await page.locator('button, a').first().waitFor({ state: 'visible' });
  if (!(await checkIfLoggedIn(page))) {
    await removeCookieBanner(page);
    await logIn(page, identifier);
  } else {
    log('Already logged in', identifier);
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
  const restart = await game(page, identifier);
  if (restart) {
    await page.waitForTimeout(STAGGER_INSTANCES);
    throw new Error('Double-joined game, restarting');
  }
  await page.waitForTimeout(3000);
  while (games < MAX_GAMES && await clickButtonIfFound(page, 'Play again')) {
    const restart = await game(page, identifier);
    if (restart) {
      await page.waitForTimeout(STAGGER_INSTANCES);
      throw new Error('Double-joined game, restarting');
    }
    games++;
    await page.waitForTimeout(3000);
  }
}

describe('Geoguessr', () => {
  for (let i = 0; i < NUMBER_OF_INSTANCES; i++) {
    const identifier = (NUMBER_OF_INSTANCES > 1 ? String(i + 1) : '');
    // Go to "geoguessr.com", log in, play a game, take a screenshot of the viewer and save the game result into a file.
    test('play countries battle royale' + (identifier ? ' - ' + identifier : ''), async ({ page }) => {
      await play(page, identifier, i);
    });    
  }
});
