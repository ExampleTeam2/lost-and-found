import { test, expect, ElementHandle, Locator } from '@playwright/test';
import fs from 'fs';
import 'dotenv/config';
import { Page } from 'playwright-core';
import { describe } from 'node:test';
import { DATA_PATH, LOCATION_FILE, LOCATION_FILE_EXTENSION, MAX_GAMES, MAX_MINUTES, MAX_ROUNDS, MODE, NUMBER_OF_INSTANCES, RESULT_FILE, RESULT_FILE_EXTENSION, SINGLEPLAYER_WIDTH, STAGGER_INSTANCES, TEMP_PATH, getTimestampString } from '../playwright_base_config';

let logProgressInterval: ReturnType<typeof setInterval> | undefined;

const error = (message: string, i?: string) => {
  console.error((i ? i + ' - ' : '') + message);
}

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

const getButtonWithText = (page: Page, text: string, only = false) => {
  return page.locator('button, a').getByText(text, { exact: only });
}

const clickButtonWithText = async (page: Page, text: string, wait = 0) => {
  const button = getButtonWithText(page, text);
  debugger;
  if (wait) {
    await button.waitFor({ state: 'visible', timeout: wait !== -1 ? wait : undefined });
  }
  return await button.click();
}

const clickButtonIfFound = async (page: Page, text: string, only = false, first = false, last = false) => {
  let button = getButtonWithText(page, text, only);
  if (first) {
    button = button.first();
  }
  if (last) {
    button = button.last();
  }
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
  fs.writeFile(DATA_PATH + 'cookies.json', JSON.stringify(cookies), e => {
    const timestamp = getTimestampString();
    if (e) {
      error(`Error occurred while saving cookies at ${timestamp}:`, identifier);
      console.error(e);
    };
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

const collectGuesses = (page: Page, identifier?: string) => {
  let intervalId: ReturnType<typeof setInterval> | undefined;

  const data: Record<number, { incorrect: string[] }> = {};

  let index = 0;
  let cleared = false;

  const task = async () => {
    try {
      // Get element 'Already made guesses' if it exists (without waiting), then get its parent and look for the alt of all img contained somewhere within it (can be nested deeper)
      const incorrectGuessesHeadings = await page.getByText('Already made guesses').all();
      const incorrectGuessesHeading = incorrectGuessesHeadings[0];
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
      if (typeof e === 'object' && e instanceof Error && (e.message.includes('Test ended') || e.message.includes('Target crashed'))) {
        clearInterval(intervalId);
        cleared = true;
      } else if (!cleared) {
        const timestamp = getTimestampString();
        error(`Error occurred in subtask 'collectGuesses' at ${timestamp}:`, identifier);
        console.error(e);
      }
    }
  };

  intervalId = setInterval(task, 1000);

  return () => {
    clearInterval(intervalId);
    cleared = true;
    return data;
  };
}

const getUsers = (page: Page) => {
  // Get links with URL like /user/...
  return page.locator('a[href^="/user/"]');
};

const getCoordinates = async (page: Page, force = false): Promise<[number, number] | undefined> => {
  // Get link with URL like https://maps.google.com/maps?ll=<lat>,<lon>&... (only if it exists, without waiting)
  // Make sure the coordinates are not 0,0, if yes then try another link
  const linksLocator = page.locator('a[href^="https://maps.google.com/maps?ll="]');
  // Await if force is enabled.
  if (force) {
    await linksLocator.waitFor({ state: 'visible' });
  }
  const links = await linksLocator.all().then(result => Promise.all(result.map(link => link.getAttribute('href'))));
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

const getMapFromElement = async (page: Page, element: Locator) => {
  // Go up chain of elements until element has a height of more than 50, you have to use evaluate to get the parent
  return (await element.evaluateHandle((el) => {
    let parent = el.parentElement;
    while (parent && parent.clientHeight <= 50) {
      parent = parent.parentElement;
    }
    return parent;
  }))?.asElement();
};

const guess = async (page: Page, force = true) => {
  // Find button with title "Keyboard shortcuts" and Img in it
  const button = page.locator('button[title="Keyboard shortcuts"] img');
  // Await if force is enabled
  if (force) {
    await button.waitFor({ state: 'visible' });
  }
  // Make sure at least one button is found
  if (await button.count() > 0) {
    // Go up chain of elements until element has a height of more than 50, you have to use evaluate to get the parent
    const mapElement = await getMapFromElement(page, button);
    if (!mapElement) {
      if (force) {
        expect(mapElement).toBeTruthy();
      }
      return;
    }
    // Move the mouse roughly to the center of the element
    await mapElement.hover();
    // Wait for the button with an image with the alt text of "Sticky map" to be visible
    await page.locator('button img[alt="Sticky map"]').waitFor({ state: 'visible' });
    // Wait for 1 second
    await page.waitForTimeout(1000);
    // Find span with text "Map data ©" in it
    const mapDataButton = page.locator('span').getByText('Map data ©');
    // Get the map element again, because it might have changed
    const mapElementExpanded = await getMapFromElement(page, mapDataButton);
    if (!mapElementExpanded) {
      if (force) {
        expect(mapElementExpanded).toBeTruthy();
      }
      return;
    }
    // Get the dimensions of the map element
    const dimensions = await mapElementExpanded.boundingBox();
    if (!dimensions) {
      if (force) {
        expect(dimensions).toBeTruthy();
      }
      return;
    }
     // Randomly move the mouse to a point within the map element, make sure to keep 40 pixels from the edge
    const gapFromEdge = 40;
    const pointX = dimensions.x + gapFromEdge + Math.random() * (dimensions.width - (gapFromEdge * 2)), pointY = dimensions.y + gapFromEdge + Math.random() * (dimensions.height - (gapFromEdge * 2));
    await page.mouse.move(pointX, pointY);
    // Wait for 1 second
    await page.waitForTimeout(1000);
    // Click the mouse
    await page.mouse.click(pointX, pointY);
    // Click the guess button
    await clickButtonWithText(page, 'Guess');
  }
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

const roundStartAndCapture = async<const T>(page: Page, mode: typeof MODE, gameId: string, roundNumber: number, identifier?: string, additional: (page: Page, gameId: string, roundNumber: number, identifier?: string) => T = (() => undefined) as () => T): Promise<[number, Awaited<T>]> => {
  log('Starting round - ' + roundNumber, identifier);
  // Wait for the street view to load
  const viewer = page.locator('.mapsConsumerUiSceneCoreScene__canvas').first();
  await viewer.waitFor({ state: 'visible', timeout: 10000 });
  const startTime = Date.now();
  const additionalResults = await additional?.(page, gameId, roundNumber, identifier);
  await page.waitForTimeout(5000);
  if (mode === 'singleplayer') {
    // Move the mouse over the text "World" to hide the UI manually (not just hover)
    const world = page.getByText('World', { exact: true });
    // Get the location of the text "World"
    const worldLocation = await world.boundingBox();
    // Move the mouse to middle of that text area
    if (worldLocation) {
      await page.mouse.move(worldLocation.x + worldLocation.width / 2, worldLocation.y + worldLocation.height / 2);
          
    }
  }
  const css = await injectCss(page, hideEverythingElseCss);
  if (mode !== 'singleplayer') {
    // Move the mouse to the top right corner to hide the UI (not top left), get the page size dynamically
    const pageWidth = await page.evaluate(() => window.innerWidth);
    await page.mouse.move(pageWidth - 1, 1);
  }
  await viewer?.screenshot({ path: DATA_PATH + LOCATION_FILE + mode + '_' + gameId + '_' + roundNumber + LOCATION_FILE_EXTENSION });
  await removeElement(css);
  return [startTime, additionalResults];
};

const roundEndAndSave = (mode: typeof MODE, result: unknown, gameId: string, roundNumber: number, identifier?: string) => {
  fs.writeFile(DATA_PATH + RESULT_FILE + mode + '_' + gameId + '_' + roundNumber + RESULT_FILE_EXTENSION, JSON.stringify(result), e => {
    const timestamp = getTimestampString();
    if (e) {
      error(`Error occurred while saving game results at ${timestamp}:`, identifier);
      console.error(e);
    };
  });
};

const roundMultiplayer = async(page: Page, gameId: string, roundNumber: number, identifier?: string) => {
  const [startTime, additionalResults] = await roundStartAndCapture(page, 'multiplayer', gameId, roundNumber, identifier, async (page, gameId, roundNumber, identifier) => {
    const people = await getUsers(page).count();
    const stopCollectingGuesses = collectGuesses(page, identifier);
    return [people, stopCollectingGuesses];
  });
  const [people, stopCollectingGuesses] = additionalResults;
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
  roundEndAndSave('multiplayer', resultJson, gameId, roundNumber, identifier);
};

const roundSingleplayer = async(page: Page, gameId: string, roundNumber: number, identifier?: string) => {
  const [startTime] = await roundStartAndCapture(page, 'singleplayer', gameId, roundNumber, identifier);
  await guess(page);
  await getButtonWithText(page, 'Next').or(await getButtonWithText(page, 'View results')).waitFor({ state: 'visible', timeout: 10000 });
  const duration = (Date.now() - startTime) / 1000;
  const coordinates = await getCoordinates(page, true);
  log('It was ' + coordinates, identifier);
  const resultJson = {
    coordinates,
    duration
  }
  roundEndAndSave('singleplayer', resultJson, gameId, roundNumber, identifier);
};

const gameStart = async (page: Page, mode: typeof MODE, waitText: string, waitTime: number, i: number, identifier?: string) => {
  await page.getByText(waitText).or(page.getByText('Rate limit')).nth(0).waitFor({ state: 'visible', timeout: 60000 });
  if (await page.getByText('Rate limit').count() > 0) {
    log('Rate-limited', identifier);
    fs.appendFileSync(TEMP_PATH + 'rate-limits', i + '\n');
    await page.waitForTimeout(STAGGER_INSTANCES);
    await (mode === 'singleplayer' ? playSingleplayer(page, i, identifier) : playMultiplayer(page, i, identifier));
    return;
  }
  await page.getByText(waitText).nth(0).waitFor({ state: 'hidden', timeout: waitTime });
  // Get the game ID from the URL (https://www.geoguessr.com/de/battle-royale/<ID>)
  const gameId = page.url().split('/').pop() ?? 'no_id_' + randomUUID();
  if (fs.readFileSync(TEMP_PATH + mode + '-games', 'utf8')?.split(/\n/g)?.includes(gameId)) {
    log('Double-joined game', identifier);
    fs.appendFileSync(TEMP_PATH + 'double-joins', i + '\n');
    await page.waitForTimeout(STAGGER_INSTANCES);
    await (mode === 'singleplayer' ? playSingleplayer(page, i, identifier) : playMultiplayer(page, i, identifier));
    return;
  }
  fs.appendFileSync(TEMP_PATH + mode + '-games', gameId + '\n');
  log('Starting game - ' + gameId, identifier);
  return gameId;
};

const gameMultiplayer = async (page: Page, i: number, identifier?: string) => {
  const gameId = await gameStart(page, 'multiplayer', 'Game starting in', 60000, i, identifier);
  if (!gameId) {
    return;
  }
  let rounds = 0;
  await roundMultiplayer(page, gameId, rounds, identifier);
  await page.waitForTimeout(1000);
  if (await clickButtonIfFound(page, 'Spectate')) {
    rounds++;
    await page.getByText('Next round starts in').waitFor({ state: 'visible' });
    await page.getByText('Next round starts in').waitFor({ state: 'hidden', timeout: 20000 });
    await roundMultiplayer(page, gameId, rounds, identifier);
    rounds++;
    // Remove footer to improve vision and avoid second "Play again" button
    await page.locator('footer').evaluate((el) => el.remove());
    while (rounds < MAX_ROUNDS && await page.getByText('Next round starts').count() > 0) {
      await page.getByText('Next round starts in').waitFor({ state: 'visible' });
      await page.getByText('Next round starts in').waitFor({ state: 'hidden', timeout: 20000 });
      await roundMultiplayer(page, gameId, rounds, identifier);
      rounds++;
      await page.waitForTimeout(1000);
    }
  }
  if (fs.readFileSync(TEMP_PATH + 'stop', 'utf8') === 'true') {
    process.exit(1);
  }
};

const gameSingleplayer = async (page: Page, i: number, identifier?: string) => {
  const gameId = await gameStart(page, 'singleplayer', 'Loading', 10000, i, identifier);
  if (!gameId) {
    return;
  }
  let rounds = 0;
  await roundSingleplayer(page, gameId, rounds, identifier);
  rounds++;
  await page.waitForTimeout(1000);
  while (rounds < MAX_ROUNDS && await clickButtonIfFound(page, 'Next')) {
    await roundSingleplayer(page, gameId, rounds, identifier);
    rounds++;
    await page.waitForTimeout(1000);
  }
  const resultsButton = getButtonWithText(page, 'View results')
  await resultsButton.click();
  await resultsButton.waitFor({ state: 'hidden', timeout: 10000 });
  if (fs.readFileSync(TEMP_PATH + 'stop', 'utf8') === 'true') {
    process.exit(1);
  }
};

const playStart = async (page: Page, i: number, identifier?: string) => {
  if (fs.readFileSync(TEMP_PATH + 'stop', 'utf8') === 'true') {
    process.exit(1);
  }
  await page.waitForTimeout(STAGGER_INSTANCES * (fs.readFileSync(TEMP_PATH + 'initial', 'utf8') === 'true' ? i : 0));
  if (i === NUMBER_OF_INSTANCES - 1) {
    fs.writeFileSync(TEMP_PATH + 'initial', 'false');
  }
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
};
const playGame = async (page: Page, mode: typeof MODE, i: number, identifier?: string) => {
  let games = 1;
  await (mode ? gameSingleplayer(page, i, identifier) : gameMultiplayer(page, i, identifier));
  await page.waitForTimeout(1000);
  while (games < MAX_GAMES && await clickButtonIfFound(page, 'Play again')) {
    await (mode ? gameSingleplayer(page, i, identifier) : gameMultiplayer(page, i, identifier));
    games++;
    await page.waitForTimeout(3000);
  }
  // Retry if ended early
  if (games !== MAX_GAMES) {
    log('Could not start next game', identifier);
    fs.appendFileSync(TEMP_PATH + 'other-restarts', i + '\n');
    await page.waitForTimeout(STAGGER_INSTANCES);
    await  (mode ? playSingleplayer(page, i, identifier) : playMultiplayer(page, i, identifier));
  }
};

const playMultiplayer = async (page: Page, i: number, identifier?: string) => {
  await playStart(page, i, identifier);
  await clickButtonWithText(page, 'Multiplayer', -1);
  await getButtonWithText(page, 'Got it').or(page.getByText('final results')).or(getButtonWithText(page, 'Unranked')).nth(0).waitFor({ state: 'visible', timeout: 10000 });
  await page.waitForTimeout(3000);
  await clickButtonIfFound(page, 'Got it');
  await clickButtonIfFound(page, '×', true);
  await getButtonWithText(page, 'Got it').waitFor({ state: 'hidden' });
  await page.getByText('final results').waitFor({ state: 'hidden' });
  await clickButtonWithText(page, 'Unranked', -1);
  await clickButtonWithText(page, 'Countries', -1);
  await playGame(page, 'multiplayer', i, identifier);
};

const playSingleplayer = async (page: Page, i: number, identifier?: string) => {
  await playStart(page, i, identifier);
  await clickButtonWithText(page, 'Classic Maps', -1);
  await page.getByText('World', { exact: true }).first().waitFor({ state: 'visible', timeout: 10000 });
  await page.waitForTimeout(1000);
  await clickButtonIfFound(page, 'Play', true, false, true);
  await playGame(page, 'singleplayer', i, identifier);
};

describe('Geoguessr', () => {
  for (let i = 0; i < NUMBER_OF_INSTANCES; i++) {
    const identifier = (NUMBER_OF_INSTANCES > 1 ? String(i + 1) : '');
    // Go to "geoguessr.com", log in, play a game, take a screenshot of the viewer and save the game result into a file.
    test('play ' + (MODE === 'singleplayer' ? 'world' : 'countries battle royale') + (identifier ? ' - ' + identifier : ''), async ({ page }) => {
      if (fs.readFileSync(TEMP_PATH + 'stop', 'utf8') === 'true') {
        process.exit(1);
      }
      // Set viewport size in singleplayer mode to align with multiplayer mode
      if (MODE === 'singleplayer') {
        page.setViewportSize({ width: SINGLEPLAYER_WIDTH, height: page.viewportSize()?.height ?? 0});
      }
      try {
        test.setTimeout(60000 * MAX_MINUTES);
        await (MODE === 'singleplayer' ? playSingleplayer(page, i, identifier) : playMultiplayer(page, i, identifier));
      } catch (e: unknown) {
        const timestamp = getTimestampString();
        // If messages includes 'Target crashed', exit program, otherwise log an error message that an Error occurred in this instance at this time and rethrow
        if (typeof e === 'object' && e instanceof Error && e.message.includes('Target crashed')) {
          fs.appendFileSync(TEMP_PATH + 'crashes', timestamp + '\n');
          fs.writeFileSync(TEMP_PATH + 'stop', 'true');
          error(`Crash occurred at ${timestamp}, stopping:`, identifier);
          console.error(e);
          process.exit(1);
        } else {
          error(`Error occurred at ${timestamp}:`, identifier);
          console.error(e);
          throw e;
        }
      }
    });
  }
});
