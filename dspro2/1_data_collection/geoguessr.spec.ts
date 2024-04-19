import { test, expect, ElementHandle, Locator } from '@playwright/test';
import fs from 'fs';
import 'dotenv/config';
import { Page } from 'playwright-core';
import { describe } from 'node:test';
import { DATA_PATH, GAMES, LOCATION_FILE, LOCATION_FILE_EXTENSION, MAX_GAMES, MAX_MINUTES, MAX_ROUNDS, MODE, NUMBER_OF_INSTANCES, RESULT_FILE, RESULT_FILE_EXTENSION, SINGLEPLAYER_WIDTH, STAGGER_INSTANCES, TEMP_PATH, getTimestampString } from './playwright_base_config';

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
  return page.locator('#onetrust-consent-sdk').or(page.locator('#snigel-cmp-framework')).first();
}

// Create function which checks for a cookie banner and removes it
const removeCookieBanner = async (page: Page) => {
  // Check if element with id "onetrust-consent-sdk" exists
  const cookieBanner = getCookieBanner(page);
  await cookieBanner.waitFor({ state: 'attached', timeout: 15000 });
  await getButtonWithText(page, 'Accept').or(page.getByText('Accept all')).first().click();
  page.waitForTimeout(1000);
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
      if (typeof e === 'object' && e instanceof Error && (e.message.includes('Test ended') || e.message.includes('Target crashed') || e.message.includes('exited unexpectedly'))) {
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

const getButtonInMap = (page: Page) => {
  // Find button with title "Keyboard shortcuts" and Img in it or span with text "Map data ©"
  return page.locator('button[title="Keyboard shortcuts"] img').or(page.locator('span').getByText('Map data ©')).nth(0);
}

const guess = async (page: Page, force = true) => {
  // Find button with title "Keyboard shortcuts" and Img in it or span with text "Map data ©"
  const button = getButtonInMap(page);
  // Await if force is enabled
  if (force) {
    await button.waitFor({ state: 'visible', timeout: 10000 });
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
    // Find button with title "Keyboard shortcuts" and Img in it or span with text "Map data ©"
    const expandedButton = getButtonInMap(page);
    // Get the map element again, because it might have changed
    const mapElementExpanded = await getMapFromElement(page, expandedButton);
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

const sidebarId = 'right-bar';

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

const addSidebarCss = `
body > div {
  margin-right: 1px !important;
}

#${sidebarId} {
  display: block !important;
  background-color: white !important;
  position: fixed !important;
  width: 1px !important;
  top: 0 !important;
  bottom: 0 !important;
  right: 0 !important;
  margin: 0 !important;
}
`;

type ElementPair = [ElementHandle, Locator];

const addSidebar = async (page: Page): Promise<ElementPair> => {
  const sidebarCss = await injectCss(page, addSidebarCss);
  // Add fixed sidebar to the page so it can be removed again
  // The mouse will be rested there
  // First remove any potential previous sidebars
  await Promise.all((await page.locator('#' + sidebarId).all()).map(async (el) => {
    const element = await el.elementHandle();
    if (element) {
      await removeElement(element);
    }
  }));
  await page.evaluate(sidebarId => {
    const rightBar = document.createElement('div');
    rightBar.id = sidebarId;
    document.body.appendChild(rightBar);
  }, sidebarId);
  const sidebar = page.locator('#' + sidebarId).first();
  // Wait for the sidebar to be visible
  await sidebar.waitFor({ state: 'visible' });
  return [sidebarCss, sidebar];
};

const removeSidebar = async (page: Page, sidebar: ElementPair) => {
  await removeElement(sidebar[0]);
  const sidebarHandle = await sidebar[1].elementHandle();
  expect(sidebarHandle).toBeTruthy();
  if (sidebarHandle) {
    await removeElement(sidebarHandle);
  }
  // Wait for the sidebar to be removed
  await sidebar[1].waitFor({ state: 'hidden' });
};

const roundStartAndCapture = async<const T>(page: Page, mode: typeof MODE, gameId: string, roundNumber: number, identifier?: string, additional: (page: Page, gameId: string, roundNumber: number, identifier?: string) => T = (() => undefined) as () => T): Promise<[number, Awaited<T>, ElementPair?]> => {
  log('Starting round - ' + roundNumber, identifier);
  // Wait for the street view to load
  const viewer = page.locator('.mapsConsumerUiSceneCoreScene__canvas').first();
  await viewer.waitFor({ state: 'visible', timeout: 10000 });
  const startTime = Date.now();
  const sidebar = mode === 'singleplayer' ? await addSidebar(page) : undefined;
  const additionalResults = await additional?.(page, gameId, roundNumber, identifier);
  await page.waitForTimeout(5000);
  const css = await injectCss(page, hideEverythingElseCss);
  if (sidebar) {
    // Move the mouse over the sidebar on the right (middle and fully to the right)
    const pageHeight = await page.evaluate(() => window.innerHeight);
    await page.mouse.move(SINGLEPLAYER_WIDTH - 1, pageHeight / 2);
  } else if (mode !== 'singleplayer') {
    // Move the mouse to the top right corner to hide the UI (not top left), get the page size dynamically
    const pageWidth = await page.evaluate(() => window.innerWidth);
    await page.mouse.move(pageWidth - 1, 1);
  }
  await viewer?.screenshot({ path: DATA_PATH + LOCATION_FILE + mode + '_' + gameId + '_' + roundNumber + LOCATION_FILE_EXTENSION });
  await removeElement(css);
  return [startTime, additionalResults, sidebar];
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
  const [startTime, _, sidebar] = await roundStartAndCapture(page, 'singleplayer', gameId, roundNumber, identifier);
  await guess(page);
  await getButtonWithText(page, 'Next').or(await getButtonWithText(page, 'View results')).waitFor({ state: 'visible', timeout: 15000 });
  if (sidebar) {
    await removeSidebar(page, sidebar);
  }
  const duration = (Date.now() - startTime) / 1000;
  const coordinates = await getCoordinates(page, true);
  log('It was ' + coordinates, identifier);
  const resultJson = {
    coordinates,
    duration
  }
  roundEndAndSave('singleplayer', resultJson, gameId, roundNumber, identifier);
};

const gameStart = async (page: Page, mode: typeof MODE, waitText: string, waitTime: number, i: number, identifier?: string, resume = false) => {
  if (!resume || (await page.getByText('World', { exact: true }).count()) === 0) {
    await page.getByText(waitText).or(page.getByText('Rate limit')).nth(0).waitFor({ state: 'visible', timeout: 60000 });
    if (await page.getByText('Rate limit').count() > 0) {
      log('Rate-limited', identifier);
      fs.appendFileSync(TEMP_PATH + 'rate-limits', i + '\n');
      await page.waitForTimeout(STAGGER_INSTANCES);
      await (mode === 'singleplayer' ? playSingleplayer(page, i, identifier) : playMultiplayer(page, i, identifier));
      return;
    }
    await page.getByText(waitText).nth(0).waitFor({ state: 'hidden', timeout: waitTime });
  }
  // Get the game ID from the URL (https://www.geoguessr.com/de/battle-royale/<ID>)
  const gameId = page.url().split('/').pop() ?? 'no_id_' + randomUUID();
  if (fs.readFileSync(TEMP_PATH + mode + '-games', 'utf8')?.split(/\n/g)?.includes(gameId)) {
    log('Double-joined game', identifier);
    fs.appendFileSync(TEMP_PATH + 'double-joins', i + '\n');
    if (resume) {
      await page.waitForTimeout(STAGGER_INSTANCES);
      await (mode === 'singleplayer' ? playSingleplayer(page, i, identifier) : playMultiplayer(page, i, identifier));
    }
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

const gameSingleplayer = async (page: Page, i: number, identifier?: string, resume = false, roundNumber = 0) => {
  const gameId = await gameStart(page, 'singleplayer', 'Loading', 10000, i, identifier, resume);
  if (!gameId) {
    return;
  }
  let rounds = roundNumber;
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
  await clickButtonWithText(page, 'Classic Maps', 10000);
  await page.getByText('World', { exact: true }).first().waitFor({ state: 'visible', timeout: 10000 });
  await page.waitForTimeout(1000);
  await clickButtonIfFound(page, 'Play', true, false, true);
  await playGame(page, 'singleplayer', i, identifier);
};

const getResults = async (page: Page, games: string[], i: number, identifier?: string) => {
  await playStart(page, i, identifier);
  await page.waitForTimeout(1000);
  // load google maps and accept cookies
  await page.goto('https://www.google.com/maps', { timeout: 60000 });
  await getButtonWithText(page, 'Accept all').or(getButtonWithText(page, 'Alle akzeptieren')).first().click();
  await page.waitForTimeout(1000);
  for (const gameId of games) {
    log('Loading game - ' + gameId, identifier);
    await page.goto('https://www.geoguessr.com/results/' + gameId, { timeout: 60000 });
    await page.waitForTimeout(1000);
    if (await page.getByText('not found').or(page.getByText('nicht gefunden')).count() > 0) {
      log('Game not found: ' + gameId, identifier);
      // Go to the next game if the current one is not found
      break;
    }
    let count = 0;
    while (count < 10 && await page.getByText('finish').or(page.getByText('finished')).or(page.getByText('Ende')).or(page.getByText('beenden')).count() > 0) {
      log('Finishing game: ' + gameId, identifier);
      count++;
      // Finish the game first
      await page.goto('https://www.geoguessr.com/game/' + gameId, { timeout: 60000 });
      await page.waitForTimeout(1000);
      const roundLabelText = await page.getByText('Round').or(page.getByText('Runde')).first();
      // Get the text one element after the text Round (the button is the next element after the text) using the parent element and then just getting the last element
      const roundText = await (await roundLabelText.evaluateHandle((el) => el.parentElement?.lastElementChild))?.asElement()?.textContent();
      // Error if the text is not found
      expect(roundText).toBeTruthy();
      // Get the text of the element
      const roundNumber = parseInt(roundText?.split(/\s/g)[0] ?? '');
      // Error if the round number is not found
      expect(roundNumber).toBeTruthy();
      try {
        await gameSingleplayer(page, i, identifier, true, roundNumber - 1);
        count = 11;
      } catch (e) {
        log('Could not finish game: ' + gameId, identifier);
        console.error(e);
      }
      await page.goto('https://www.geoguessr.com/results/' + gameId, { timeout: 60000 });
    }
    if (count === 10) {
      expect('Game ' + gameId).toBe('finished, could not finnish game ' + gameId);
    }
    await page.waitForTimeout(1000);
    // Press view results button
    const found = await clickButtonIfFound(page, 'View results');
    if (found) {
      // Get button right before text Breakdown
      const breakdownText = await page.getByText('Breakdown', { exact: true });
      // Get the button one element before the text Breakdown (the button is the last element before the text) using the parent element and then just getting the button element
      const button = await breakdownText.evaluateHandle((el) => el.parentElement?.querySelector('button'));
      // Error if the button is not found
      expect(button).toBeTruthy();
      // Click the button
      await button?.asElement()?.click();
    } else {
      expect(await page.getByText('World', { exact: true }).count()).toBeGreaterThan(0);
    }
    // Get the pins
    const rounds = Array.from({ length: 5 }, (_, i) => i + 1);
    const roundLabels = rounds.map(round => page.getByText(String(round), { exact: true }));

    const oneOfLabels: Locator | undefined = roundLabels.reduce((acc: Locator | undefined, label) => acc ? acc.or(label) : label, undefined);

    const roundCoordinates: [number, number][] = [];

    let roundsChecked = 0;

    let stopWaiting = () => {};

    const stopWaitingPromise = new Promise<void>((resolve) => {
      stopWaiting = resolve;
    });

    // Click it and capture the url it tries to open (done via js, no href, formatted like https://www.google.com/maps?q&layer=c&cbll=66.40950012207031,14.124077796936035&cbp=0,undefined,0,0,undefined)
    // Can I capture the url it tries to open?
    page.context().on('page', async page => {
      // Get the first url of the page from the history
      let url = page.url();
      // If the url is a google consent url, click the accept button
      if (url.startsWith('https://consent.google.com/')) {
        await getButtonWithText(page, 'Accept all').or(getButtonWithText(page, 'Alle akzeptieren')).first().click();
        // Wait for the page to load
        await page.waitForURL(/https:\/\/www\.google\.com\/maps\/@/);
        url = page.url();
      }
      let coordinates = [];
      if (url.startsWith('https://www.google.com/maps?q&layer=c&cbll=')) {
        coordinates = url.split('cbll=')[1].split('&')[0].split(',');
      } else if (url.startsWith('https://www.google.com/maps/@')) {
        log('Potentially incorrect label for ' + gameId, identifier);
        coordinates = url.split('@')[1].split(',');
      } else {
        // Close the page
        page.close();
        return;
      }
      // If the url is a google maps url, save the coordinates
      const lat = parseFloat(coordinates[0]);
      const lon = parseFloat(coordinates[1]);
      if ((lat !== 0 || lon !== 0) && !isNaN(lat) && !isNaN(lon)) {
        roundCoordinates.push([lat, lon]);
      }
      roundsChecked++;
      if (roundsChecked === rounds.length) {
        stopWaiting();
      }
      // Close the page
      page.close();
    });

    await oneOfLabels?.first().waitFor({ state: 'visible' });
    
    let roundLabel: Locator | ElementHandle<HTMLElement> | null = null;
    let index = 0;
    for (roundLabel of roundLabels) {
      index++;
      roundLabel = await roundLabel.first();
      if ((await roundLabel.count()) > 0) {
        let count = 0;
        while (count < 10 && roundLabel) {
          count++;
          try {
            await roundLabel.click({ timeout: 1000 });
            break;
          } catch (e) {
            log('Could not click label ' + index + ': ' + gameId, identifier);
            // Otherwise check parent element
            roundLabel = 'or' in roundLabel ? (await roundLabel.evaluateHandle((el) => el.parentElement)).asElement() : (await roundLabel.evaluateHandle((el) => el.parentElement)).asElement();
            if (!roundLabel) {
              count = 10;
              console.error(e);
            }
          }
        }

        if (count === 10) {
          expect('Label ' + index + ' in game ' + gameId).toBe('Clickable labels, could not click label ' + index);
        }
      }
    }

    // Wait for the coordinates to be collected
    await stopWaitingPromise;
    
    for (let roundNumber = 0; roundNumber < roundCoordinates.length; roundNumber++) {
      const coordinates = roundCoordinates[roundNumber];
      log('It was ' + coordinates, identifier);
      const resultJson = {
        coordinates
      }

      roundEndAndSave('singleplayer', resultJson, gameId, roundNumber, identifier);
    }
  }
}

describe('Geoguessr', () => {
  for (let i = 0; i < NUMBER_OF_INSTANCES; i++) {
    const identifier = (NUMBER_OF_INSTANCES > 1 ? String(i + 1) : '');
    let gamesToCheck = MODE === 'results' ? GAMES : [];
    if (gamesToCheck.length) {
      // Get already checked games by listing files in data folder
      let checkedGames = fs.readdirSync(DATA_PATH).filter(file => file.startsWith(RESULT_FILE) && file.endsWith(RESULT_FILE_EXTENSION)).map(file => file.slice(RESULT_FILE.length, -RESULT_FILE_EXTENSION.length));
      // Get only games that start with "singleplayer_", then remove that.
      checkedGames = checkedGames.filter(game => game.startsWith('singleplayer_')).map(game => game.slice('singleplayer_'.length, -'_1'.length));
      // Filter out already checked games
      gamesToCheck = gamesToCheck.filter(game => !checkedGames.includes(game));
      // Segment into runners
      const runnerIndex = Math.round(gamesToCheck.length / NUMBER_OF_INSTANCES) * i;
      const runnerIndexEnd = i === NUMBER_OF_INSTANCES - 1 ? gamesToCheck.length : Math.round(gamesToCheck.length / NUMBER_OF_INSTANCES) * (i + 1);
      gamesToCheck = gamesToCheck.slice(runnerIndex, runnerIndexEnd);
    }
    // Go to "geoguessr.com", log in, play a game, take a screenshot of the viewer and save the game result into a file.
    test('play ' + (MODE === 'singleplayer' ? 'world' : MODE === 'multiplayer' ? 'countries battle royale' : 'results') + (identifier ? ' - ' + identifier : ''), async ({ page }) => {
      if (fs.readFileSync(TEMP_PATH + 'stop', 'utf8') === 'true') {
        process.exit(1);
      }
      // Set viewport size in singleplayer mode to align with multiplayer mode
      if (MODE === 'singleplayer') {
        page.setViewportSize({ width: SINGLEPLAYER_WIDTH, height: page.viewportSize()?.height ?? 0});
      }
      try {
        test.setTimeout(60000 * MAX_MINUTES);
        await (MODE === 'singleplayer' ? playSingleplayer(page, i, identifier) : MODE === 'multiplayer' ? playMultiplayer(page, i, identifier) : getResults(page, gamesToCheck, i, identifier));
      } catch (e: unknown) {
        const timestamp = getTimestampString();
        // If messages includes 'Target crashed', exit program, otherwise log an error message that an Error occurred in this instance at this time and rethrow
        if (typeof e === 'object' && e instanceof Error && (e.message.includes('Target crashed') || e.message.includes('exited unexpectedly'))) {
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
