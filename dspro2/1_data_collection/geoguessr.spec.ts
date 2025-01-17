import { test, expect, ElementHandle, Locator } from '@playwright/test';
import fs from 'fs';
import 'dotenv/config';
import { Page } from 'playwright-core';
import { describe } from 'node:test';
import { DATA_PATH, DEMO_DATA_PATH, DEMO_WIDTH, GAMES, LOCATION_FILE, LOCATION_FILE_EXTENSION, MAX_GAMES, MAX_MINUTES, MAX_ROUNDS, MODE, NUMBER_OF_INSTANCES, RESULT_FILE, RESULT_FILE_EXTENSION, SINGLEPLAYER_WIDTH, STAGGER_INSTANCES, TEMP_PATH, getTimestampString } from './playwright_base_config';
import checkDiskSpace from 'check-disk-space';
import path from 'path';
import { watch } from 'chokidar';

let logProgressInterval: ReturnType<typeof setInterval> | undefined;

const error = (message: string, i?: string) => {
  console.error((i ? i + ' - ' : '') + message);
};

let progressLogDisabled = false;

// Log a message, print a dot (on the same line) every 10 seconds in a background process to get progress indication until another message is logged
const log = (message: string, i?: string) => {
  console.log((i ? i + ' - ' : '') + message);
  if (!i) {
    if (logProgressInterval) {
      clearInterval(logProgressInterval);
    }
    logProgressInterval = setInterval(() => (!progressLogDisabled ? process.stdout.write('.') : undefined), 10000);
  }
};

const getVolumeRoot = (currentPath: string) => {
  if (process.platform === 'win32') {
    // Get the drive letter and append ':\'
    return currentPath.split(path.sep)[0] + '\\';
  } else {
    return '/';
  }
};

const hasEnoughFreeDiskSpace = async () => {
  const cwd = process.cwd();

  const volumeRoot = getVolumeRoot(cwd);

  const diskSpace = await checkDiskSpace(volumeRoot);

  // Is there more free storage than 100MB in bytes
  return diskSpace.free > 100 * 1024 * 1024;
};

const waitForFreeDiskSpace = async (identifier?: string) => {
  let first = true;
  while (!(await hasEnoughFreeDiskSpace())) {
    if (first) {
      log('Waiting for free disk space', identifier);
    }
    first = false;
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
};

const getButtonWithText = (page: Page, text: string, only = false) => {
  return page.locator('button, a').getByText(text, { exact: only });
};

const clickButtonWithText = async (page: Page, text: string, wait = 0, only = false) => {
  const button = getButtonWithText(page, text, only);
  if (wait) {
    await button.waitFor({ state: 'visible', timeout: wait !== -1 ? wait : undefined });
  }
  return await button.click();
};

const clickButtonIfFound = async (page: Page, text: string, only = false, first = false, last = false, noWaitAfter = false) => {
  let button = getButtonWithText(page, text, only);
  if (first) {
    button = button.first();
  }
  if (last) {
    button = button.last();
  }
  if ((await button.count()) > 0) {
    await button.click({ noWaitAfter });
    return true;
  }
  return false;
};

const setCookies = async (page: Page) => {
  // File could be undefined, so check if it exists, but keep synchronous
  if (fs.existsSync(DATA_PATH + 'cookies.json')) {
    const cookies = JSON.parse(fs.readFileSync(DATA_PATH + 'cookies.json', 'utf8'));
    await page.context().addCookies(cookies);
  }
};

const checkIfLoggedIn = async (page: Page) => {
  return (await page.getByText('Log in').count()) === 0;
};

const getCookieBanner = (page: Page) => {
  return page.locator('#onetrust-consent-sdk').or(page.locator('#snigel-cmp-framework')).first();
};

// Create function which checks for a cookie banner and removes it
const removeCookieBanner = async (page: Page, timeout = 15000, allowSkipping = true) => {
  // Check if element with id "onetrust-consent-sdk" exists
  const cookieBanner = getCookieBanner(page);
  if (allowSkipping) {
    page.waitForTimeout(timeout);
    if ((await cookieBanner.count()) === 0) {
      return;
    }
  }
  await cookieBanner.waitFor({ state: 'attached', timeout: timeout });
  await getButtonWithText(page, 'Accept').or(page.getByText('Accept all')).first().click();
  page.waitForTimeout(1000);
  await cookieBanner.evaluate((el) => el.remove());
};

const logIn = async (page: Page, identifier?: string) => {
  log('Log in needed', identifier);
  // Fail the test if the environment variables are not set
  expect(process.env.GEOGUESSR_EMAIL, 'Please set a `GEOGUESSR_EMAIL` and `GEOGUESSR_PASSWORD` in your .env').toBeTruthy();
  expect(process.env.GEOGUESSR_PASSWORD, 'Please set a `GEOGUESSR_PASSWORD` in your .env').toBeTruthy();
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
  fs.writeFile(DATA_PATH + 'cookies.json', JSON.stringify(cookies), (e) => {
    const timestamp = getTimestampString();
    if (e) {
      error(`Error occurred while saving cookies at ${timestamp}:`, identifier);
      console.error(e);
    }
  });
};

// Inject css with a way to remove it again
const injectCss = async (page: Page, css: string) => {
  return await page.addStyleTag({ content: css });
};

const removeElement = async (element: ElementHandle<Node>) => {
  return await element.evaluate((el) => el.parentNode?.removeChild(el));
};

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
        const incorrect = await incorrectGuessesHeading.evaluate((el): string[] => {
          if (!el.parentElement) {
            return [];
          }
          return Array.from(el.parentElement.querySelectorAll('img'))
            .map((img) => img.getAttribute('alt'))
            .filter((text) => text) as string[];
        });
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
};

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
  const links = await linksLocator.all().then((result) => Promise.all(result.map((link) => link.getAttribute('href'))));
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
  return (
    await element.evaluateHandle((el) => {
      let parent = el.parentElement;
      while (parent && parent.clientHeight <= 50) {
        parent = parent.parentElement;
      }
      return parent;
    })
  )?.asElement();
};

const getButtonInMap = (page: Page) => {
  // Find button with title "Keyboard shortcuts" and Img in it or span with text "Map data ©"
  return page.locator('button[title="Keyboard shortcuts"] img').or(page.locator('span').getByText('Map data ©')).nth(0);
};

const guess = async (page: Page, force = true) => {
  // Find button with title "Keyboard shortcuts" and Img in it or span with text "Map data ©"
  const button = getButtonInMap(page);
  // Await if force is enabled
  if (force) {
    await button.waitFor({ state: 'visible', timeout: 10000 });
  }
  // Make sure at least one button is found
  if ((await button.count()) > 0) {
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
    const pointX = dimensions.x + gapFromEdge + Math.random() * (dimensions.width - gapFromEdge * 2),
      pointY = dimensions.y + gapFromEdge + Math.random() * (dimensions.height - gapFromEdge * 2);
    await page.mouse.move(pointX, pointY);
    // Wait for 1 second
    await page.waitForTimeout(1000);
    // Click the mouse
    await page.mouse.click(pointX, pointY);
    // Click the guess button
    await clickButtonWithText(page, 'Guess', undefined, true);
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
  await Promise.all(
    (await page.locator('#' + sidebarId).all()).map(async (el) => {
      const element = await el.elementHandle();
      if (element) {
        await removeElement(element);
      }
    }),
  );
  await page.evaluate((sidebarId) => {
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

const getCoordinatesFromPin = async (page: Page, gameId: string, identifier?: string, all = false) => {
  // Get the pins
  const rounds = all ? Array.from({ length: 5 }, (_, i) => i + 1) : [false];
  // Get labels with label as text and data-qa="correct-location-marker" (first data-qa="correct-location-marker", then the text
  const roundLabels = rounds.map((round) => {
    let loc = page.locator('css=[data-qa="correct-location-marker"]');
    if (round) {
      loc = loc.filter({ hasText: new RegExp(`^${round}$`) });
    }
    return loc;
  });

  const roundCoordinates: [number, number][] = [];

  // Click it and capture the url it tries to open (done via js, no href, formatted like https://www.google.com/maps?q&layer=c&cbll=66.40950012207031,14.124077796936035&cbp=0,undefined,0,0,undefined)
  const handlePopup = async (popup: Page, index: number) => {
    // Get the first url of the page from the history
    const url = popup.url();
    let coordinates = [];
    if (url.startsWith('https://www.google.com/maps?q&layer=c&cbll=')) {
      coordinates = url.split('cbll=')[1].split('&')[0].split(',');
    } else {
      // Close the page
      await popup.close();
      expect('Label' + (all ? ' ' + index : '') + ' in game ' + gameId).toBe('Google Maps URL, could not find coordinates');
      return;
    }
    // If the url is a google maps url, save the coordinates
    const lat = parseFloat(coordinates[0]);
    const lon = parseFloat(coordinates[1]);
    if ((lat !== 0 || lon !== 0) && !isNaN(lat) && !isNaN(lon)) {
      roundCoordinates[index] = [lat, lon];
    } else {
      await popup.close();
      expect('Coordinates of ' + (all ? ' ' + index : '') + ' in game ' + gameId).toBe('Valid coordinates');
    }
    // Close the page
    await popup.close();
  };

  // Hide all the labels of roundLabels
  for (let roundLabel of roundLabels) {
    await roundLabel.waitFor({ state: 'visible' });
    if (all) {
      await roundLabel.evaluate((el) => (el.style.display = 'none'));
    }
  }

  let index = 0;
  for (const roundLabel of roundLabels) {
    index++;
    if (all) {
      // Show the label
      await roundLabel.evaluate((el) => (el.style.display = ''));
      // Make sure the label is visible
      await roundLabel.waitFor({ state: 'visible' });
    }
    let currentRoundLabel: Locator | ElementHandle<HTMLElement> | null = roundLabel;
    const popup = page.waitForEvent('popup');
    let found = false;
    try {
      await currentRoundLabel.click({ timeout: 10000 });
      found = true;
    } catch (e) {
      log('Could not click label' + (all ? ' ' + index : '') + ': ' + gameId, identifier);
      throw e;
    }

    if (all) {
      // Hide the label
      await roundLabel.evaluate((el) => (el.style.display = 'none'));
    }

    if (found) {
      await handlePopup(await popup, index - 1);
    } else {
      expect('Label' + (all ? ' ' + index : '') + ' in game ' + gameId).toBe('Clickable labels, could not click label' + (all ? ' ' + index : ''));
    }
  }

  return roundCoordinates;
};

const roundStartAndCapture = async <const T>(page: Page, mode: typeof MODE, gameId: string, roundNumber: number, identifier?: string, additional: (page: Page, gameId: string, roundNumber: number, identifier?: string) => T = (() => undefined) as () => T): Promise<[number, Awaited<T>, ElementPair?]> => {
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
  await viewer?.screenshot({ path: (MODE !== 'demo' ? DATA_PATH : DEMO_DATA_PATH) + LOCATION_FILE + mode + '_' + gameId + '_' + roundNumber + LOCATION_FILE_EXTENSION });
  await removeElement(css);
  return [startTime, additionalResults, sidebar];
};

const roundEndAndSave = (mode: typeof MODE, result: unknown, gameId: string, roundNumber: number, identifier?: string) => {
  if (MODE !== 'demo') {
    fs.writeFile(DATA_PATH + RESULT_FILE + mode + '_' + gameId + '_' + roundNumber + RESULT_FILE_EXTENSION, JSON.stringify(result), (e) => {
      const timestamp = getTimestampString();
      if (e) {
        error(`Error occurred while saving game results at ${timestamp}:`, identifier);
        console.error(e);
      }
    });
  }
};

const roundMultiplayer = async (page: Page, gameId: string, roundNumber: number, identifier?: string) => {
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
  const country = resultText?.split('right answer was')[1].split('.')[0].replace('indeed', '').replace('actually', '').trim();
  log('It was ' + country, identifier);
  const resultJson = {
    country,
    coordinates,
    guesses,
    people,
    duration,
  };
  roundEndAndSave('multiplayer', resultJson, gameId, roundNumber, identifier);
};

const roundSingleplayer = async (page: Page, gameId: string, roundNumber: number, identifier?: string, resume = false) => {
  const [startTime, _, sidebar] = await roundStartAndCapture(page, 'singleplayer', gameId, roundNumber, identifier);
  await guess(page);
  await getButtonWithText(page, 'Next')
    .or(await getButtonWithText(page, 'View results'))
    .waitFor({ state: 'visible', timeout: 15000 });
  if (sidebar) {
    await removeSidebar(page, sidebar);
  }
  const duration = (Date.now() - startTime) / 1000;
  const coordinates = (await getCoordinatesFromPin(page, gameId, identifier, false))[0];
  log('It was ' + coordinates, identifier);
  const resultJson = {
    coordinates,
    duration,
  };
  if (!resume) {
    roundEndAndSave('singleplayer', resultJson, gameId, roundNumber, identifier);
  }
};

const gameStart = async (page: Page, mode: typeof MODE, waitText: string, waitTime: number, i: number, identifier?: string, resume = false) => {
  if (!resume || (await page.getByText('World', { exact: true }).count()) === 0) {
    await page.getByText(waitText).or(page.getByText('Rate limit')).nth(0).waitFor({ state: 'visible', timeout: 60000 });
    await waitForFreeDiskSpace(identifier);
    if ((await page.getByText('Rate limit').count()) > 0) {
      log('Rate-limited', identifier);
      fs.appendFileSync(TEMP_PATH + 'rate-limits', i + '\n');
      await page.waitForTimeout(STAGGER_INSTANCES);
      await (mode === 'singleplayer' ? playSingleplayer(page, i, identifier) : playMultiplayer(page, i, identifier));
      return;
    }
    await page.getByText(waitText).nth(0).waitFor({ state: 'hidden', timeout: waitTime });
  } else {
    await waitForFreeDiskSpace(identifier);
  }
  // Get the game ID from the URL (https://www.geoguessr.com/battle-royale/<ID>)
  const gameId = page.url().split('/').pop() ?? 'no_id_' + randomUUID();
  if (
    MODE !== 'demo' &&
    !resume &&
    fs
      .readFileSync(TEMP_PATH + mode + '-games', 'utf8')
      ?.split(/\n/g)
      ?.includes(gameId)
  ) {
    log('Double-joined game', identifier);
    fs.appendFileSync(TEMP_PATH + 'double-joins', i + '\n');
    if (!resume) {
      await page.waitForTimeout(STAGGER_INSTANCES);
      await (mode === 'singleplayer' ? playSingleplayer(page, i, identifier) : playMultiplayer(page, i, identifier));
    }
    return;
  }
  if (MODE !== 'demo') {
    fs.appendFileSync(TEMP_PATH + mode + '-games', gameId + '\n');
  }
  log('Starting game - ' + gameId, identifier);
  return gameId;
};

// Stop if a crash happened before or by another instance.
const stopIfCrashedBefore = () => {
  if (fs.readFileSync(TEMP_PATH + 'stop', 'utf8') === 'true') {
    process.exit(1);
  }
};

const stopShadowing: Record<string, boolean> = {};

const gameMultiplayer = async (page: Page, i: number, identifier?: string) => {
  const gameId = await gameStart(page, 'multiplayer', 'Game starting in', 60000, i, identifier);
  if (!gameId) {
    return;
  }
  stopShadowing[gameId] = false;
  try {
    let rounds = 0;
    await roundMultiplayer(page, gameId, rounds, identifier);
    await page.waitForTimeout(1000);
    if (stopShadowing[gameId]) {
      return;
    }
    if (MODE === 'demo' || (await clickButtonIfFound(page, 'Spectate'))) {
      if (MODE === 'demo') {
        if ((await page.getByText('Correct').count()) === 0) {
          await clickButtonIfFound(page, 'Spectate');
        }
      }
      rounds++;
      if (MODE !== 'demo') {
        await page.getByText('Next round starts in').waitFor({ state: 'visible' });
        await page.getByText('Next round starts in').waitFor({ state: 'hidden', timeout: 20000 });
      } else {
        await page.getByText('Correct').or(page.getByText('Next round starts in')).or(page.getByText('Next round to start')).or(getButtonWithText(page, 'Spectate')).first().waitFor({ state: 'visible' });
        if (stopShadowing[gameId]) {
          return;
        }
        await page.getByText('Correct').or(page.getByText('Next round starts in')).or(page.getByText('Next round to start')).waitFor({ state: 'hidden', timeout: 120000 });
      }
      await roundMultiplayer(page, gameId, rounds, identifier);
      rounds++;
      // Remove footer to improve vision and avoid second "Play again" button
      if (MODE !== 'demo') {
        await page.locator('footer').evaluate((el) => el.remove());
      }
      while ((rounds < MAX_ROUNDS && MODE !== 'demo' && (await page.getByText('Next round starts').count()) > 0) || (MODE === 'demo' && (await page.getByText('Correct').or(page.getByText('Next round starts')).count()) > 0)) {
        if (stopShadowing[gameId]) {
          return;
        }
        if (MODE !== 'demo') {
          await page.getByText('Next round starts in').waitFor({ state: 'visible' });
          await page.getByText('Next round starts in').waitFor({ state: 'hidden', timeout: 20000 });
        } else {
          await page.getByText('Correct').or(page.getByText('Next round starts in')).or(page.getByText('Next round to start')).or(getButtonWithText(page, 'Spectate')).first().waitFor({ state: 'visible' });
          if (stopShadowing[gameId]) {
            return;
          }
          await page.getByText('Correct').or(page.getByText('Next round starts in')).or(page.getByText('Next round to start')).waitFor({ state: 'hidden', timeout: 120000 });
        }
        await roundMultiplayer(page, gameId, rounds, identifier);
        rounds++;
        await page.waitForTimeout(1000);
      }
    }
  } catch (e: unknown) {
    if (stopShadowing) {
      stopIfCrashedBefore();
      return;
    } else {
      throw e;
    }
  }
  stopIfCrashedBefore();
};

const gameSingleplayer = async (page: Page, i: number, identifier?: string, resume = false, roundNumber = 0) => {
  const gameId = await gameStart(page, 'singleplayer', 'Loading location', 10000, i, identifier, resume);
  if (!gameId) {
    return;
  }
  let rounds = roundNumber;
  await roundSingleplayer(page, gameId, rounds, identifier, resume);
  rounds++;
  await page.waitForTimeout(1000);
  while (rounds < MAX_ROUNDS && (await clickButtonIfFound(page, 'Next'))) {
    await roundSingleplayer(page, gameId, rounds, identifier, resume);
    rounds++;
    await page.waitForTimeout(1000);
  }
  const resultsButton = getButtonWithText(page, 'View results');
  await resultsButton.click();
  await resultsButton.waitFor({ state: 'hidden', timeout: 10000 });
  stopIfCrashedBefore();
};

const playStart = async (page: Page, i: number, identifier?: string) => {
  stopIfCrashedBefore();
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
    await removeCookieBanner(page, 10000, true);
    log('Already logged in', identifier);
  }
  page.setDefaultTimeout(5000);
};
const playGame = async (page: Page, mode: typeof MODE, i: number, identifier?: string) => {
  let games = 1;
  await (mode === 'singleplayer' ? gameSingleplayer(page, i, identifier) : gameMultiplayer(page, i, identifier));
  await page.waitForTimeout(1000);
  while (games < (MODE === 'demo' ? 1 : MAX_GAMES) && (await clickButtonIfFound(page, 'Play again', undefined, undefined, undefined, true))) {
    await (mode === 'singleplayer' ? gameSingleplayer(page, i, identifier) : gameMultiplayer(page, i, identifier));
    games++;
    await page.waitForTimeout(3000);
  }
  // Retry if ended early
  if (games !== MAX_GAMES && MODE !== 'demo') {
    log('Could not start next game', identifier);
    fs.appendFileSync(TEMP_PATH + 'other-restarts', i + '\n');
    await page.waitForTimeout(STAGGER_INSTANCES);
    await (mode ? playSingleplayer(page, i, identifier) : playMultiplayer(page, i, identifier));
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
  await acceptGoogleCookies(page);
  await playStart(page, i, identifier);
  await clickButtonWithText(page, 'Classic Maps', 10000);
  await page.getByText('World', { exact: true }).first().waitFor({ state: 'visible', timeout: 10000 });
  await page.waitForTimeout(1000);
  await clickButtonIfFound(page, 'Play', true, false, true);
  await playGame(page, 'singleplayer', i, identifier);
};

// Shadow given game, can not be used with multiple instances.
const shadowGame = async (page: Page, gameId: string) => {
  // End previous games.
  for (const gameId of Object.keys(stopShadowing)) {
    stopShadowing[gameId] = true;
  }
  log('Loading game - ' + gameId);
  await page.goto('https://www.geoguessr.com/battle-royale/' + gameId, { timeout: 60000 });
  await page.waitForTimeout(1000);
  if ((await page.getByText('not found').or(page.getByText('nicht gefunden')).count()) > 0) {
    log('Game not found: ' + gameId);
    // Go to the next game if the current one is not found
    return;
  }
  await playGame(page, 'multiplayer', 0);
};

// Shadow given games, can not be used with multiple instances.
const shadowGames = async (page: Page) => {
  await playStart(page, 0);
  await page.waitForTimeout(1000);
  await acceptGoogleCookies(page);

  progressLogDisabled = true;

  // Wait for a TEMP_PATH + 'shadow_game' file to be created and print its content.
  const watcher = watch(TEMP_PATH, { persistent: true });

  console.log('Waiting for multiplayer game URLs to be queued...');

  watcher.on('add', async (path) => {
    try {
      let gameUrl: string | undefined = undefined;
      if (path.endsWith('/shadow_game')) {
        try {
          gameUrl = fs.readFileSync(path, { encoding: 'utf8' }).replace(/\s/g, '');
          // Delete the file after reading it.
          fs.unlinkSync(path);
        } catch (error) {
          console.error('Error reading game URL:', error);
        }
      }
      if (!gameUrl) {
        if (gameUrl === '') {
          console.error('No game URL provided.');
        }
        return;
      }
      if (!gameUrl.startsWith('https://www.geoguessr.com/battle-royale/')) {
        console.error('Game URL is not of correct mode:', gameUrl);
        return;
      }
      const gameId = gameUrl.replace('https://www.geoguessr.com/battle-royale/', '').replace(/\?.*/, '');
      if (!gameId || gameId.includes('/') || gameId.includes('?') || gameId.includes('#')) {
        console.error('Invalid game URL:', gameUrl);
        return;
      }
      progressLogDisabled = false;
      console.log(`Shadowing game ${gameId}`);
      await shadowGame(page, gameId);
      progressLogDisabled = true;
    } catch (e: unknown) {
      handleErrors(e);
    }
  });

  // Never resolve the promise to keep the watcher running forever
  return new Promise<void>(() => {});
};

const handleErrors = (e: unknown, identifier?: string) => {
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
};

const acceptGoogleCookies = async (page: Page) => {
  // load google maps and accept cookies
  await page.goto('https://www.google.com/maps', { timeout: 60000 });
  await getButtonWithText(page, 'Accept all').or(getButtonWithText(page, 'Alle akzeptieren')).first().click();
  await page.waitForTimeout(1000);
};

const getResult = async (page: Page, gameId: string, i: number, identifier?: string) => {
  log('Loading game - ' + gameId, identifier);
  await page.goto('https://www.geoguessr.com/results/' + gameId, { timeout: 60000 });
  await page.waitForTimeout(1000);
  if ((await page.getByText('not found').or(page.getByText('nicht gefunden')).count()) > 0) {
    log('Game not found: ' + gameId, identifier);
    // Go to the next game if the current one is not found
    return;
  }
  let count = 0;
  while (count < 10 && (await page.getByText('finish').or(page.getByText('finished')).or(page.getByText('Ende')).or(page.getByText('beenden')).count()) > 0) {
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
      if (typeof e === 'object' && e instanceof Error && (e.message.includes('Target crashed') || e.message.includes('exited unexpectedly'))) {
        throw e;
      } else {
        console.error(e);
      }
    }
    await page.goto('https://www.geoguessr.com/results/' + gameId, { timeout: 60000 });
  }
  if (count === 10) {
    expect('Game ' + gameId).toBe('finished, could not finnish game ' + gameId);
  }
  // Press view results buttons
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

  const roundCoordinates = await getCoordinatesFromPin(page, gameId, identifier, true);

  for (let roundNumber = 0; roundNumber < roundCoordinates.length; roundNumber++) {
    const coordinates = roundCoordinates[roundNumber];
    log(roundNumber + 1 + ' was ' + coordinates, identifier);
    const resultJson = {
      coordinates,
    };

    roundEndAndSave('singleplayer', resultJson, gameId, roundNumber, identifier);
  }

  fs.appendFileSync(TEMP_PATH + 'results-games', gameId + '\n');

  stopIfCrashedBefore();
};

const getResults = async (page: Page, games: string[], i: number, identifier?: string) => {
  await playStart(page, i, identifier);
  await page.waitForTimeout(1000);
  await acceptGoogleCookies(page);
  for (const gameId of games) {
    await getResult(page, gameId, i, identifier);
  }
};

// Get games to check for results mode.
const getGamesToCheck = (i: number) => {
  let gamesToCheck = GAMES;
  if (gamesToCheck.length) {
    // Get already checked games by listing files in data folder
    let checkedGames = new Set(
      fs
        .readFileSync(TEMP_PATH + 'results-games', 'utf8')
        ?.split(/\n/g)
        .filter((file) => file),
    );
    // Filter out already checked games
    gamesToCheck = gamesToCheck.filter((game) => !checkedGames.has(game));
    // Segment into runners
    const runnerIndex = Math.round(gamesToCheck.length / NUMBER_OF_INSTANCES) * i;
    const runnerIndexEnd = i === NUMBER_OF_INSTANCES - 1 ? gamesToCheck.length : Math.round(gamesToCheck.length / NUMBER_OF_INSTANCES) * (i + 1);
    gamesToCheck = gamesToCheck.slice(runnerIndex, runnerIndexEnd);
  }
  return gamesToCheck;
};

describe('Geoguessr', () => {
  if (MODE !== 'demo') {
    for (let i = 0; i < NUMBER_OF_INSTANCES; i++) {
      const identifier = NUMBER_OF_INSTANCES > 1 ? String(i + 1) : '';

      // Get games to check for results mode.
      const gamesToCheck = MODE === 'results' ? getGamesToCheck(i) : [];

      // Go to "geoguessr.com", log in, play a game, take a screenshot of the viewer and save the game result into a file, (or just get the results of a previous game), and repeat.
      test('play ' + (MODE === 'singleplayer' ? 'world' : MODE === 'multiplayer' ? 'countries battle royale' : 'results') + (identifier ? ' - ' + identifier : ''), async ({ page }) => {
        stopIfCrashedBefore();
        // Set viewport size in singleplayer mode to align with multiplayer mode
        if (MODE === 'singleplayer') {
          page.setViewportSize({ width: SINGLEPLAYER_WIDTH, height: page.viewportSize()?.height ?? 0 });
        }
        try {
          test.setTimeout(60000 * MAX_MINUTES);
          await (MODE === 'singleplayer' ? playSingleplayer(page, i, identifier) : MODE === 'multiplayer' ? playMultiplayer(page, i, identifier) : MODE === 'results' ? getResults(page, gamesToCheck, i, identifier) : Promise.reject('Invalid mode'));
        } catch (e: unknown) {
          handleErrors(e, identifier);
        }
      });
    }
  } else {
    // Shadow games for demo.
    // TODO: For now only works if joined at beginning of game.
    if (NUMBER_OF_INSTANCES > 1) {
      throw new Error('Demo mode can only be run with one instance');
    }
    // Go to "geoguessr.com", log in, shadow given game, take a screenshot of the viewer and save into a file in the demo data folder, and repeat.
    test('shadow countries battle royale', async ({ page }) => {
      stopIfCrashedBefore();
      try {
        page.setViewportSize({ width: DEMO_WIDTH, height: page.viewportSize()?.height ?? 0 });
        test.setTimeout(60000 * MAX_MINUTES);
        await shadowGames(page);
      } catch (e: unknown) {
        handleErrors(e);
      }
    });
  }
});
