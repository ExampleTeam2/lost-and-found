import { test, expect } from '@playwright/test';
import fs from 'fs';

const DATA_PATH = 'scraping/data/';

// Go to "hslu.ch", navigate to "Informatik", print the text, look for a field of study containing "Artificial Intelligence" and take a screenshot of that field of study.
test('hslu.ch', async ({ page }) => {
  await page.goto('https://www.hslu.ch');
  await page.click('text=Informatik');
  const text = await page.textContent('body');
  // Format into json of sub sections and write into a file async (log if error)
  fs.writeFile(DATA_PATH + 'hslu.json', JSON.stringify(text), (err) => {
    if (err) console.log(err);
  });
  const ai = await page.$('text=Artificial Intelligence');
  expect(ai).toBeTruthy();
  await ai?.screenshot({ path: DATA_PATH + 'ai.png' });
});
