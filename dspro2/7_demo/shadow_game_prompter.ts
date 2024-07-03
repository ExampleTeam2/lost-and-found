import fs from 'fs';
import prompt from 'prompt-sync';

const TEMP_PATH = 'dspro2/1_data_collection/tmp/'; // Keep in sync with playwright_base_config.ts
const prompter = prompt();

// If command line arguments are provided, write the first argument to the file TEMP_PATH + 'shadow_game'.
if (process.argv.length > 2) {
  fs.writeFileSync(TEMP_PATH + 'shadow_game', process.argv[2]);
  process.exit(0);
}

let url: string | null = null;

// Repeatedly prompt the user for the URL of the game they want to shadow.
do {
  // Wait for the user answer to the prompt, then write synchronously to the file TEMP_PATH + 'shadow_game' the answer.
  url = prompter('Enter the URL of the game you want to shadow: ');

  if (url) {
    fs.writeFileSync(TEMP_PATH + 'shadow_game', url);
  }
} while (url !== null);
