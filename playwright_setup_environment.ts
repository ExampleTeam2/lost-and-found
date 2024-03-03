import { DATA_PATH, TEMP_PATH, getTimestampString } from './playwright_base_config.js';
import fs from 'fs';

const timestamp = getTimestampString();

console.log('Setting up environment...');
if (!fs.existsSync(DATA_PATH)){
  fs.mkdirSync(DATA_PATH);
}
if (!fs.existsSync(TEMP_PATH)){
  fs.mkdirSync(TEMP_PATH);
}
fs.appendFileSync(TEMP_PATH + 'games', '');
fs.appendFileSync(TEMP_PATH + 'crashes', '');
fs.writeFileSync(TEMP_PATH + 'initial', 'true');
fs.appendFileSync(TEMP_PATH + 'rate-limits', timestamp + '\n');
fs.appendFileSync(TEMP_PATH + 'double-joins', timestamp + '\n');
fs.appendFileSync(TEMP_PATH + 'other-restarts', timestamp + '\n');
fs.appendFileSync(TEMP_PATH + 'starts', timestamp + '\n');
fs.writeFileSync(TEMP_PATH + 'stop', 'false');
