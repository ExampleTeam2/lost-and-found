import { DATA_PATH, MODE, TEMP_PATH, getTimestampString } from './playwright_base_config.js';
import fs from 'fs';

const timestamp = getTimestampString();

console.log('Setting up environment...');
if (!fs.existsSync(DATA_PATH)) {
  fs.mkdirSync(DATA_PATH);
}
if (!fs.existsSync(TEMP_PATH)) {
  fs.mkdirSync(TEMP_PATH);
}
if (MODE === 'results') {
  fs.writeFileSync(TEMP_PATH + 'singleplayer-games', '');
}
fs.appendFileSync(TEMP_PATH + MODE + '-games', '');
fs.appendFileSync(TEMP_PATH + 'crashes', '');
fs.writeFileSync(TEMP_PATH + 'initial', 'true');
fs.appendFileSync(TEMP_PATH + 'rate-limits', timestamp + '\n');
fs.appendFileSync(TEMP_PATH + 'double-joins', timestamp + '\n');
fs.appendFileSync(TEMP_PATH + 'other-restarts', timestamp + '\n');
fs.appendFileSync(TEMP_PATH + 'starts', timestamp + '\n');
if (MODE === 'results') {
  fs.writeFileSync(TEMP_PATH + 'singleplayer-starts', timestamp + '\n');
}
fs.appendFileSync(TEMP_PATH + MODE + '-starts', timestamp + '\n');
fs.writeFileSync(TEMP_PATH + 'stop', 'false');
