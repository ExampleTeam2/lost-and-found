import fs from 'fs';

export const DATA_PATH = 'dspro2/1_data_collection/.data/';
export const TEMP_PATH = 'dspro2/1_data_collection/tmp/';
export const DEMO_DATA_PATH = 'dspro2/7_demo/.data/';
export const LOCATION_FILE = 'geoguessr_location_';
export const LOCATION_FILE_EXTENSION = '.png';
export const RESULT_FILE = 'geoguessr_result_';
export const RESULT_FILE_EXTENSION = '.json';
export const MAX_ROUNDS = process.env.CI ? 100 : 15;
export const MAX_GAMES = process.env.CI ? 100000 : 10;
export const MAX_MINUTES = process.env.CI ? 60 * 24 * 14 : 60;
export const MAX_RETRIES = process.env.CI ? 10000 : 0;
export const NUMBER_OF_INSTANCES = process.env.CI ? 5 : 1;
export const STAGGER_INSTANCES = 40000;
type Mode = 'multiplayer' | 'singleplayer' | 'results' | 'demo';
export const MODE: Mode = !process.env.PLAYWRIGHT_MODE ? 'singleplayer' : (process.env.PLAYWRIGHT_MODE as Mode);
export const GAMES = (MODE as any) === 'results' ? fs.readFileSync(DATA_PATH + 'games', 'utf-8').split('\n') : [];
// To allow for a small sidebar, wider than multiplayer, for same width use 992 + 1
export const SINGLEPLAYER_WIDTH = 1280 + 1;
export const getTimestampString = () => new Date().toISOString().replace(/[:.]/g, '-');
