import yaml from 'js-yaml';
import fs from 'fs';
import path from 'path';

interface Config {
    apiBaseUrl: string;
}

let config: Config;

export const loadConfig = () => {
    try {
        const filePath = path.resolve(__dirname, '../config.yaml');
        const fileContents = fs.readFileSync(filePath, 'utf8');
        config = yaml.load(fileContents) as Config;
    } catch (e) {
        console.error(e);
    }
};

export const getConfig = (): Config => {
    if (!config) {
        loadConfig();
    }
    return config;
};