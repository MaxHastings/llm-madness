import { initNav } from './modules/nav.js';
import { initPipeline } from './modules/pipeline.js';
import { initRuns } from './modules/runs.js';
import { initSession } from './modules/session.js';
import { initPrompt } from './modules/prompt.js';
import { initTraining } from './modules/training.js';
import { initInspect } from './modules/inspect.js';
import { initDatasets } from './modules/datasets.js';

initNav();
initSession();
initPrompt();
initTraining();
initInspect();
initPipeline();
initRuns();
initDatasets();
