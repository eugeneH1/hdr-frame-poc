import './style.css';
import { mountWorkbench } from './ui/workbench.ts';

/**
 * Application entry point.
 *
 * Responsibility: mount the plain-DOM workbench into `#app`. All rendering logic
 * lives behind `runFramePipeline`, which the workbench invokes on Run; nothing
 * beyond mounting happens here.
 */

const root = document.querySelector<HTMLDivElement>('#app');

if (!root) {
  throw new Error('main: expected a #app root element in index.html');
}

mountWorkbench({ root });
