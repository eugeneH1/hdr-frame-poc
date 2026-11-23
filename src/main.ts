import './style.css';
import { mountWorkbench } from './ui/workbench.ts';
import { logInfo } from './diagnostics/logger.ts';

/**
 * Application entry point.
 *
 * Responsibility: mount the plain-DOM workbench into `#app`. All rendering logic
 * lives behind `runFramePipeline`, which the workbench invokes on Start; nothing
 * beyond mounting (and dev-server teardown) happens here.
 */

const root = document.querySelector<HTMLDivElement>('#app');

if (!root) {
  throw new Error('main: expected a #app root element in index.html');
}

const workbench = mountWorkbench({ root });
logInfo('app', 'Workbench mounted');

// Tear down the active run and DOM on Vite HMR updates so a reload cannot leave
// an AbortController or GPU device alive behind a detached workbench.
if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    logInfo('app', 'HMR dispose: unmounting workbench');
    workbench.unmount();
  });
}
