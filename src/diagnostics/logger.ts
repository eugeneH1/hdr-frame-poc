/**
 * Console diagnostics for the frame pipeline and its GPU/encoder dependencies.
 *
 * Responsibility: provide a single, consistently prefixed logging surface so a
 * run that stalls or produces fewer frames than requested leaves a readable
 * trail in the browser console. Every line is tagged with a scope
 * (`pipeline`, `gpu`, `readback`, `encoder`, `workbench`) and the wall-clock
 * seconds since the app started, so filtering by scope plus the timestamped
 * prelude is enough to locate a hang.
 *
 * Verbosity:
 * - `info`/`debug` lines are verbose and enabled by default.
 * - `warn`/`error` lines are always emitted; they must survive `?verbose=0`.
 * - `?verbose=0` (or `localStorage["hdr-frame-poc.verbose"] = "0"`) disables
 *   verbose output; `?verbose=1` forces it on. URL parameters win over storage
 *   so a single reload can override a persisted preference.
 *
 * This module deliberately depends on nothing in the project so every layer can
 * import it without creating cycles.
 */

const PREFIX = '[hdr-frame-poc';
const VERBOSE_STORAGE_KEY = 'hdr-frame-poc.verbose';
const VERBOSE_QUERY_PARAM = 'verbose';

/** Structured, console-inspectable context attached to a log line. */
export type LogDetails = Record<string, unknown>;

const startedAt = typeof performance !== 'undefined' ? performance.now() : 0;

function parsePreference(raw: string | null): boolean | null {
  if (raw === null) return null;
  const value = raw.trim().toLowerCase();
  if (value === '' || value === '1' || value === 'true' || value === 'on') return true;
  if (value === '0' || value === 'false' || value === 'off') return false;
  return null;
}

function readUrlPreference(): boolean | null {
  if (typeof location === 'undefined') return null;
  try {
    return parsePreference(new URLSearchParams(location.search).get(VERBOSE_QUERY_PARAM));
  } catch {
    return null;
  }
}

function readStoragePreference(): boolean | null {
  if (typeof localStorage === 'undefined') return null;
  try {
    return parsePreference(localStorage.getItem(VERBOSE_STORAGE_KEY));
  } catch {
    // Storage can be unavailable in private/partitioned contexts.
    return null;
  }
}

let verboseEnabled = readUrlPreference() ?? readStoragePreference() ?? true;

/** Whether verbose (`info`/`debug`) lines are currently emitted. */
export function isVerboseLoggingEnabled(): boolean {
  return verboseEnabled;
}

/**
 * Enables or disables verbose logging for this session and persists the choice.
 * `warn`/`error` lines are unaffected.
 */
export function setVerboseLogging(enabled: boolean): void {
  verboseEnabled = enabled;
  try {
    localStorage.setItem(VERBOSE_STORAGE_KEY, enabled ? '1' : '0');
  } catch {
    // Best-effort persistence only.
  }
  // Emit directly: `logInfo` would be suppressed when disabling.
  emit('log', 'diagnostics', `Verbose logging ${enabled ? 'enabled' : 'disabled'}`);
}

function header(scope: string): string {
  const elapsedSeconds =
    typeof performance !== 'undefined' ? (performance.now() - startedAt) / 1000 : 0;
  return `${PREFIX}:${scope} +${elapsedSeconds.toFixed(3)}s]`;
}

function emit(
  method: 'log' | 'warn' | 'error',
  scope: string,
  message: string,
  error?: unknown,
  details?: LogDetails,
): void {
  const line = `${header(scope)} ${message}`;
  const args: unknown[] = [line];
  if (error !== undefined) args.push(error);
  if (details && Object.keys(details).length > 0) args.push(details);
  console[method](...args);
}

/** Verbose detail, e.g. one line per frame. Hidden when verbose is disabled. */
export function logDebug(scope: string, message: string, details?: LogDetails): void {
  if (!verboseEnabled) return;
  emit('log', scope, message, undefined, details);
}

/** Verbose lifecycle event. Hidden when verbose is disabled. */
export function logInfo(scope: string, message: string, details?: LogDetails): void {
  if (!verboseEnabled) return;
  emit('log', scope, message, undefined, details);
}

/** Non-fatal problem. Always emitted so silent degradation is visible. */
export function logWarn(scope: string, message: string, details?: LogDetails): void {
  emit('warn', scope, message, undefined, details);
}

/** Fatal or near-fatal failure. Always emitted. */
export function logError(
  scope: string,
  message: string,
  error?: unknown,
  details?: LogDetails,
): void {
  emit('error', scope, message, error, details);
}

/** Current verbose preference as a compact value for diagnostic dumps. */
export function describeLogging(): string {
  return verboseEnabled ? 'verbose' : 'warnings-only';
}
