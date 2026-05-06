/**
 * Cascade conversation reuse pool (experimental).
 *
 * Goal: when a multi-turn chat continues a previous exchange, reuse the same
 * Windsurf `cascade_id` instead of starting a fresh one. This lets the
 * Windsurf backend keep its own per-cascade context cached — we avoid
 * resending the full history on each turn and the server responds faster.
 *
 * The key is a "fingerprint" of the conversation up to (but not including)
 * the newest user message. A client sending [u1, a1, u2] looks up fp([u1, a1]);
 * a hit means we already drove the cascade to exactly that state. We then
 * `SendUserCascadeMessage(u2)` on the stored cascade_id and, on success,
 * re-store the entry under fp([u1, a1, u2, a2]) for the next turn.
 *
 * Safety rails:
 *   - Entries are pinned to a specific (apiKey, lsPort) pair. We must reuse
 *     the same LS and the same account or the cascade_id is meaningless.
 *   - A checked-out entry is removed from the pool. Concurrent second request
 *     with the same fingerprint falls back to a fresh cascade.
 *   - TTL defaults to 30 min (override with CASCADE_POOL_TTL_MS); LRU eviction
 *     at 500 entries.
 */

import { createHash } from 'crypto';

function positiveIntEnv(name, fallback) {
  const n = parseInt(process.env[name] || '', 10);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

const POOL_TTL_MS = positiveIntEnv('CASCADE_POOL_TTL_MS', 30 * 60 * 1000);
const POOL_MAX = 500;

// fingerprint -> { cascadeId, sessionId, lsPort, apiKey, lastUserHash, requestShapeHash, createdAt, lastAccess }
const _pool = new Map();
// sessionKey -> latest fingerprint. This mirrors sub2api's sticky-session
// layer, but session fallback is opt-in and restricted to explicit session
// identifiers; content-derived fallback keys are too broad for cascade reuse.
const _sessionIndex = new Map();

const stats = { hits: 0, sessionHits: 0, misses: 0, stores: 0, evictions: 0, expired: 0 };

function sha256(s) {
  return createHash('sha256').update(s).digest('hex');
}

function messageContentString(message) {
  if (!message) return '';
  if (typeof message.content === 'string') return message.content;
  if (Array.isArray(message.content)) {
    return message.content.map(p => (typeof p?.text === 'string' ? p.text : JSON.stringify(p))).join('');
  }
  return JSON.stringify(message.content ?? '');
}

// Client-injected meta tags whose bodies change every turn (cwd snapshot,
// todo state, current time, hook output, slash-command echo). If we hash
// these, the fingerprint drifts even when the real user text is unchanged
// and Cascade reuse silently falls back to fresh for every call (#24).
const META_TAG_NAMES = [
  'system-reminder',
  'command-message',
  'command-name',
  'command-args',
  'local-command-stdout',
  'local-command-stderr',
  'user-prompt-submit-hook',
  'analysis',
  'summary',
  'example',
];
const META_TAG_RE = new RegExp(
  `<(${META_TAG_NAMES.join('|')})[^>]*>[\\s\\S]*?</\\1>`,
  'g'
);

function stripMetaTags(s) {
  if (typeof s !== 'string' || !s) return s;
  return s.replace(META_TAG_RE, '').replace(/[ \t]+\n/g, '\n').replace(/\n{3,}/g, '\n\n').trim();
}

function canonicalise(messages) {
  return messages.map(m => {
    let raw;
    if (typeof m.content === 'string') raw = m.content;
    else if (Array.isArray(m.content)) raw = m.content.map(p => (typeof p?.text === 'string' ? p.text : JSON.stringify(p))).join('');
    else raw = JSON.stringify(m.content ?? '');
    return { role: m.role, content: stripMetaTags(raw) };
  });
}

function hasToolCalls(message) {
  return Array.isArray(message?.tool_calls) && message.tool_calls.length > 0;
}

/**
 * Fingerprint for "resume this conversation". Hash stable caller-visible turns
 * plus system instructions. Plain assistant text is excluded because clients
 * may restructure it between turns, but assistant tool_call turns are kept so
 * tool-result conversations resume the exact cascade that requested the tool.
 */
function stableTurns(messages) {
  return messages
    .filter(m => m.role === 'system' || m.role === 'user' || m.role === 'tool' || (m.role === 'assistant' && hasToolCalls(m)))
    .map(m => {
      if (m.role === 'tool') return { ...m, role: 'tool_result' };
      if (m.role === 'assistant' && hasToolCalls(m)) {
        return { role: 'assistant_tool_calls', content: JSON.stringify(m.tool_calls) };
      }
      return m;
    });
}

export function fingerprintBefore(messages, modelKey = '', callerKey = '') {
  if (!Array.isArray(messages) || messages.length < 2) return null;
  const turns = stableTurns(messages);
  if (turns.length < 2) return null;
  return sha256(String(callerKey || '') + '\0' + modelKey + '\0' + JSON.stringify(canonicalise(turns.slice(0, -1))));
}

export function fingerprintAfter(messages, modelKey = '', callerKey = '') {
  const turns = stableTurns(messages);
  if (!turns.length) return null;
  return sha256(String(callerKey || '') + '\0' + modelKey + '\0' + JSON.stringify(canonicalise(turns)));
}

export function latestUserHash(messages, modelKey = '', callerKey = '') {
  if (!Array.isArray(messages)) return '';
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]?.role !== 'user') continue;
    const content = stripMetaTags(messageContentString(messages[i]));
    return sha256(String(callerKey || '') + '\0' + modelKey + '\0' + content);
  }
  return '';
}

export function requestShapeHash(body = {}, modelKey = '', callerKey = '') {
  if (!body || typeof body !== 'object') return '';
  const shape = {};
  for (const key of Object.keys(body).sort()) {
    if (key === 'messages' || key === '_source') continue;
    const value = body[key];
    if (typeof value === 'undefined') continue;
    shape[key] = value;
  }
  return sha256(String(callerKey || '') + '\0' + modelKey + '\0' + JSON.stringify(shape));
}

function prune(now) {
  for (const [fp, e] of _pool) {
    if (now - e.lastAccess > POOL_TTL_MS) { deleteFingerprint(fp); stats.expired++; }
  }
  if (_pool.size <= POOL_MAX) return;
  const entries = [..._pool.entries()].sort((a, b) => a[1].lastAccess - b[1].lastAccess);
  const toDrop = entries.length - POOL_MAX;
  for (let i = 0; i < toDrop; i++) {
    deleteFingerprint(entries[i][0]);
    stats.evictions++;
  }
}

function deleteFingerprint(fingerprint) {
  const entry = _pool.get(fingerprint);
  _pool.delete(fingerprint);
  if (entry?.sessionKey && _sessionIndex.get(entry.sessionKey) === fingerprint) {
    _sessionIndex.delete(entry.sessionKey);
  }
}

function validEntry(entry, callerKey) {
  if (!entry) return false;
  if (entry.callerKey && callerKey && entry.callerKey !== callerKey) return false;
  return Date.now() - entry.lastAccess <= POOL_TTL_MS;
}

/**
 * Check out a conversation if we have a matching fingerprint AND the caller
 * is willing to use the same (apiKey, lsPort) we stored. Removes the entry
 * from the pool — caller is expected to call `checkin()` with a new
 * fingerprint on success (or just drop it on failure and a fresh cascade
 * will be created next turn).
 */
export function checkout(fingerprint, callerKey = '', sessionKey = '', opts = {}) {
  if (fingerprint) {
    const entry = _pool.get(fingerprint);
    if (validEntry(entry, callerKey)) {
      const currentLastUserHash = opts?.lastUserHash || '';
      if (currentLastUserHash && entry.lastUserHash && currentLastUserHash === entry.lastUserHash) {
        deleteFingerprint(fingerprint);
        stats.misses++;
        return null;
      }
      const currentRequestShapeHash = opts?.requestShapeHash || '';
      if (currentRequestShapeHash && entry.requestShapeHash && currentRequestShapeHash !== entry.requestShapeHash) {
        deleteFingerprint(fingerprint);
        stats.misses++;
        return null;
      }
      deleteFingerprint(fingerprint);
      stats.hits++;
      return { ...entry, reuseReason: 'fingerprint' };
    }
    if (entry) {
      if (Date.now() - entry.lastAccess > POOL_TTL_MS) stats.expired++;
      deleteFingerprint(fingerprint);
    }
  }

  const allowSessionFallback = opts?.allowSessionFallback === true && String(sessionKey || '').startsWith('session:');
  if (allowSessionFallback) {
    const sessionFp = _sessionIndex.get(sessionKey);
    const entry = sessionFp ? _pool.get(sessionFp) : null;
    if (validEntry(entry, callerKey)) {
      const currentLastUserHash = opts?.lastUserHash || '';
      if (currentLastUserHash && entry.lastUserHash && currentLastUserHash === entry.lastUserHash) {
        deleteFingerprint(sessionFp);
        stats.misses++;
        return null;
      }
      const currentRequestShapeHash = opts?.requestShapeHash || '';
      if (currentRequestShapeHash && entry.requestShapeHash && currentRequestShapeHash !== entry.requestShapeHash) {
        deleteFingerprint(sessionFp);
        stats.misses++;
        return null;
      }
      deleteFingerprint(sessionFp);
      stats.sessionHits++;
      return { ...entry, reuseReason: 'session' };
    }
    if (sessionFp) {
      if (entry && Date.now() - entry.lastAccess > POOL_TTL_MS) stats.expired++;
      deleteFingerprint(sessionFp);
    }
  }

  stats.misses++;
  return null;
}

/**
 * Store (or restore) a conversation entry under a new fingerprint.
 */
export function checkin(fingerprint, entry, callerKey = '', sessionKey = '') {
  if (!fingerprint || !entry) return;
  const now = Date.now();
  _pool.set(fingerprint, {
    cascadeId: entry.cascadeId,
    sessionId: entry.sessionId,
    lsPort: entry.lsPort,
    apiKey: entry.apiKey,
    lastUserHash: entry.lastUserHash || '',
    requestShapeHash: entry.requestShapeHash || '',
    callerKey: callerKey || entry.callerKey || '',
    sessionKey: sessionKey || entry.sessionKey || '',
    createdAt: entry.createdAt || now,
    lastAccess: now,
  });
  const key = sessionKey || entry.sessionKey || '';
  if (key) _sessionIndex.set(key, fingerprint);
  stats.stores++;
  prune(now);
}

/**
 * Drop any entries that belong to a (apiKey, lsPort) pair that just went
 * away (account removed, LS restarted). Keeps the pool honest.
 */
export function invalidateFor({ apiKey, lsPort }) {
  let dropped = 0;
  for (const [fp, e] of _pool) {
    if ((apiKey && e.apiKey === apiKey) || (lsPort && e.lsPort === lsPort)) {
      deleteFingerprint(fp);
      dropped++;
    }
  }
  return dropped;
}

export function poolStats() {
  return {
    size: _pool.size,
    sessionIndexSize: _sessionIndex.size,
    maxSize: POOL_MAX,
    ttlMs: POOL_TTL_MS,
    ...stats,
    hitRate: stats.hits + stats.sessionHits + stats.misses > 0
      ? (((stats.hits + stats.sessionHits) / (stats.hits + stats.sessionHits + stats.misses)) * 100).toFixed(1)
      : '0.0',
  };
}

export function poolClear() {
  const n = _pool.size;
  _pool.clear();
  _sessionIndex.clear();
  return n;
}

// Background prune — expired entries accumulate when there are no
// checkin() calls. .unref() so this timer never holds the process open.
setInterval(() => prune(Date.now()), 5 * 60 * 1000).unref();
