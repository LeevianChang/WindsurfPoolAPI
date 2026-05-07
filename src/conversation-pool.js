/**
 * Cascade conversation reuse pool (experimental).
 *
 * Goal: when a multi-turn chat continues a previous exchange, reuse the same
 * Windsurf `cascade_id` instead of starting a fresh one. This lets the
 * Windsurf backend keep its own per-cascade context cached — we avoid
 * resending the full history on each turn and the server responds faster.
 *
 * The key is a semantic fingerprint of the caller-visible trajectory up to
 * (but not including) the newest user/tool turn. It includes assistant text,
 * assistant tool calls, normalized system prompt, and model/caller scope so
 * we resume only when the upstream cascade should already be at that state.
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

const KEY_VERSION = 2;

// fingerprint -> { cascadeId, sessionId, lsPort, apiKey, stepOffset, generatorOffset, lastUserHash, requestShapeHash, createdAt, lastAccess }
const _pool = new Map();
// sessionKey -> latest fingerprint. This mirrors sub2api's sticky-session
// layer, but session fallback is opt-in and restricted to explicit session
// identifiers; content-derived fallback keys are too broad for cascade reuse.
const _sessionIndex = new Map();

const stats = { hits: 0, sessionHits: 0, misses: 0, stores: 0, evictions: 0, expired: 0 };

function sha256(s) {
  return createHash('sha256').update(s).digest('hex');
}

function shortDigest(s, n = 32) {
  return sha256(String(s ?? '')).slice(0, n);
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

function normalizeSystemPromptForHash(s) {
  let out = String(s || '')
    .replace(/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?/g, '<ts>')
    .replace(/\b(Today(?:'s)?\s+(?:date|is)(?:\s+is)?\s*[:\-]?\s*)\d{4}-\d{2}-\d{2}/gi, '$1<date>')
    .replace(/(?<!\d)\d{4}-\d{2}-\d{2}(?!\d|T)/g, '<date>')
    .replace(/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/gi, '<uuid>')
    .replace(/(^[ \t]*[-*•]?\s*(?:Working\s+directory|Current\s+working\s+directory|cwd|CWD)\s*[:：])[^\n]*/gim, '$1 <cwd>')
    .replace(/(^[ \t]*[-*•]?\s*(?:Current\s+(?:date|time)|Time)\s*[:：])[^\n]*/gim, '$1 <time>')
    .replace(/(^[ \t]*[-*•]?\s*(?:Session\s*ID|sessionId|session_id)\s*[:：])[^\n]*/gim, '$1 <sessionid>')
    .replace(/(?<![\d.])(?:1[7-9]|20)\d{8}(?:\d{3})?(?![\d.])/g, '<epoch>');

  const nextHeading = '(?:Status|Recent commits|Recent files|gitStatus|Current branch|Main branch|Git user)\\s*:';
  const blockEnd = `(?=^[ \\t]*${nextHeading}|^\\s*$|$(?![\\s\\S]))`;
  out = out.replace(new RegExp(`^([ \\t]*Status\\s*:)[ \\t]*\\n[\\s\\S]*?${blockEnd}`, 'gim'), '$1\n<git-status>\n');
  out = out.replace(new RegExp(`^([ \\t]*Recent commits\\s*:)[ \\t]*\\n[\\s\\S]*?${blockEnd}`, 'gim'), '$1\n<recent-commits>\n');
  out = out.replace(new RegExp(`^([ \\t]*Recent files\\s*:)[ \\t]*\\n[\\s\\S]*?${blockEnd}`, 'gim'), '$1\n<recent-files>\n');
  out = out.replace(/(?<![`'"\w])(?=[a-f0-9]*\d)(?=[a-f0-9]*[a-f])[a-f0-9]{7,12}(?![`'"\w])/gi, '<gitsha>');
  return out;
}

function stableStringify(v) {
  if (v === null || typeof v !== 'object') return JSON.stringify(v);
  if (Array.isArray(v)) return '[' + v.map(stableStringify).join(',') + ']';
  const keys = Object.keys(v).sort();
  return '{' + keys.map(k => JSON.stringify(k) + ':' + stableStringify(v[k])).join(',') + '}';
}

function canonicalContent(content, { system = false } = {}) {
  const normalizeText = (text) => {
    const stripped = stripMetaTags(String(text || ''));
    return system ? normalizeSystemPromptForHash(stripped) : stripped;
  };
  if (typeof content === 'string') return [{ type: 'text', text: normalizeText(content) }];
  if (!Array.isArray(content)) return [{ type: 'json', json: stableStringify(content ?? '') }];
  return content.map(part => {
    if (typeof part?.text === 'string') return { type: 'text', text: normalizeText(part.text) };
    if (typeof part === 'string') return { type: 'text', text: normalizeText(part) };
    const type = String(part?.type || '').toLowerCase();
    const url = part?.image_url?.url || part?.url || part?.source?.url || '';
    if (type === 'image' || type === 'image_url' || type === 'input_image') {
      if (typeof url === 'string' && url.startsWith('data:')) {
        const comma = url.indexOf(',');
        return { type: 'image', hash: shortDigest(comma >= 0 ? url.slice(comma + 1) : url, 16) };
      }
      if (typeof part?.source?.data === 'string') return { type: 'image', hash: shortDigest(part.source.data, 16) };
      return { type: 'image', url: String(url || part?.source?.file_id || '') };
    }
    return { type: type || 'unknown', json: stableStringify(part ?? '') };
  });
}

function hasToolCalls(message) {
  return Array.isArray(message?.tool_calls) && message.tool_calls.length > 0;
}

function projectAssistantToolCalls(message) {
  if (!hasToolCalls(message)) return [];
  return message.tool_calls.map(tc => {
    const name = tc?.function?.name || tc?.name || '';
    const args = tc?.function?.arguments ?? tc?.arguments ?? tc?.input ?? '';
    let canonicalArgs = args;
    if (typeof args === 'string') {
      try { canonicalArgs = stableStringify(JSON.parse(args)); } catch { canonicalArgs = args; }
    } else {
      canonicalArgs = stableStringify(args ?? null);
    }
    return { name, args: canonicalArgs };
  });
}

function toolContextDigest(opts = {}) {
  if (!opts.emulateTools) return '';
  const tools = (Array.isArray(opts.tools) ? opts.tools.map(t => {
    const fn = t?.function || t;
    return {
      name: fn?.name || '',
      description: fn?.description || '',
      parameters: fn?.parameters ?? fn?.input_schema ?? null,
    };
  }) : []).sort((a, b) => (a.name || '').localeCompare(b.name || ''));
  return shortDigest(stableStringify({
    tools,
    tool_choice: opts.toolChoice ?? null,
    preambleHash: opts.toolPreamble ? shortDigest(opts.toolPreamble, 16) : '',
    preambleTier: opts.preambleTier ?? null,
  }));
}

/**
 * Fingerprint for "resume this conversation". Hash stable caller-visible
 * turns plus system instructions. Assistant text is included so the stored
 * post-turn key matches clients that send [u1, a1, u2] on the next request.
 */
function projectMessage(message) {
  const role = message?.role;
  if (role === 'system') return { role: 'system', content: canonicalContent(message.content, { system: true }) };
  if (role === 'user') return { role: 'user', content: canonicalContent(message.content) };
  if (role === 'tool') {
    return {
      role: 'tool_result',
      tool_call_id: typeof message?.tool_call_id === 'string' ? message.tool_call_id : '',
      content: canonicalContent(message.content),
    };
  }
  if (role === 'assistant') {
    const blocks = canonicalContent(message.content);
    const text = blocks
      .filter(b => b.type === 'text')
      .map(b => (b.text || '').replace(/\s+/g, ' ').trim())
      .join('\n')
      .trim();
    return { role: 'assistant', text, tool_calls: projectAssistantToolCalls(message) };
  }
  return { role: String(role || 'unknown'), content: canonicalContent(message?.content) };
}

function priorTurnsForBefore(messages) {
  if (!Array.isArray(messages)) return null;
  let newest = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    const role = messages[i]?.role;
    if (role === 'user' || role === 'tool') { newest = i; break; }
  }
  if (newest <= 0) return null;
  return messages.slice(0, newest);
}

function keyPayload(messages, modelKey, callerKey, scope, opts = {}) {
  const turns = scope === 'after' ? messages : priorTurnsForBefore(messages);
  if (!Array.isArray(turns) || !turns.length) return null;
  return stableStringify({
    v: KEY_VERSION,
    caller: String(callerKey || ''),
    model: String(modelKey || ''),
    route: opts.route || 'chat',
    tools: toolContextDigest(opts),
    turns: turns.map(projectMessage),
  });
}

export function fingerprintBefore(messages, modelKey = '', callerKey = '', opts = {}) {
  const payload = keyPayload(messages, modelKey, callerKey, 'before', opts);
  return payload ? sha256(payload) : null;
}

export function fingerprintAfter(messages, modelKey = '', callerKey = '', opts = {}) {
  if (!Array.isArray(messages) || !messages.length) return null;
  const payload = keyPayload(messages, modelKey, callerKey, 'after', opts);
  return payload ? sha256(payload) : null;
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
    if (now - e.lastAccess > effectiveTtl(e)) { deleteFingerprint(fp); stats.expired++; }
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

function effectiveTtl(entry) {
  const hint = Number(entry?.ttlHintMs);
  return Number.isFinite(hint) && hint > 0 ? hint : POOL_TTL_MS;
}

function validEntry(entry, callerKey) {
  if (!entry) return false;
  if (entry.callerKey && callerKey && entry.callerKey !== callerKey) return false;
  return Date.now() - entry.lastAccess <= effectiveTtl(entry);
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
      if (Date.now() - entry.lastAccess > effectiveTtl(entry)) stats.expired++;
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
      if (entry && Date.now() - entry.lastAccess > effectiveTtl(entry)) stats.expired++;
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
  const fingerprints = Array.isArray(fingerprint)
    ? fingerprint.filter(fp => typeof fp === 'string' && fp)
    : [fingerprint];
  if (!fingerprints.length) return;
  const now = Date.now();
  for (const fp of fingerprints) {
    _pool.set(fp, {
      cascadeId: entry.cascadeId,
      sessionId: entry.sessionId,
      lsPort: entry.lsPort,
      apiKey: entry.apiKey,
      stepOffset: Number.isFinite(entry.stepOffset) ? entry.stepOffset : 0,
      generatorOffset: Number.isFinite(entry.generatorOffset) ? entry.generatorOffset : 0,
      lastUserHash: entry.lastUserHash || '',
      requestShapeHash: entry.requestShapeHash || '',
      callerKey: callerKey || entry.callerKey || '',
      sessionKey: sessionKey || entry.sessionKey || '',
      createdAt: entry.createdAt || now,
      lastAccess: now,
      ...(Number.isFinite(entry.ttlHintMs) && entry.ttlHintMs > 0 ? { ttlHintMs: entry.ttlHintMs } : {}),
    });
  }
  const key = sessionKey || entry.sessionKey || '';
  if (key) _sessionIndex.set(key, fingerprints[0]);
  stats.stores++;
  if (fingerprints.length > 1) stats.aliasWrites = (stats.aliasWrites || 0) + fingerprints.length - 1;
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
