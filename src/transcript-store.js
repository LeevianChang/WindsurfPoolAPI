import { existsSync, mkdirSync, readFileSync, appendFileSync, renameSync, writeFileSync } from 'fs';
import { dirname, join } from 'path';

const STORE_ENABLED = process.env.TRANSCRIPT_STORE_ENABLED !== '0';
const STORE_PATH = process.env.TRANSCRIPT_STORE_PATH || join(process.cwd(), 'data', 'transcripts.jsonl');
const MAX_MESSAGES = parseInt(process.env.TRANSCRIPT_MAX_MESSAGES || '80', 10);
const MAX_BYTES = parseInt(process.env.TRANSCRIPT_MAX_BYTES || String(900 * 1024), 10);
const FLUSH_COMPACT_EVERY = parseInt(process.env.TRANSCRIPT_COMPACT_EVERY || '200', 10);

const sessions = new Map();
let writesSinceCompact = 0;

function ensureDir() {
  mkdirSync(dirname(STORE_PATH), { recursive: true });
}

function contentBytes(message) {
  return Buffer.byteLength(JSON.stringify(message || {}), 'utf8');
}

function trimMessages(messages) {
  const out = Array.isArray(messages) ? messages.slice(-MAX_MESSAGES) : [];
  let bytes = out.reduce((n, m) => n + contentBytes(m), 0);
  while (out.length > 1 && bytes > MAX_BYTES) {
    const removed = out.shift();
    bytes -= contentBytes(removed);
  }
  return out;
}

function normalizeMessage(m) {
  if (!m || typeof m !== 'object') return null;
  const role = m.role;
  if (!['system', 'user', 'assistant', 'tool'].includes(role)) return null;
  const out = { role, content: m.content ?? '' };
  if (Array.isArray(m.tool_calls)) out.tool_calls = m.tool_calls;
  if (typeof m.tool_call_id === 'string') out.tool_call_id = m.tool_call_id;
  if (m.name != null) out.name = String(m.name);
  return out;
}

function appendRecord(record) {
  if (!STORE_ENABLED) return;
  ensureDir();
  appendFileSync(STORE_PATH, JSON.stringify(record) + '\n');
  writesSinceCompact++;
  if (writesSinceCompact >= FLUSH_COMPACT_EVERY) compactTranscripts();
}

export function loadTranscripts() {
  if (!STORE_ENABLED || !existsSync(STORE_PATH)) return;
  const lines = readFileSync(STORE_PATH, 'utf8').split('\n');
  for (const line of lines) {
    if (!line.trim()) continue;
    try {
      const rec = JSON.parse(line);
      if (!rec.sessionKey) continue;
      if (rec.type === 'replace' && Array.isArray(rec.messages)) {
        sessions.set(rec.sessionKey, { messages: trimMessages(rec.messages.map(normalizeMessage).filter(Boolean)), updatedAt: rec.ts || Date.now() });
      } else if (rec.type === 'append' && rec.message) {
        const cur = sessions.get(rec.sessionKey) || { messages: [], updatedAt: 0 };
        const msg = normalizeMessage(rec.message);
        if (msg) {
          cur.messages = trimMessages([...cur.messages, msg]);
          cur.updatedAt = rec.ts || Date.now();
          sessions.set(rec.sessionKey, cur);
        }
      }
    } catch {}
  }
}

export function compactTranscripts() {
  if (!STORE_ENABLED) return;
  ensureDir();
  const tmp = STORE_PATH + '.tmp';
  const lines = [];
  for (const [sessionKey, entry] of sessions) {
    lines.push(JSON.stringify({ type: 'replace', sessionKey, messages: trimMessages(entry.messages), ts: entry.updatedAt || Date.now() }));
  }
  writeFileSync(tmp, lines.join('\n') + (lines.length ? '\n' : ''));
  renameSync(tmp, STORE_PATH);
  writesSinceCompact = 0;
}

export function getTranscript(sessionKey) {
  if (!STORE_ENABLED || !String(sessionKey || '').startsWith('session:')) return [];
  return sessions.get(sessionKey)?.messages?.slice() || [];
}

export function mergeTranscriptMessages(sessionKey, messages) {
  if (!STORE_ENABLED || !String(sessionKey || '').startsWith('session:') || !Array.isArray(messages)) return messages;
  const prior = getTranscript(sessionKey);
  if (!prior.length || messages.length > 1) return messages;
  return trimMessages([...prior, ...messages.map(normalizeMessage).filter(Boolean)]);
}

export function replaceTranscript(sessionKey, messages) {
  if (!STORE_ENABLED || !String(sessionKey || '').startsWith('session:') || !Array.isArray(messages)) return;
  const normalized = trimMessages(messages.map(normalizeMessage).filter(Boolean));
  sessions.set(sessionKey, { messages: normalized, updatedAt: Date.now() });
  appendRecord({ type: 'replace', sessionKey, messages: normalized, ts: Date.now() });
}

export function appendTranscriptMessage(sessionKey, message) {
  if (!STORE_ENABLED || !String(sessionKey || '').startsWith('session:')) return;
  const msg = normalizeMessage(message);
  if (!msg) return;
  const cur = sessions.get(sessionKey) || { messages: [], updatedAt: 0 };
  cur.messages = trimMessages([...cur.messages, msg]);
  cur.updatedAt = Date.now();
  sessions.set(sessionKey, cur);
  appendRecord({ type: 'append', sessionKey, message: msg, ts: cur.updatedAt });
}

loadTranscripts();
