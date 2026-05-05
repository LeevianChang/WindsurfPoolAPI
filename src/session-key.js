import { createHash } from 'crypto';

function sha256Hex(value) {
  return createHash('sha256').update(String(value || '')).digest('hex');
}

function headerValue(req, name) {
  const value = req?.headers?.[name.toLowerCase()];
  return Array.isArray(value) ? value[0] : String(value || '');
}

function promptCacheKey(body) {
  if (!body || typeof body !== 'object') return '';
  return String(body.prompt_cache_key || body.promptCacheKey || '').trim();
}

function metadataUserId(body) {
  if (!body || typeof body !== 'object') return '';
  const metadata = body.metadata;
  if (!metadata || typeof metadata !== 'object') return '';
  return String(metadata.user_id || metadata.userId || '').trim();
}

function contentSeed(body) {
  if (!body || typeof body !== 'object') return '';
  const parts = [body.model || ''];
  if (body.system) parts.push(typeof body.system === 'string' ? body.system : JSON.stringify(body.system));
  if (Array.isArray(body.tools) && body.tools.length) parts.push(JSON.stringify(body.tools));
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const firstUser = messages.find(m => m?.role === 'user');
  if (firstUser) parts.push(typeof firstUser.content === 'string' ? firstUser.content : JSON.stringify(firstUser.content ?? ''));
  const seed = parts.filter(Boolean).join('\0');
  return seed ? `content:${sha256Hex(seed).slice(0, 32)}` : '';
}

export function sessionKeyFromRequest(req, body, callerKey = '') {
  const explicit = [
    headerValue(req, 'session_id'),
    headerValue(req, 'conversation_id'),
    headerValue(req, 'x-session-id'),
    headerValue(req, 'x-dashboard-session'),
    promptCacheKey(body),
    metadataUserId(body),
  ].map(v => String(v || '').trim()).find(Boolean);

  if (explicit) {
    return `session:${sha256Hex(`${callerKey}\0${explicit}`).slice(0, 32)}`;
  }

  const fallback = contentSeed(body);
  return fallback ? `fallback:${sha256Hex(`${callerKey}\0${fallback}`).slice(0, 32)}` : '';
}

export function sessionKeyFromAnthropicBody(body, callerKey = '') {
  const explicit = [promptCacheKey(body), metadataUserId(body)].map(v => String(v || '').trim()).find(Boolean);
  if (explicit) return `session:${sha256Hex(`${callerKey}\0${explicit}`).slice(0, 32)}`;
  const fallback = contentSeed(body);
  return fallback ? `fallback:${sha256Hex(`${callerKey}\0${fallback}`).slice(0, 32)}` : '';
}
