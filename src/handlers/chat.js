/**
 * POST /v1/chat/completions — OpenAI-compatible chat completions.
 * Routes to RawGetChatMessage (legacy) or Cascade (premium) based on model type.
 */

import { randomUUID } from 'crypto';
import { WindsurfClient } from '../client.js';
import { waitForApiKey, acquireAccountByKey, reportError, reportSuccess, markRateLimited, reportInternalError, updateCapability, getAccountList, isAllRateLimited } from '../auth.js';
import { resolveModelWithOptions, getModelInfo } from '../models.js';
import { getLsFor, ensureLs } from '../langserver.js';
import { config, log } from '../config.js';
import { recordRequest } from '../dashboard/stats.js';
import { isModelAllowed } from '../dashboard/model-access.js';
import { cacheKey, cacheGet, cacheSet } from '../cache.js';
import { isExperimentalEnabled, getIdentityPromptFor } from '../runtime-config.js';
import { checkMessageRateLimit } from '../windsurf-api.js';
import { getEffectiveProxy } from '../dashboard/proxy-config.js';
import {
  fingerprintBefore, fingerprintAfter, latestUserHash, requestShapeHash, checkout as poolCheckout, checkin as poolCheckin,
} from '../conversation-pool.js';
import {
  normalizeMessagesForCascade, ToolCallStreamParser, parseToolCallsFromText,
  buildToolPreambleForProto,
} from './tool-emulation.js';
import { sanitizeText, PathSanitizeStream } from '../sanitize.js';

const HEARTBEAT_MS = 5_000;
const QUEUE_MAX_WAIT_MS = 30_000;

function classifyUpstreamError(err) {
  const message = err?.message || String(err || '');
  const lower = message.toLowerCase();

  if (/content policy|blocked by our content policy|unsafe content|safety policy|policy violation/.test(lower)) {
    return { action: 'return', status: 400, type: 'content_policy_block', tag: 'content_policy_block' };
  }
  if (/context.*(too long|length|limit)|prompt.*too long|input.*too long|token limit|max(?:imum)? context/.test(lower)) {
    return { action: 'return', status: 400, type: 'context_too_long', tag: 'context_too_long' };
  }
  if (/rate limit|rate_limit|too many requests|quota/.test(lower)) {
    return { action: 'switch_account', status: 429, type: 'rate_limit_exceeded', tag: 'rate_limit' };
  }
  if (/unauthenticated|invalid api key|invalid_grant|permission_denied.*account/.test(lower)) {
    return { action: 'switch_account', status: 502, type: 'auth_error', tag: 'auth_error' };
  }
  if (/internal error occurred.*error id|overloaded|temporarily unavailable|timeout|timed out|stalled|unavailable|econnreset/.test(lower)) {
    return { action: 'switch_account', status: 502, type: 'upstream_error', tag: 'transient_error' };
  }
  if (/permission_denied|failed_precondition|model.*not.*available|not entitled|not subscribed|does not have access/.test(lower)) {
    return { action: 'switch_account', status: 403, type: 'model_not_available', tag: 'model_not_available', capabilityFailure: true };
  }

  return { action: 'return', status: err?.isModelError ? 403 : 502, type: err?.isModelError ? 'model_error' : 'upstream_error', tag: err?.isModelError ? 'model_error' : 'upstream_error' };
}

function applyErrorSideEffects(err, apiKey, modelKey) {
  const cls = classifyUpstreamError(err);
  if (cls.tag === 'auth_error') reportError(apiKey);
  if (cls.tag === 'rate_limit') markRateLimited(apiKey, 5 * 60 * 1000, modelKey);
  if (cls.tag === 'transient_error' && /internal error occurred.*error id/i.test(err?.message || '')) {
    reportInternalError(apiKey);
  }
  if (err?.isModelError && cls.capabilityFailure) {
    updateCapability(apiKey, modelKey, false, 'model_error');
  }
  return cls;
}

// ── Language-following reinforcement ──────────────────────────
// Claude Code injects ~100KB of English system prompt + tool definitions
// which drowns out the communication_section language instruction. Detecting
// CJK/JP/KR characters in the user's latest message and appending a brief
// reminder ensures the model sees it at the point of highest attention.
const CJK_RE = /[\u4e00-\u9fff\u3400-\u4dbf]/;
const JP_RE  = /[\u3040-\u309f\u30a0-\u30ff]/;
const KR_RE  = /[\uac00-\ud7af]/;

function detectLanguageHint(msgs) {
  if (!Array.isArray(msgs)) return { text: '', code: '' };
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (msgs[i]?.role !== 'user') continue;
    const c = msgs[i].content;
    const text = typeof c === 'string' ? c
      : Array.isArray(c) ? c.filter(p => p?.type === 'text').map(p => p.text).join('') : '';
    // Check JP/KR FIRST — Japanese text always contains kanji (CJK range)
    // so checking CJK first would false-match Japanese as Chinese.
    if (JP_RE.test(text)) return { text: '[重要: 最後まで日本語だけで返答してください。英語に切り替えないでください。]', code: 'ja' };
    if (KR_RE.test(text)) return { text: '[중요: 끝까지 한국어로만 답변하세요. 영어로 전환하지 마세요.]', code: 'ko' };
    if (CJK_RE.test(text)) return { text: '[重要：请全程只用中文回答。不要在回答过程中切换成英文。]', code: 'zh' };
    return { text: '', code: '' };
  }
  return { text: '', code: '' };
}

function injectLanguageHint(msgs, hintText) {
  if (!Array.isArray(msgs) || !hintText) return;
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (msgs[i]?.role !== 'user') continue;
    msgs[i] = { ...msgs[i] };
    const hint = '\n\n' + hintText;
    if (typeof msgs[i].content === 'string') msgs[i].content += hint;
    else if (Array.isArray(msgs[i].content)) msgs[i].content = [...msgs[i].content, { type: 'text', text: hint }];
    break;
  }
}

// ── Model identity prompt ──────────────────────────────────
function buildIdentitySystemMessage(displayModel, provider) {
  const template = getIdentityPromptFor(provider);
  if (!template) return null;
  return template.replace(/\{model\}/g, displayModel);
}

function genId() {
  return 'chatcmpl-' + randomUUID().replace(/-/g, '').slice(0, 29);
}

// Rough token estimate (~4 chars/token). Used only to populate the
// OpenAI-compatible `usage.prompt_tokens_details.cached_tokens` field so
// upstream billing/dashboards (new-api) can recognise our local cache hits.
function estimateTokens(messages) {
  if (!Array.isArray(messages)) return 0;
  let chars = 0;
  for (const m of messages) {
    if (typeof m?.content === 'string') chars += m.content.length;
    else if (Array.isArray(m?.content)) {
      for (const p of m.content) if (typeof p?.text === 'string') chars += p.text.length;
    }
  }
  return Math.max(1, Math.ceil(chars / 4));
}

function cachedUsage(messages, completionText) {
  const prompt = estimateTokens(messages);
  const completion = Math.max(1, Math.ceil((completionText || '').length / 4));
  return {
    prompt_tokens: prompt,
    completion_tokens: completion,
    total_tokens: prompt + completion,
    prompt_tokens_details: { cached_tokens: prompt },
    cached: true,
  };
}

/**
 * Build an OpenAI-shaped `usage` object, preferring server-reported token
 * counts from Cascade's CortexStepMetadata.model_usage when available, and
 * falling back to the local chars/4 estimate otherwise. Keeps the same shape
 * in both branches so downstream billing doesn't have to care which source
 * produced the numbers.
 *
 * The Cascade backend reports usage as {inputTokens, outputTokens,
 * cacheReadTokens, cacheWriteTokens}. We map them onto the OpenAI shape:
 *   prompt_tokens     = inputTokens + cacheReadTokens + cacheWriteTokens
 *                       (total input tokens the model processed, whether fresh,
 *                       cache-read, or cache-written — matches the OpenAI
 *                       convention where prompt_tokens is the grand total)
 *   completion_tokens = outputTokens
 *   prompt_tokens_details.cached_tokens       = cacheReadTokens
 *   cache_creation_input_tokens (Anthropic ext) = cacheWriteTokens
 */
function buildUsageBody(serverUsage, messages, completionText, thinkingText = '') {
  if (serverUsage && (serverUsage.inputTokens || serverUsage.outputTokens)) {
    const inputTokens = serverUsage.inputTokens || 0;
    const outputTokens = serverUsage.outputTokens || 0;
    const cacheRead = serverUsage.cacheReadTokens || 0;
    const cacheWrite = serverUsage.cacheWriteTokens || 0;
    const promptTotal = inputTokens + cacheRead + cacheWrite;
    return {
      prompt_tokens: promptTotal,
      completion_tokens: outputTokens,
      total_tokens: promptTotal + outputTokens,
      prompt_tokens_details: { cached_tokens: cacheRead },
      cache_creation_input_tokens: cacheWrite,
    };
  }
  const prompt = estimateTokens(messages);
  const completion = Math.max(1, Math.ceil(((completionText || '').length + (thinkingText || '').length) / 4));
  return {
    prompt_tokens: prompt,
    completion_tokens: completion,
    total_tokens: prompt + completion,
    prompt_tokens_details: { cached_tokens: 0 },
  };
}

async function waitForAccount(tried, signal, maxWaitMs = QUEUE_MAX_WAIT_MS, modelKey = null) {
  return waitForApiKey(tried, modelKey, signal, maxWaitMs);
}

export async function handleChatCompletions(body, deps = {}) {
  const {
    model: reqModel,
    messages,
    stream = false,
    max_tokens,
    tools,
    tool_choice,
  } = body;

  const modelKey = resolveModelWithOptions(reqModel || config.defaultModel, body);
  const modelInfo = getModelInfo(modelKey);
  const displayModel = modelInfo?.name || reqModel || config.defaultModel;
  const creditMultiplier = modelInfo?.credit || 0;
  const source = body._source || 'POST /v1/chat/completions';
  const callerKey = deps.callerKey || deps.context?.callerKey || '';
  const sessionKey = deps.sessionKey || deps.context?.sessionKey || '';
  const modelEnum = modelInfo?.enumValue || 0;
  const modelUid = modelInfo?.modelUid || null;
  // Models with a modelUid use the Cascade flow (StartCascade → SendUserCascadeMessage).
  // Legacy RawGetChatMessage only for models with enumValue>0 and NO modelUid.
  // Newer models (gemini-3.0, gpt-5.2, etc.) have both enumValue AND modelUid but
  // their high enum values cause "cannot parse invalid wire-format data" in the
  // legacy proto endpoint. Cascade handles them correctly via uid string.
  const useCascade = !!modelUid;

  // Tool-call emulation: if the client passed OpenAI-style tools[], we rewrite
  // tool-result turns into synthetic user text and inject the tool protocol
  // at the system-prompt level via CascadeConversationalPlannerConfig's
  // tool_calling_section (SectionOverrideConfig, OVERRIDE mode). This is far
  // more reliable than user-message-level injection because NO_TOOL mode's
  // baked-in system prompt tells the model "you have no tools" — which
  // overpowers user-message preambles. The section override replaces that
  // section directly so the model sees our emulated tool definitions as
  // authoritative system instructions.
  const hasTools = Array.isArray(tools) && tools.length > 0;
  const hasToolHistory = Array.isArray(messages) && messages.some(m => m?.role === 'tool' || (m?.role === 'assistant' && Array.isArray(m.tool_calls) && m.tool_calls.length));
  const emulateTools = useCascade && (hasTools || hasToolHistory);
  // Build proto-level preamble (goes into tool_calling_section override);
  // pass empty tools to normalizeMessagesForCascade so it only rewrites
  // role:tool / assistant.tool_calls messages without injecting a user-level
  // preamble (that's now handled at the proto layer).
  const toolPreamble = emulateTools ? buildToolPreambleForProto(tools || [], tool_choice) : '';
  let cascadeMessages = emulateTools
    ? normalizeMessagesForCascade(messages, [])
    : [...messages];

  // ── Model identity prompt injection ──
  // When enabled, prepend a system message so the model identifies itself as
  // the requested model (e.g. "I am Claude Opus 4.6") instead of leaking the
  // Cascade/Windsurf backend identity.
  if (isExperimentalEnabled('modelIdentityPrompt') && modelInfo?.provider) {
    const identityText = buildIdentitySystemMessage(displayModel, modelInfo.provider);
    if (identityText) {
      cascadeMessages = [{ role: 'system', content: identityText }, ...cascadeMessages];
    }
  }

  // Language-following reinforcement: inject hint into latest user message
  // so the model responds in the user's language even when drowned in English
  // system prompts.
  const languageHint = detectLanguageHint(cascadeMessages);
  injectLanguageHint(cascadeMessages, languageHint.text);

  // Global model access control (allowlist / blocklist from dashboard)
  const access = isModelAllowed(modelKey);
  if (!access.allowed) {
    return { status: 403, body: { error: { message: access.reason, type: 'model_blocked' } } };
  }

  // Per-account model routing preflight: if NO active account has this
  // model in its tier ∩ available list, fail fast instead of looping
  // through every account trying to find one. This surfaces tier
  // entitlement and blocklist errors as a clean 403 rather than a 30s
  // queue timeout → pool_exhausted.
  const anyEligible = getAccountList().some(a =>
    a.status === 'active' && (a.availableModels || []).includes(modelKey)
  );
  if (!anyEligible) {
    return {
      status: 403,
      body: {
        error: {
          message: `模型 ${displayModel} 在当前账号池中不可用（未订阅或已被封禁）`,
          type: 'model_not_entitled',
        },
      },
    };
  }

  const chatId = genId();
  const created = Math.floor(Date.now() / 1000);
  const ckey = emulateTools ? null : cacheKey(body, callerKey);

  if (stream) {
    return streamResponse(chatId, created, displayModel, modelKey, messages, cascadeMessages, modelEnum, modelUid, useCascade, ckey, emulateTools, toolPreamble, languageHint, source, creditMultiplier, callerKey, sessionKey, body);
  }

  // ── Local response cache (exact body match) ─────────────
  const cached = cacheGet(ckey);
  if (cached) {
    log.info(`Chat: cache HIT model=${displayModel} flow=non-stream`);
    recordRequest({
      model: displayModel, success: true, durationMs: 0, accountId: null,
      source, credit: 0, tokens: null,
    });
    const message = { role: 'assistant', content: cached.text || null };
    if (cached.thinking) message.reasoning_content = cached.thinking;
    return {
      status: 200,
      body: {
        id: chatId, object: 'chat.completion', created, model: displayModel,
        choices: [{ index: 0, message, finish_reason: 'stop' }],
        usage: cachedUsage(messages, cached.text),
      },
    };
  }

  // ── Cascade conversation pool (experimental) ──
  // If the client is continuing a prior conversation and we still hold the
  // cascade_id from last turn, pin this request to that exact (account, LS)
  // pair so the Windsurf backend serves from its hot per-cascade context
  // instead of replaying the whole history.
  //
  // Reuse Cascade state whenever possible. The fingerprint canonicalises tool
  // turns so tool-result conversations can keep the cascade that requested the
  // tool instead of replaying the whole transcript every turn.
  const reuseEnabled = useCascade && isExperimentalEnabled('cascadeConversationReuse');
  const fpBefore = reuseEnabled ? fingerprintBefore(messages, modelKey, callerKey) : null;
  const lastUserHash = reuseEnabled ? latestUserHash(messages, modelKey, callerKey) : '';
  const shapeHash = reuseEnabled ? requestShapeHash(body, modelKey, callerKey) : '';
  const allowSessionFallback = reuseEnabled && isExperimentalEnabled('cascadeSessionFallbackReuse');
  let reuseEntry = reuseEnabled ? poolCheckout(fpBefore, callerKey, sessionKey, { lastUserHash, requestShapeHash: shapeHash, allowSessionFallback }) : null;
  if (reuseEntry) log.info(`Chat: cascade reuse HIT reason=${reuseEntry.reuseReason || 'unknown'} cascadeId=${reuseEntry.cascadeId.slice(0, 8)}… model=${displayModel}`);

  // Non-stream: retry with a different account on model-not-available errors
  const tried = [];
  let lastErr = null;
  // Dynamic: try every active account (capped at 10) so a large pool with
  // many rate-limited accounts can still fall through to a healthy one.
  const maxAttempts = Math.min(10, Math.max(3, getAccountList().filter(a => a.status === 'active').length));
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    let acct = null;
    if (reuseEntry && attempt === 0) {
      // First attempt pins to the account that owns the cached cascade.
      acct = acquireAccountByKey(reuseEntry.apiKey, modelKey);
      if (!acct) {
        log.info('Chat: cascade reuse skipped — owning account not available, falling back to fresh cascade');
        reuseEntry = null;
      }
    }
    if (!acct) {
      acct = await waitForAccount(tried, null, QUEUE_MAX_WAIT_MS, modelKey);
      if (!acct) break;
    }
    try {
      tried.push(acct.apiKey);

      // Pre-flight rate limit check (experimental): ask server.codeium.com if
      // this account still has message capacity before burning an LS round trip.
      if (isExperimentalEnabled('preflightRateLimit')) {
        try {
          const px = getEffectiveProxy(acct.id) || null;
          const rl = await checkMessageRateLimit(acct.apiKey, px);
          if (!rl.hasCapacity) {
            log.warn(`Preflight: ${acct.email} has no capacity (remaining=${rl.messagesRemaining}), skipping`);
            markRateLimited(acct.id, modelKey);
            continue;
          }
        } catch (e) {
          log.debug(`Preflight check failed for ${acct.email}: ${e.message}`);
          // Fail open — proceed with the request
        }
      }

      await ensureLs(acct.proxy);
      const ls = getLsFor(acct.proxy);
      if (!ls) { lastErr = { status: 503, body: { error: { message: 'No LS instance available', type: 'ls_unavailable' } } }; break; }
      // Cascade pins cascade_id to a specific LS port too; if the LS it was
      // born on has been replaced, the cascade_id is dead.
      if (reuseEntry && reuseEntry.lsPort !== ls.port) {
        log.info('Chat: cascade reuse skipped — LS port changed');
        reuseEntry = null;
      }
      log.info(`Chat: model=${displayModel} flow=${useCascade ? 'cascade' : 'legacy'} attempt=${attempt + 1} account=${acct.email} ls=${ls.port}${reuseEntry ? ' reuse=1' : ''}${emulateTools ? ' tools=emu' : ''}`);
      const client = new WindsurfClient(acct.apiKey, ls.port, ls.csrfToken);
      const result = await nonStreamResponse(
        client, chatId, created, displayModel, modelKey, messages, cascadeMessages, modelEnum, modelUid,
        useCascade, acct.apiKey, ckey,
        reuseEnabled ? { reuseEntry, lsPort: ls.port, apiKey: acct.apiKey, callerKey, sessionKey, requestShapeHash: shapeHash } : null,
        emulateTools, toolPreamble, languageHint,
        source, creditMultiplier,
      );
      if (result.status === 200) return result;
      reuseEntry = null; // don't try to reuse on the retry
      lastErr = result;
      const errType = result.body?.error?.type;
      // Only known account-specific or transient failures should move to the
      // next account. Terminal upstream rejections (policy, bad input, context)
      // are returned directly instead of trying to route around them.
      if (result.body?.error?.retryable === true) {
        log.warn(`Account ${acct.email} failed (${errType}) on ${displayModel}, trying next account`);
        continue;
      }
      break; // other errors (502, transport) — don't retry
    } finally {
      acct.release?.();
    }
  }
  // If all accounts exhausted, check if it's because they're all rate-limited
  if (!lastErr || lastErr.status === 429) {
    const rl = isAllRateLimited(modelKey);
    if (rl.allLimited) {
      return { status: 429, body: { error: { message: `${displayModel} 所有账号均已达速率限制，请 ${Math.ceil(rl.retryAfterMs / 1000)} 秒后重试`, type: 'rate_limit_exceeded', retry_after_ms: rl.retryAfterMs } } };
    }
  }
  return lastErr || { status: 503, body: { error: { message: 'No active accounts available', type: 'pool_exhausted' } } };
}

async function nonStreamResponse(client, id, created, model, modelKey, messages, cascadeMessages, modelEnum, modelUid, useCascade, apiKey, ckey, poolCtx, emulateTools, toolPreamble, languageHint, source = 'POST /v1/chat/completions', creditMultiplier = 0) {
  const startTime = Date.now();
  try {
    let allText = '';
    let allThinking = '';
    let cascadeMeta = null;
    let toolCalls = [];
    // Server-reported token usage from CortexStepMetadata.model_usage, summed
    // across all trajectory steps. Preferred over the chars/4 estimate when
    // present so downstream billing (new-api, etc.) sees real Cascade numbers.
    let serverUsage = null;

    if (useCascade) {
      const chunks = await client.cascadeChat(cascadeMessages, modelEnum, modelUid, { reuseEntry: poolCtx?.reuseEntry || null, toolPreamble, languageHint });
      for (const c of chunks) {
        if (c.text) allText += c.text;
        if (c.thinking) allThinking += c.thinking;
      }
      cascadeMeta = { cascadeId: chunks.cascadeId, sessionId: chunks.sessionId, endReason: chunks.endReason };
      serverUsage = chunks.usage || null;
      // Always strip <tool_call>/<tool_result> blocks from Cascade text.
      // - emulateTools=true: parsed tool_calls become OpenAI-format tool_calls.
      // - emulateTools=false: blocks are silently discarded (defense-in-depth
      //   against Cascade's system prompt inducing tool markup even after we
      //   override tool_calling_section).
      {
        const parsed = parseToolCallsFromText(allText);
        allText = parsed.text;
        if (emulateTools) toolCalls = parsed.toolCalls;
      }
      // Built-in Cascade tool calls (chunks.toolCalls — edit_file, view_file,
      // list_directory, run_command, etc.) are intentionally DROPPED. Their
      // argumentsJson and result fields reference server-internal paths like
      // /tmp/windsurf-workspace/config.yaml and must never be exposed to an
      // API caller. Emulated tool calls (above) are safe because they
      // reference the caller's own tool schema.
    } else {
      const chunks = await client.rawGetChatMessage(messages, modelEnum, modelUid);
      for (const c of chunks) {
        if (c.text) allText += c.text;
      }
    }

    // Scrub server-internal filesystem paths from everything we're about to
    // return. See src/sanitize.js for the patterns and rationale.
    allText = sanitizeText(allText);
    allThinking = sanitizeText(allThinking);
    if (toolCalls.length) {
      toolCalls = toolCalls.map(tc => ({
        ...tc,
        argumentsJson: sanitizeText(tc.argumentsJson || ''),
      }));
    }

    const responseToolCalls = toolCalls.map((tc, i) => ({
      id: tc.id || `call_${i}_${Date.now().toString(36)}`,
      type: 'function',
      function: {
        name: tc.name || 'unknown',
        arguments: tc.argumentsJson || tc.arguments || '{}',
      },
    }));

    // Check the cascade back into the pool under the *post-turn* fingerprint
    // so the next request in the same conversation can resume it.
    if (poolCtx && cascadeMeta?.cascadeId && cascadeMeta.endReason && cascadeMeta.endReason !== 'idle_done') {
      log.info(`Chat: cascade pool checkin skipped reason=${cascadeMeta.endReason} cascadeId=${cascadeMeta.cascadeId.slice(0, 8)}…`);
    }
    if (poolCtx && cascadeMeta?.cascadeId && cascadeMeta.endReason === 'idle_done' && (allText || responseToolCalls.length)) {
      const poolMessages = responseToolCalls.length
        ? [...messages, { role: 'assistant', content: null, tool_calls: responseToolCalls }]
        : messages;
      const fpAfter = fingerprintAfter(poolMessages, modelKey, poolCtx.callerKey || '');
      poolCheckin(fpAfter, {
        cascadeId: cascadeMeta.cascadeId,
        sessionId: cascadeMeta.sessionId,
        lsPort: poolCtx.lsPort,
        apiKey: poolCtx.apiKey,
        lastUserHash: latestUserHash(messages, modelKey, poolCtx.callerKey || ''),
        requestShapeHash: poolCtx.requestShapeHash || '',
        createdAt: poolCtx.reuseEntry?.createdAt,
      }, poolCtx.callerKey || '', poolCtx.sessionKey || '');
    }

    reportSuccess(apiKey);
    updateCapability(apiKey, modelKey, true, 'success');
    recordRequest({
      model,
      success: true,
      durationMs: Date.now() - startTime,
      accountId: apiKey,
      source,
      credit: creditMultiplier,
      tokens: serverUsage ? {
        input:     (serverUsage.inputTokens || 0) + (serverUsage.cacheWriteTokens || 0),
        output:    serverUsage.outputTokens || 0,
        reasoning: 0,
        cached:    serverUsage.cacheReadTokens || 0,
        total:     (serverUsage.inputTokens || 0) + (serverUsage.outputTokens || 0) +
                   (serverUsage.cacheReadTokens || 0) + (serverUsage.cacheWriteTokens || 0),
      } : null,
    });

    // Store in cache for next identical request. Skip caching tool_call
    // responses — they're inherently contextual and the cache doesn't
    // preserve the tool_calls array, so a cache hit would return a
    // content-only response with finish_reason:stop, breaking tool flow.
    if (ckey && !toolCalls.length) cacheSet(ckey, { text: allText, thinking: allThinking });

    const message = { role: 'assistant', content: allText || null };
    if (allThinking) message.reasoning_content = allThinking;
    if (responseToolCalls.length) {
      message.tool_calls = responseToolCalls;
      // OpenAI convention: content is null when finish_reason is tool_calls.
      // In text emulation the model often emits an inline answer alongside the
      // <tool_call> block (e.g., hallucinated weather data). Set content to
      // null so clients that check `content !== null` behave correctly and the
      // caller waits for the real tool result rather than showing hallucinated
      // data.
      message.content = null;
    }

    // Prefer server-reported usage; fall back to chars/4 estimate only when
    // the trajectory didn't include a ModelUsageStats field.
    const usage = buildUsageBody(serverUsage, messages, allText, allThinking);
    const finishReason = responseToolCalls.length ? 'tool_calls' : 'stop';
    return {
      status: 200,
      body: {
        id, object: 'chat.completion', created, model,
        choices: [{ index: 0, message, finish_reason: finishReason }],
        usage,
      },
    };
  } catch (err) {
    const cls = applyErrorSideEffects(err, apiKey, modelKey);
    recordRequest({
      model, success: false, durationMs: Date.now() - startTime,
      accountId: apiKey, source, credit: 0, tokens: null,
    });
    log.error('Chat error:', err.message);
    if (cls.tag === 'rate_limit') {
      const rl = isAllRateLimited(modelKey);
      return {
        status: 429,
        body: { error: { message: `${model} 已达速率限制，请稍后重试`, type: 'rate_limit_exceeded', retry_after_ms: rl.retryAfterMs || 60000, retryable: true } },
      };
    }
    return {
      status: cls.status,
      body: { error: { message: sanitizeText(err.message), type: cls.type, retryable: cls.action === 'switch_account' } },
    };
  }
}

function streamResponse(id, created, model, modelKey, messages, cascadeMessages, modelEnum, modelUid, useCascade, ckey, emulateTools, toolPreamble, languageHint, source = 'POST /v1/chat/completions', creditMultiplier = 0, callerKey = '', sessionKey = '', requestBody = {}) {
  return {
    status: 200,
    stream: true,
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'X-Accel-Buffering': 'no',
    },
    async handler(res) {
      const abortController = new AbortController();
      res.on('close', () => {
        if (!res.writableEnded) {
          log.info('Client disconnected mid-stream, aborting upstream');
          abortController.abort();
        }
      });
      // Immediate kick so the TCP layer flushes headers + first bytes to the
      // client right away — otherwise SSE-over-keepalive clients (esp. CC) can
      // sit in "connecting" state for the full cold-start duration.
      if (!res.writableEnded) res.write(': ping\n\n');
      const send = (data) => {
        if (!res.writableEnded) res.write(`data: ${JSON.stringify(data)}\n\n`);
      };

      // SSE heartbeat: keep the TCP/HTTP connection alive through any silent
      // period (LS warmup, Cascade "thinking", queue wait). `:` prefix is a
      // comment line per the SSE spec — clients ignore it, intermediaries see
      // bytes flowing, idle timers get reset.
      const heartbeat = setInterval(() => {
        if (!res.writableEnded) res.write(': ping\n\n');
      }, HEARTBEAT_MS);
      const stopHeartbeat = () => clearInterval(heartbeat);
      res.on('close', stopHeartbeat);

      // ── Cache hit: replay stored response as a fake stream ──
      const cached = cacheGet(ckey);
      if (cached) {
        log.info(`Chat: cache HIT model=${model} flow=stream`);
        recordRequest({
          model, success: true, durationMs: 0, accountId: null,
          source, credit: 0, tokens: null,
        });
        try {
          send({ id, object: 'chat.completion.chunk', created, model,
            choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] });
          if (cached.thinking) {
            send({ id, object: 'chat.completion.chunk', created, model,
              choices: [{ index: 0, delta: { reasoning_content: cached.thinking }, finish_reason: null }] });
          }
          if (cached.text) {
            send({ id, object: 'chat.completion.chunk', created, model,
              choices: [{ index: 0, delta: { content: cached.text }, finish_reason: null }] });
          }
          send({ id, object: 'chat.completion.chunk', created, model,
            choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
            usage: cachedUsage(messages, cached.text) });
          if (!res.writableEnded) { res.write('data: [DONE]\n\n'); res.end(); }
        } finally {
          stopHeartbeat();
        }
        return;
      }

      const startTime = Date.now();
      const tried = [];
      let hadSuccess = false;
      let rolePrinted = false;
      let currentApiKey = null;
      let lastErr = null;
      const maxAttempts = Math.min(10, Math.max(3, getAccountList().filter(a => a.status === 'active').length));

      // Accumulate chunks so we can cache a successful response at the end.
      let accText = '';
      let accThinking = '';

      // Cascade conversation pool (experimental, stream path). Tool-call
      // turns are canonicalised by the fingerprint so tool-result follow-ups
      // can resume the same cascade instead of replaying the whole transcript.
      const reuseEnabled = useCascade && isExperimentalEnabled('cascadeConversationReuse');
      const fpBefore = reuseEnabled ? fingerprintBefore(messages, modelKey, callerKey) : null;
      const lastUserHash = reuseEnabled ? latestUserHash(messages, modelKey, callerKey) : '';
      const shapeHash = reuseEnabled ? requestShapeHash(requestBody, modelKey, callerKey) : '';
      const allowSessionFallback = reuseEnabled && isExperimentalEnabled('cascadeSessionFallbackReuse');
      let reuseEntry = reuseEnabled ? poolCheckout(fpBefore, callerKey, sessionKey, { lastUserHash, requestShapeHash: shapeHash, allowSessionFallback }) : null;
      if (reuseEntry) log.info(`Chat: cascade reuse HIT reason=${reuseEntry.reuseReason || 'unknown'} cascadeId=${reuseEntry.cascadeId.slice(0, 8)}… stream model=${model}`);

      // Always strip <tool_call>/<tool_result> blocks in Cascade mode.
      // In emulation mode, parsed calls are emitted as OpenAI tool_calls.
      // In non-emulation mode, blocks are silently stripped (defense-in-depth
      // against Cascade's system prompt inducing tool markup).
      const toolParser = useCascade ? new ToolCallStreamParser() : null;
      const collectedToolCalls = [];

      // Streaming path sanitizers. Every text/thinking delta flows through a
      // PathSanitizeStream before leaving the server so /tmp/windsurf-workspace,
      // /opt/windsurf and /root/WindsurfPoolAPI literals can never slip out even
      // if a path straddles a chunk boundary. See src/sanitize.js.
      const pathStreamText = new PathSanitizeStream();
      const pathStreamThinking = new PathSanitizeStream();

      const emitContent = (clean) => {
        if (!clean) return;
        accText += clean;
        send({ id, object: 'chat.completion.chunk', created, model,
          choices: [{ index: 0, delta: { content: clean }, finish_reason: null }] });
      };
      const emitThinking = (clean) => {
        if (!clean) return;
        accThinking += clean;
        send({ id, object: 'chat.completion.chunk', created, model,
          choices: [{ index: 0, delta: { reasoning_content: clean }, finish_reason: null }] });
      };

      const emitToolCallDelta = (tc, idx) => {
        send({ id, object: 'chat.completion.chunk', created, model,
          choices: [{ index: 0, delta: {
            tool_calls: [{
              index: idx,
              id: tc.id,
              type: 'function',
              function: { name: tc.name, arguments: sanitizeText(tc.argumentsJson || '{}') },
            }],
          }, finish_reason: null }] });
      };

      const onChunk = (chunk) => {
        if (!rolePrinted) {
          rolePrinted = true;
          send({ id, object: 'chat.completion.chunk', created, model,
            choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] });
        }
        hadSuccess = true;

        if (chunk.text) {
          // Pipeline for text deltas:
          //   raw chunk  →  ToolCallStreamParser (strip <tool_call> blocks)
          //              →  PathSanitizeStream   (scrub server paths)
          //              →  client
          let safeText = chunk.text;
          if (toolParser) {
            const { text: safe, toolCalls: done } = toolParser.feed(chunk.text);
            safeText = safe;
            // Only emit tool_call deltas when emulating — otherwise the
            // parsed calls came from Cascade's built-in tools and are
            // silently discarded.
            if (emulateTools) {
              for (const tc of done) {
                const idx = collectedToolCalls.length;
                collectedToolCalls.push(tc);
                emitToolCallDelta(tc, idx);
              }
            }
          }
          if (safeText) emitContent(pathStreamText.feed(safeText));
        }
        if (chunk.thinking) {
          emitThinking(pathStreamThinking.feed(chunk.thinking));
        }
      };

      try {
        for (let attempt = 0; attempt < maxAttempts; attempt++) {
          if (abortController.signal.aborted) return;
          let acct = null;
          if (reuseEntry && attempt === 0) {
            acct = acquireAccountByKey(reuseEntry.apiKey, modelKey);
            if (!acct) {
              log.info('Chat: cascade reuse skipped — owning account not available');
              reuseEntry = null;
            }
          }
          if (!acct) {
            acct = await waitForAccount(tried, abortController.signal, QUEUE_MAX_WAIT_MS, modelKey);
            if (!acct) break;
          }
          try {
            tried.push(acct.apiKey);
            currentApiKey = acct.apiKey;

            // Pre-flight rate limit check (experimental)
            if (isExperimentalEnabled('preflightRateLimit')) {
              try {
                const px = getEffectiveProxy(acct.id) || null;
                const rl = await checkMessageRateLimit(acct.apiKey, px);
                if (!rl.hasCapacity) {
                  log.warn(`Preflight: ${acct.email} has no capacity (remaining=${rl.messagesRemaining}), skipping`);
                  markRateLimited(acct.id, modelKey);
                  continue;
                }
              } catch (e) {
                log.debug(`Preflight check failed for ${acct.email}: ${e.message}`);
              }
            }

            try { await ensureLs(acct.proxy); } catch (e) { lastErr = e; break; }
            const ls = getLsFor(acct.proxy);
            if (!ls) { lastErr = new Error('No LS instance available'); break; }
            if (reuseEntry && reuseEntry.lsPort !== ls.port) {
              log.info('Chat: cascade reuse skipped — LS port changed');
              reuseEntry = null;
            }
            log.info(`Chat: model=${model} flow=${useCascade ? 'cascade' : 'legacy'} stream=true attempt=${attempt + 1} account=${acct.email} ls=${ls.port}${reuseEntry ? ' reuse=1' : ''}`);
            const client = new WindsurfClient(acct.apiKey, ls.port, ls.csrfToken);
            let cascadeResult = null;
            if (useCascade) {
              cascadeResult = await client.cascadeChat(cascadeMessages, modelEnum, modelUid, {
                onChunk, signal: abortController.signal, reuseEntry, toolPreamble, languageHint,
              });
            } else {
              await client.rawGetChatMessage(messages, modelEnum, modelUid, { onChunk });
            }
            // Flush order matters:
            //   1. ToolCallStreamParser tail → may produce more text deltas
            //      (e.g., a dangling <tool_call> that never closed falls
            //      through as literal text)
            //   2. PathSanitizeStream tail (text) → scrubs anything the tool
            //      parser held back AND anything we were holding ourselves
            //   3. PathSanitizeStream tail (thinking)
            if (toolParser) {
              const tail = toolParser.flush();
              if (tail.text) emitContent(pathStreamText.feed(tail.text));
              if (emulateTools) {
                for (const tc of tail.toolCalls) {
                  const idx = collectedToolCalls.length;
                  collectedToolCalls.push(tc);
                  emitToolCallDelta(tc, idx);
                }
              }
            }
            emitContent(pathStreamText.flush());
            emitThinking(pathStreamThinking.flush());
            // Pool check-in on success (cascade only)
            if (reuseEnabled && cascadeResult?.cascadeId && cascadeResult.endReason && cascadeResult.endReason !== 'idle_done') {
              log.info(`Chat: cascade pool checkin skipped reason=${cascadeResult.endReason} cascadeId=${cascadeResult.cascadeId.slice(0, 8)}…`);
            }
            if (reuseEnabled && cascadeResult?.cascadeId && cascadeResult.endReason === 'idle_done' && (accText || collectedToolCalls.length)) {
              const responseToolCalls = collectedToolCalls.map((tc, i) => ({
                id: tc.id || `call_${i}_${Date.now().toString(36)}`,
                type: 'function',
                function: {
                  name: tc.name || 'unknown',
                  arguments: sanitizeText(tc.argumentsJson || '{}'),
                },
              }));
              const poolMessages = responseToolCalls.length
                ? [...messages, { role: 'assistant', content: null, tool_calls: responseToolCalls }]
                : messages;
              const fpAfter = fingerprintAfter(poolMessages, modelKey, callerKey);
              poolCheckin(fpAfter, {
                cascadeId: cascadeResult.cascadeId,
                sessionId: cascadeResult.sessionId,
                lsPort: ls.port,
                apiKey: currentApiKey,
                lastUserHash,
                requestShapeHash: shapeHash,
                createdAt: reuseEntry?.createdAt,
              }, callerKey, sessionKey);
            }
            // success
            if (hadSuccess) reportSuccess(currentApiKey);
            updateCapability(currentApiKey, modelKey, true, 'success');
            recordRequest({
              model,
              success: true,
              durationMs: Date.now() - startTime,
              accountId: currentApiKey,
              source,
              credit: creditMultiplier,
              tokens: cascadeResult?.usage ? {
                input:     (cascadeResult.usage.inputTokens || 0) + (cascadeResult.usage.cacheWriteTokens || 0),
                output:    cascadeResult.usage.outputTokens || 0,
                reasoning: 0,
                cached:    cascadeResult.usage.cacheReadTokens || 0,
                total:     (cascadeResult.usage.inputTokens || 0) + (cascadeResult.usage.outputTokens || 0) +
                           (cascadeResult.usage.cacheReadTokens || 0) + (cascadeResult.usage.cacheWriteTokens || 0),
              } : null,
            });
            if (!rolePrinted) {
              send({ id, object: 'chat.completion.chunk', created, model,
                choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] });
            }
            const finalReason = collectedToolCalls.length ? 'tool_calls' : 'stop';
            send({ id, object: 'chat.completion.chunk', created, model,
              choices: [{ index: 0, delta: {}, finish_reason: finalReason }] });
            // OpenAI-compat: terminal usage chunk (stream_options.include_usage
            // convention — empty choices[] + usage). Prefer Cascade's own
            // CortexStepMetadata.model_usage numbers when present, fall back
            // to the local chars/4 estimator. See buildUsageBody().
            {
              const usage = buildUsageBody(cascadeResult?.usage || null, messages, accText, accThinking);
              send({ id, object: 'chat.completion.chunk', created, model,
                choices: [], usage });
            }
            if (!res.writableEnded) { res.write('data: [DONE]\n\n'); res.end(); }
            if (ckey && !collectedToolCalls.length && (accText || accThinking)) {
              cacheSet(ckey, { text: accText, thinking: accThinking });
            }
            return;
          } catch (err) {
            lastErr = err;
            reuseEntry = null; // don't try to reuse on retry
            const cls = applyErrorSideEffects(err, currentApiKey, modelKey);
            // Retry only if nothing has been streamed yet AND the classifier says
            // the failure is account-specific or transient. Terminal upstream
            // rejections must be returned directly.
            if (!hadSuccess && cls.action === 'switch_account') {
              log.warn(`Account ${acct.email} failed (${cls.tag}) on ${model}, trying next`);
              continue;
            }
            break;
          } finally {
            acct.release?.();
          }
        }

        // All attempts failed
        log.error('Stream error after retries:', lastErr?.message);
        recordRequest({
          model, success: false, durationMs: Date.now() - startTime,
          accountId: currentApiKey, source, credit: 0, tokens: null,
        });
        try {
          if (!rolePrinted) {
            send({ id, object: 'chat.completion.chunk', created, model,
              choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] });
          }
          // Check if failure is due to all accounts being rate-limited
          const rl = isAllRateLimited(modelKey);
          const errMsg = rl.allLimited
            ? `${model} 所有账号均已达速率限制，请 ${Math.ceil(rl.retryAfterMs / 1000)} 秒后重试`
            : sanitizeText(lastErr?.message || 'no accounts');
          send({ id, object: 'chat.completion.chunk', created, model,
            choices: [{ index: 0, delta: { content: `\n[Error: ${errMsg}]` }, finish_reason: 'stop' }] });
          res.write('data: [DONE]\n\n');
        } catch {}
        if (!res.writableEnded) res.end();
      } finally {
        stopHeartbeat();
      }
    },
  };
}
