# GBRAIN Research Report for Elliott's Voice Chat Project

## WHAT GBRAIN IS

GBrain is a persistent knowledge brain system for AI agents, built by Garry Tan (Y Combinator CEO). It gives AI agents a searchable, compounding memory that grows with every interaction. Think of it as an Obsidian vault that the AI can actually query, write to, and maintain autonomously.

The production deployment: 17,888 pages, 4,383 people, 723 companies, 21 autonomous cron jobs.

**Core principle:** Every conversation makes the brain smarter. Every brain lookup makes responses better. The loop compounds daily.

---

## ARCHITECTURE OVERVIEW

### Three-Layer Knowledge Model
1. **Raw Sources** — Immutable input documents (meetings, emails, tweets). LLM reads but never modifies.
2. **The Wiki** — LLM-generated/maintained markdown pages: summaries, entity pages, concept pages. Each page has:
   - **Compiled truth** (above `---`) — Current best understanding, gets rewritten as knowledge evolves
   - **Timeline** (below `---`) — Append-only evidence trail, never edited
3. **The Schema** — Config docs (SOUL.md, CLAUDE.md) defining structure, conventions, workflows

### Brain-Agent Loop
```
Signal arrives (message, meeting, email, tweet, link)
  → DETECT entities (people, companies, concepts) — async, never blocks
  → READ: check brain FIRST (before responding)
  → RESPOND with brain context
  → WRITE: update brain pages with new info + citations
  → SYNC: gbrain indexes changes for next query
(next signal — agent is now smarter)
```

### Contract-First Design
- `src/core/operations.ts` — Single source of truth. ~30 operations defined here.
- CLI and MCP server are BOTH generated from this file. No duplication.
- `src/core/engine.ts` — Pluggable `BrainEngine` interface (17 methods, 92 lines)
- `src/core/engine-factory.ts` — Dynamic engine selection: `'pglite'` or `'postgres'`

### Engine Architecture
```
CLI / MCP Server (thin wrappers)
     |
BrainEngine interface (pluggable)
     |
  +--------+--------+
  |                  |
PGLiteEngine       PostgresEngine
(default)          (Supabase)
  |                  |
~/.gbrain/          Supabase Pro ($25/mo)
brain.pglite        Postgres + pgvector
embedded PG 17.5
```

**PGLite** (default): Embedded Postgres 17.5 via WASM (@electric-sql/pglite). Zero config. Single process. Good for <1,000 files.

**PostgresEngine**: Supabase or self-hosted. Connection pooling. Production-proven at 10K+ pages. Multi-device via remote MCP.

**Migration**: `gbrain migrate --to supabase|pglite` — bidirectional, lossless.

### The 30 MCP Operations
Defined in `src/core/operations.ts`:
- **Page CRUD**: get_page, put_page, delete_page, list_pages
- **Search**: search (keyword), query (hybrid vector+keyword+RRF)
- **Tags**: add_tag, remove_tag, get_tags
- **Links**: add_link, remove_link, get_links, get_backlinks, traverse_graph
- **Timeline**: add_timeline_entry, get_timeline
- **Admin**: get_stats, get_health, get_versions, revert_version
- **Sync**: sync_brain
- **Raw Data**: put_raw_data, get_raw_data
- **Resolution & Chunks**: resolve_slugs, get_chunks
- **Ingest Log**: log_ingest, get_ingest_log
- **Files**: file_list, file_upload, file_url

### Search Pipeline
```
Query → Intent classifier (entity? temporal? event? general?)
     → Multi-query expansion (Claude Haiku)
     → Vector search (pgvector HNSW cosine) + Keyword search (tsvector)
     → RRF fusion: score = sum(1/(60 + rank))
     → Cosine re-scoring + compiled truth boost
     → 4-layer dedup + compiled truth guarantee
     → Results
```

Embedding: OpenAI text-embedding-3-large, 1536 dimensions, batch processing with retry.

### Skill System
26 markdown-based skills in `skills/` directory. Fat markdown documents that encode entire workflows. Agent reads the skill and executes it. Skills call deterministic TypeScript for reliability.

Key skills for voice: signal-detector (fires on every message, async), brain-ops (brain-first lookup loop), meeting-ingestion (transcripts → brain pages).

---

## VOICE CHAT SYSTEM

GBrain ships a **Voice-to-Brain** recipe at `recipes/twilio-voice-brain.md`. This is the most detailed document in the entire repo (798 lines, 35KB).

### Two Voice Architectures

**Option A: OpenAI Realtime (turnkey)**
```
Caller → Twilio (WebSocket, g711_ulaw) → Voice Server (Node.js)
  ↔ OpenAI Realtime API (STT + LLM + TTS in one pipeline)
  → Function calls → GBrain MCP (search, page reads, page writes)
  → Post-call → Brain page + messaging notification
```

**Option B: DIY STT+LLM+TTS (full control, production-grade)**
```
Caller (phone or WebRTC browser) → Twilio WS OR WebRTC → Voice Server
  → Deepgram STT → Claude API (streaming SSE) → Cartesia/OpenAI TTS
  → Function calls → GBrain MCP
  → Post-call → Brain page + audio + transcript
```

### WebRTC Support (BROWSER-BASED — KEY FOR ELLIOTT)
The voice recipe INCLUDES WebRTC browser calling. No phone needed.

- `POST /session` — Accepts SDP offer, forwards to OpenAI Realtime `/v1/realtime/calls`, returns SDP answer
- `GET /call` — Serves a web client HTML page with:
  - WebRTC connection to OpenAI Realtime API
  - RNNoise WASM noise suppression (AudioWorklet)
  - Push-to-talk AND auto-VAD mode switching
  - Pipeline: Microphone → RNNoise denoise → MediaStream → WebRTC → OpenAI
- `POST /tool` — Receives tool calls from WebRTC data channel, executes them, returns results

**This is exactly what Elliott wants — webpage-based voice UI, no Twilio required.**

### WebRTC Gotchas (from the recipe)
- `voice` config goes under `audio.output.voice`, NOT top-level
- Do NOT send `turn_detection` in session config (not accepted by `/v1/realtime/calls`)
- Do NOT send `session.update` on connect (server already configured it)
- All `session.update` calls must include `type: 'realtime'` to avoid session.type errors
- `input_audio_transcription` is NOT supported over WebRTC data channel — use Whisper post-call on recorded audio instead
- Trigger greeting via data channel after WebRTC connects

### Production Fixes (CRITICAL — from real deployment)
1. **Unicode Crash Fix** — Non-ASCII chars (em dashes, smart quotes, arrows) crash Twilio WebSocket. Must sanitize to ASCII.
2. **PII Scrub** — Brain context loaded into voice prompt may contain phone numbers/emails. Voice agent reads them aloud. Must regex-strip.
3. **Identity-First Prompt** — Put identity FIRST in system prompt, before context/rules. Model reverts to base persona otherwise. "You ARE [Name]. You are NOT Claude."
4. **Auto-Upload Call Audio** — Immediately upload all call audio on call end in case post-call processing fails.
5. **Smart VAD (Silero)** — Default to Silero VAD, not push-to-talk. PTT available as fallback.
6. **Conversation Timing** — The #1 fix. "Caller talking or thinking: SHUT UP. Even 3-5 second pauses. Caller done (complete thought + 2-3 sec silence): NOW respond. Never let silence go past 5 seconds after a COMPLETE thought."

### Voice Client UI
The repo includes `docs/images/voice-client.png` showing a production WebRTC voice client interface.

---

## TECH STACK

| Component | Tech | Notes |
|-----------|------|-------|
| Runtime | **Bun** (not Node.js) | Package manager + runtime |
| Language | **TypeScript** | ESM modules throughout |
| Database (default) | **PGLite** (@electric-sql/pglite 0.4.4) | Embedded Postgres 17.5 via WASM |
| Database (production) | **Postgres + pgvector** | Supabase Pro or self-hosted |
| Embeddings | **OpenAI text-embedding-3-large** | 1536 dimensions |
| LLM (expansion) | **Claude Haiku** | Multi-query expansion for search |
| Transcription | **Groq Whisper** (default) / **OpenAI Whisper** (fallback) | ffmpeg segmentation for >25MB |
| MCP SDK | **@modelcontextprotocol/sdk 1.0.0** | stdio transport |
| HTTP | **Express** | Voice server |
| WebSocket | **ws** | Voice media stream |
| Storage | S3 / Supabase Storage / Local | Pluggable file storage |
| Auth | Bearer tokens (SHA-256 hashed) | For remote MCP access |

---

## DEPENDENCIES & SETUP

### Prerequisites
- **Bun** runtime (NOT Node.js — GBrain uses Bun-specific features)
- **OpenAI API key** (for embeddings + optional voice Realtime API)
- **Anthropic API key** (optional, for query expansion)
- For PGLite: nothing else — zero config
- For Postgres: Supabase account or self-hosted Postgres with pgvector

### Quick Setup (Standalone)
```bash
git clone https://github.com/garrytan/gbrain.git && cd gbrain
bun install && bun link
gbrain init              # PGLite brain, ready in 2 seconds
gbrain import ~/notes/   # index your markdown
gbrain query "what themes show up across my notes?"
```

### For Voice Chat
```bash
mkdir -p voice-agent && cd voice-agent
npm init -y
npm install ws express
# Build server.mjs following recipe architecture
# WebRTC route: GET /call serves browser client, POST /session handles SDP
# Tool route: POST /tool executes GBrain MCP calls
```

### Environment Variables
- `OPENAI_API_KEY` — Required for embeddings + voice
- `GBRAIN_DATABASE_URL` or `DATABASE_URL` — For Postgres engine
- `GROQ_API_KEY` — For transcription (optional, falls back to OpenAI)
- `ANTHROPIC_API_KEY` — For query expansion (optional)

---

## CUSTOMIZATION NEEDED FOR ELLIOTT'S USE CASE

### 1. Alina as Primary Agent (CRITICAL)
GBrain's identity system uses SOUL.md. Need to replace default identity:
- Fill in `templates/SOUL.md.template` with Alina's personality, voice, mission
- The **identity-first prompt** pattern from the voice recipe: "You ARE Alina. You are NOT Claude. You are NOT a general AI assistant."
- Alina's personality traits, communication style, operating principles
- The SOUL.md template has sections: Identity, Vibe, Mission, Operating Principles, Communication Style

### 2. Webpage-Based Voice UI (Already Supported!)
The WebRTC route in the voice recipe is exactly this. No Twilio needed.

**What to build:**
- Custom web page at `GET /call` (replace the basic one in the recipe)
- WebRTC connection to voice backend
- Dropdown for agent switching (see below)
- Agent identity panel showing current agent + personality
- Push-to-talk + auto-VAD toggle
- RNNoise noise suppression (already in recipe)

**Elliott's concern about latency:** The voice recipe documents several latency patterns:
- OpenAI Realtime API: ~200-500ms round-trip (STT+LLM+TTS in one pipeline)
- DIY pipeline: Can be optimized but needs tuning
- Key latency reducers from the recipe:
  - Auth-before-speech (call auth tool BEFORE greeting)
  - Radical prompt compression (cut from 13K to 4.7K tokens = 65% cut)
  - Report-aware query routing (keyword map before full vector search)
  - Thinking sounds during tool execution (pre-generated g711_ulaw audio chunks)
  - Sentence-boundary TTS dispatch (stream audio as soon as sentence is complete, don't wait for full response)

### 3. Custom Voice Model for Alina
The voice recipe's WebRTC endpoint supports voice configuration:

```javascript
sessionConfig = JSON.stringify({
  type: 'realtime',
  model: 'gpt-4o-realtime-preview',
  audio: { output: { voice: VOICE } },  // <-- voice goes here
  instructions: buildPrompt(null),
  tools: TOOL_SETS.unauthenticated,
})
```

**For OpenAI Realtime**: Set `VOICE` to the desired OpenAI voice name. Available voices: alloy, echo, fable, onyx, nova, shimmer.

**For custom voice model (local server arriving in ~2 weeks):**
- This requires the **DIY pipeline (Option B)** — swap the TTS stage to point at local server
- The DIY pipeline uses: Deepgram STT → Claude API → Cartesia/OpenAI TTS
- Replace the TTS step with Elliott's local voice server endpoint
- The pipeline architecture is modular — each stage is swappable
- This gives full control over voice quality and latency

### 4. NanoClaw Agent Swapping (Dropdown UI)
GBrain doesn't have this natively. Need to build:

**System prompt builder modification:**
- Current: `buildPrompt(callerPhone)` returns different prompts based on caller
- New: `buildPrompt(agentName)` returns different prompts based on selected agent
- Each agent gets a SOUL.md file: Alina, other NanoClaw agents
- The `session.update` API can swap system instructions, voice, and tools mid-conversation

**Web UI dropdown:**
- List of available agents from a config file or directory scan
- On agent change: send `session.update` with new `instructions` + `voice` config
- Include `type: 'realtime'` in all session updates (critical per recipe)

### 5. No Twilio (WebRTC Only)
The WebRTC approach is already documented. Just skip Twilio setup entirely:
- No phone number needed
- No ngrok needed (if server is locally accessible or deployed)
- Simpler deployment
- The `GET /call` web client is the user interface

---

## GOTCHAS

1. **Bun required, not Node.js** — GBrain uses Bun-specific features. `bun install && bun link` for setup.
2. **PGLite doesn't support concurrent access** — Single process only. File lock prevents crashes.
3. **OpenAI API key required for embeddings** — Hybrid search won't work without it. Keyword search still works.
4. **Embedded PG must be initialized** — `gbrain init` creates the schema. Without it, nothing works.
5. **Unicode crashes Twilio** — Must sanitize non-ASCII in prompts (em dashes, smart quotes, arrows).
6. **PII leaks through voice** — Brain context may contain phone numbers/emails that the agent reads aloud.
7. **Identity drift** — Model reverts to base persona without identity-first prompting.
8. **WebRTC `input_audio_transcription` not supported** — Must use Whisper post-call for transcripts.
9. **`session.update` must include `type: 'realtime'`** — Otherwise session.type errors.
10. **Voice inside `audio.output.voice`, not top-level** — Common WebRTC config mistake.
11. **Latency is real** — OpenAI Realtime is ~200-500ms. The recipe has extensive optimization patterns. Elliott's past experience with latency is justified.
12. **The voice server is NOT included in the GBrain repo** — The recipe describes the architecture but you build server.mjs yourself. It's a guide, not a package.

---

## RECOMMENDED SETUP PLAN

### Phase 1: GBrain Core (Day 1, ~1 hour)
1. Install Bun: `curl -fsSL https://bun.sh/install | bash`
2. Install GBrain: `cd gbrain && bun install && bun link`
3. Initialize: `gbrain init` (PGLite, zero config)
4. Set OPENAI_API_KEY env var
5. Import existing Obsidian notes: `gbrain import /home/elliott/obsidian/`
6. Test: `gbrain query "what do I know about..."` and verify results
7. Start MCP server: `gbrain serve` (for agent platforms)

### Phase 2: Voice Server with WebRTC (Days 2-3)
1. Create voice-agent directory per recipe
2. Build server.mjs with:
   - WebRTC endpoint: POST /session, GET /call, POST /tool
   - GBrain MCP client integration (spawn `gbrain serve` as child process)
   - System prompt builder with Alina identity
3. Build basic web client (/call page):
   - WebRTC connection to OpenAI Realtime API
   - RNNoise noise suppression
   - Push-to-talk + auto-VAD
4. Test: Open browser tab, talk to Alina, verify brain pages created post-call
5. Apply latency optimizations from recipe: prompt compression, auth-before-speech, report-aware routing, thinking sounds

### Phase 3: Alina Identity + Agent Switching (Days 4-5)
1. Create Alina SOUL.md from template
2. Create SOUL.md files for other NanoClaw agents
3. Add dropdown UI to web client for agent selection
4. Implement session.update flow for agent switching
5. Test: Switch agents mid-conversation, verify identity changes

### Phase 4: Custom Voice Model (~2 weeks, when local server arrives)
1. Switch to DIY pipeline (Option B): Deepgram STT → Claude API → TTS
2. Replace TTS endpoint with local voice server
3. Tune voice parameters for Alina
4. Test latency and quality
5. May need to adjust audio format compatibility

### Phase 5: Integration with NanoClaw (Week 2+)
1. Register GBrain MCP in NanoClaw container config
2. Wire agent CLAUDE.md files to use brain-first lookup
3. Set up cron jobs for nightly maintenance (lint, sync, embed)
4. Import existing memory.md files into brain pages
5. Configure signal-detector skill for autonomous entity capture

### Ongoing Costs Estimate
| Component | Monthly Cost |
|-----------|-------------|
| PGLite (local) | $0 |
| OpenAI embeddings | ~$5-20 depending on volume |
| OpenAI Realtime voice (100 min) | ~$18 |
| GROQ transcription | Free tier likely sufficient |
| **Total (PGLite + WebRTC)** | **~$25-40/mo** |

If scaling to Supabase later: +$25/mo

---

## HOW THIS RELATES TO THE EXISTING PROPOSAL

Elliott's existing proposal at `/home/elliott/obsidian/Atlas/Resources/proposals/gbrain-implementation.md` focuses on GBrain as a **knowledge brain** for NanoClaw agents (Phase 1-3 in the proposal). This is about the memory/search/compounding-knowledge aspect.

The voice chat project is a **different but complementary** use of GBrain. The voice system uses GBrain as its retrieval and knowledge layer — the agent on the phone/browser queries the brain for context. The two projects share the same GBrain instance but serve different purposes:

- **Knowledge brain**: Agents autonomously building and querying shared wiki
- **Voice chat**: Real-time voice interface that queries the brain for caller context

Both feed data back into the brain. Voice calls create brain pages (meetings/YYYY-MM-DD-call-{caller}.md). The knowledge brain enriches people and companies. The voice agent benefits from that enrichment. It's a flywheel.