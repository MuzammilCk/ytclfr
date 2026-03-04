# YTCLFR — Builder Rules & Constraints
## Absolute Laws for AI-Assisted Development
### Version 1.0 | These Rules Are Non-Negotiable

---

## CRITICAL: READ THIS BEFORE EVERY SINGLE RESPONSE

You are building YTCLFR — a production-grade YouTube intelligent extraction system. Before writing any code, before making any change, before responding to any instruction:

1. **Read `context.md`** — Understand the vision, the problem, the user. Every decision must serve the north star.
2. **Read `build.md`** — Find the current phase. Find the current task within that phase. Work only on that task.
3. **Read the completion criteria** for the current phase. Do not move to the next phase until ALL criteria are met.
4. **After completing any phase or module**, update `build.md` with the completion status, date, and notes before doing anything else.

This is not optional. Skipping these steps is the single biggest source of wasted work.

---

## SECTION 1 — CODE QUALITY LAWS

### LAW 1: No Fake Code. Ever.
Every function you write must be fully implemented. The following are **strictly forbidden**:

```python
# FORBIDDEN:
def extract_songs(frames):
    # TODO: implement this
    pass

def search_spotify(title, artist):
    # Implementation would go here
    return None

def process_frames():
    raise NotImplementedError("Coming soon")
```

If you are asked to implement something and you are not certain how to implement it correctly, say so explicitly and ask for clarification. Do not write a stub and pretend it is an implementation.

**The only acceptable `pass` is in abstract base class definitions.** Everything else must have a real body.

### LAW 2: No Hardcoded Values
Every value that could vary between environments is a configuration. The following are **strictly forbidden**:

```python
# FORBIDDEN:
client = Redis(host="localhost", port=6379)
api_key = "sk-abc123xyz"
model_path = "/home/user/models/whisper-base"
ALLOWED_ORIGINS = ["http://localhost:3000"]  # in code, not config
```

**Every** host, port, credential, path, threshold, model name, and URL must be in `core/config.py` loaded from environment variables.

### LAW 3: No Hallucinated APIs
You must not invent function signatures, library methods, or API endpoints that do not exist. If you are not certain a library function exists, say so. Do not write code that calls `library.method_that_might_exist()` hoping it works.

Verify these before using in code:
- Whisper `task="translate"` — this is real, confirmed in faster-whisper docs
- `model.transcribe()` return value structure — check actual faster-whisper API
- Spotify `user_playlist_create()` parameters — check spotipy docs
- TMDb endpoints — check TMDb API docs

### LAW 4: No Silent Failures
Every exception must be caught, logged, and handled. The following are **strictly forbidden**:

```python
# FORBIDDEN:
try:
    result = do_something()
except:
    pass  # silently ignore

try:
    result = do_something()
except Exception:
    return None  # no logging
```

**Required pattern:**
```python
try:
    result = do_something()
except SpecificException as exc:
    logger.error(f"[{analysis_id}] Step name failed: {exc}")
    # Then either: raise, return fallback, or update status
```

### LAW 5: No Breaking Changes Without Migration
If you change a database schema (PostgreSQL), you must create an Alembic migration. Never modify existing migration files. Never drop a column or table without explicitly being told to. New columns must have defaults so existing rows are not broken.

### LAW 6: No Async/Sync Mixing Without Justification
FastAPI routes must be `async def`. Celery tasks run sync code — use `asyncio.run()` correctly or `loop.run_until_complete()` with proper lifecycle management. Never call `asyncio.new_event_loop()` and abandon the loop. Never call `asyncio.get_event_loop()` in a context where there is no running loop.

### LAW 7: Type Everything
Every function parameter and return value must have a type annotation. This is non-negotiable for the backend. Pydantic models must be used for all API request/response schemas. Dataclasses must be used for internal data transfer objects.

```python
# REQUIRED:
async def search_track(
    self,
    title: str,
    artist: str,
    ocr_raw: Optional[str] = None,
) -> Optional[TrackInfo]:
```

---

## SECTION 2 — ARCHITECTURE LAWS

### LAW 8: OCR Is Primary, Audio Is Supplementary
This is the foundational architectural decision. Never revert to audio-first extraction. If a change you are making would make audio transcription the primary extraction source for music, listicle, or shopping content, stop and re-read `context.md`.

Correct order: **OCR → Audio → Metadata → Heuristics**

### LAW 9: Per-Frame Data Must Be Preserved
Frame OCR results must be stored with their `frame_index`, `timestamp_secs`, and `frame_path`. Never discard per-frame data. Never aggregate frames before storing. The user needs to know WHICH frame a song title came from. This is a core feature, not optional.

### LAW 10: Extractors Must Accept frame_ocr_results
After Phase 1, every extractor's `extract()` method signature must include `frame_ocr_results: List[FrameOCRData]` as a parameter. If you are writing or modifying an extractor that does not have this parameter, you are doing it wrong.

### LAW 11: Never Delete Frames Prematurely
Frames must be served to the frontend for the timeline feature. Do not delete frames immediately after analysis. The `KEEP_FRAMES_AFTER_ANALYSIS` config flag controls deletion. Default is `True` (keep frames).

### LAW 12: One Source of Truth for Config
`core/config.py` is the only place settings live. No `.env` reading anywhere else. No `os.getenv()` calls outside `config.py`. No duplicate defaults in different files.

---

## SECTION 3 — PROCESS LAWS

### LAW 13: Phase Order Is Mandatory
You must complete phases in order: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8. You may not begin Phase 2 if Phase 1 completion criteria are not met. You may not cherry-pick features from a later phase and implement them before an earlier phase is complete.

**Exception:** Bug fixes for crashes or data loss can be done at any time regardless of phase.

### LAW 14: Update build.md After Every Module
When you finish implementing any numbered module (e.g., 1.2 Rewrite MusicExtractor), immediately:
1. Mark it complete in `build.md`
2. Add the completion date
3. Add any notes about implementation decisions made
4. Update the BUILD STATUS TRACKER table at the bottom of `build.md`

Do not proceed to the next module until this is done.

### LAW 15: Test Before Marking Complete
A module is not complete when the code is written. A module is complete when:
1. The code is written
2. The code runs without errors
3. The code passes its tests
4. The completion criteria in `build.md` are met

"It should work" is not verification. Actually run it.

### LAW 16: Never Modify Completed Phases Without Flagging
If implementing a later phase requires changing code from a completed phase, flag it explicitly:
- State what you are changing
- State which completed module it is in
- State why the change is necessary
- Update `build.md` to reflect the change

Never silently modify completed code.

---

## SECTION 4 — ANTI-PATTERN BLACKLIST

These patterns have been seen before and must never appear in this codebase:

### Anti-Pattern 1: OCR on a sample of frames
```python
# FORBIDDEN in extraction path:
sampled_frames = frame_paths[::5]  # every 5th frame
ocr_results = await ocr.extract_from_frames(sampled_frames, max_frames=20)
```
OCR must run on ALL frames in the extraction path. Sampling is only allowed in the lightweight pre-classification preview.

### Anti-Pattern 2: Throwing away frame data
```python
# FORBIDDEN:
ocr_results = await ocr.extract_from_frames(frame_paths)
aggregated = ocr.aggregate_text(ocr_results)  # don't throw away individual results
# passing only aggregated_text to extractor — WRONG
extraction = extractor.extract(ocr_text=aggregated)  
```
Always pass the full `List[FrameOCRData]` to extractors. The aggregated text is a secondary convenience.

### Anti-Pattern 3: Parsing the wrong source
```python
# FORBIDDEN in MusicExtractor:
source = metadata.get("description", "") or transcript_text  # WRONG primary source
raw_tracks = _parse_music_entries(source)
```
Music, listicle, and shopping extractors must parse `frame_ocr_results` first.

### Anti-Pattern 4: Returning nothing for silent videos
```python
# FORBIDDEN:
if not transcript_text.strip():
    return {"type": "music", "tracks": [], "total_count": 0}
```
A silent video with on-screen text is the primary use case. Never short-circuit extraction based on empty transcript.

### Anti-Pattern 5: Swallowing Spotify failures
```python
# FORBIDDEN:
try:
    playlist = await spotify.create_playlist(...)
except Exception:
    pass  # just skip it
```
Spotify failures must be logged and the result must include a `spotify_error` field explaining what failed. The user paid attention to this feature.

### Anti-Pattern 6: Global model loading at import time
```python
# FORBIDDEN at module top level:
classifier = MultiModalClassifier()  # loads BERT and EfficientNet at import
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
```
Models are loaded once in `worker_process_init` signal handler and accessed via `_models` dict. Never load models at import time.

### Anti-Pattern 7: Direct database calls in extractors
```python
# FORBIDDEN:
class MusicExtractor:
    def extract(self, ...):
        with get_sync_db() as session:  # NO
            session.execute(...)
```
Extractors are pure data processing units. They receive data, process it, return data. No database calls inside extractors.

### Anti-Pattern 8: Blocking calls in async context
```python
# FORBIDDEN in async FastAPI routes:
result = requests.get("https://api.spotify.com/...")  # blocking HTTP in async
time.sleep(2)  # blocking sleep in async
```
Use `httpx.AsyncClient` for async HTTP. Use `asyncio.sleep()` for delays. Use `run_in_executor()` for blocking CPU operations.

---

## SECTION 5 — TESTING REQUIREMENTS

### Every new feature must have:

**Unit test** — Tests the function in isolation with mocked dependencies. Must not require:
- Real Spotify API key
- Real TMDb API key
- Real YouTube video download
- Real Tesseract installation
- Real PyTorch/CUDA

**Integration test** — Tests the endpoint end-to-end with mocked external services. Must pass in CI without any API credentials.

### Test file locations:
- Unit tests: `tests/unit/test_{feature_name}.py`
- Integration tests: `tests/integration/test_{feature_name}.py`

### Test naming convention:
```python
class TestMusicExtractor:
    def test_extracts_songs_from_frame_ocr(self):
    def test_handles_no_ocr_text(self):
    def test_deduplicates_songs_across_frames(self):
    def test_falls_back_to_description_when_ocr_empty(self):
```

### Minimum test coverage per phase:
- Phase 0: All existing tests still pass (no regression)
- Phase 1: `TestMusicExtractor`, `TestListicleExtractor`, `TestRecipeExtractor` with OCR input
- Phase 2: `TestTranslation` for non-English audio
- Phase 3: `TestFrameEndpoint` for `/frames/{id}` serving
- Phase 4: `TestSpotifyMatching` for multi-strategy search
- Phase 5: `TestTMDbEnrichment` with year hints

---

## SECTION 6 — COMMUNICATION RULES

### Rule 1: State What Phase You Are In
Every implementation response must begin with:
```
## Working on Phase X — Module X.Y: [Name]
Current status: [what was done before / what is being done now]
```

### Rule 2: Flag Blockers Immediately
If you encounter an issue that prevents completing the current module:
- State the blocker clearly
- Do not silently skip it
- Do not implement a workaround without explaining it
- Ask for a decision before proceeding

### Rule 3: No Out-of-Scope Work
If the current task is "Rewrite MusicExtractor" (Phase 1.2), do not also:
- Rewrite the frontend MusicOutput component
- Add a new API endpoint
- Refactor the database schema
- "Clean up" unrelated code

Do exactly what the current module requires. Nothing more. Nothing less.

### Rule 4: Declare All Files Being Changed
Before making changes, list every file that will be modified:
```
Files being modified:
- backend/services/extraction/extractors.py (rewrite MusicExtractor.extract())
- backend/services/pipeline.py (pass frame_ocr_results to extractor)
- tests/unit/test_extraction.py (add TestMusicExtractor.test_extracts_from_ocr)
```

### Rule 5: Summarize Changes After Completion
After completing a module, provide:
1. What was changed and in which files
2. What the output looks like (sample data structure)
3. What test to run to verify
4. Any known limitations or follow-up needed in a later phase

---

## SECTION 7 — THE NORTH STAR RULE

When in doubt about any decision, ask this question:

> *"If someone pastes a YouTube link for 'Top 20 Songs of 2020' where the audio is just music playing — will this change help the system return a Spotify playlist with all 20 songs?"*

If the answer is no, or unclear, the change is wrong or not the priority.

Every architectural decision, every feature, every refactor must serve this goal.

**This is the rule above all rules.**

---

## SECTION 8 — FORBIDDEN SHORTCUTS

The following shortcuts are explicitly banned regardless of time pressure:

| Shortcut | Why Forbidden |
|---|---|
| Using `max_frames` limit on extraction OCR | Causes missed songs in frames not sampled |
| Returning empty result for silent videos | Destroys the primary use case |
| Parsing only description for music titles | Fails for videos without tracklists in description |
| Skipping deduplication across frames | Returns duplicate songs in playlist |
| Not storing per-frame OCR in MongoDB | Breaks the timeline feature forever |
| Using synchronous requests in async routes | Blocks the event loop, degrades all users |
| Hardcoding localhost in any URL | Breaks in production, Docker, or any non-local env |
| Skipping Alembic migration for schema change | Corrupts production database on upgrade |
| Storing secrets in code | Security breach waiting to happen |
| Using `print()` instead of `logger` | Logs not captured in production log aggregation |

---

## VERSION HISTORY

| Version | Date | Changes |
|---|---|---|
| 1.0 | Initial | Document created |
