# Rules for YouTube Intelligent Classifier

> These rules apply to **all contributors** (human and AI assistants) working on this project.
> Violating any of these rules should be treated as a blocking issue.

---

## 1. Code Integrity — No Hallucination, No Fake Code

- **Never write placeholder or fake implementation code.** Every function, route, and service must do real work.
- **Never hallucinate APIs or libraries.** Only use methods confirmed to exist in the installed package versions.
- **Never fake test results.** Tests must actually exercise the real logic; mocking is allowed only for external I/O (network, DB).
- If you do not know how to implement something, say so explicitly instead of guessing.

## 2. No Hard-Coded Secrets

- API keys, passwords, JWT secrets, and database credentials **must never** appear in source code.
- All secrets live in the `.env` file (see `.env.example`) and are loaded via `core/config.py`.
- The `.env` file is git-ignored. **Never commit it.**

## 3. Type Safety

- All Python functions must have **full type annotations** (`typing`, `pydantic`, `dataclasses`).
- Pydantic schemas (v2) are used for all API request/response validation. No raw `dict` at API boundaries.
- Never use `Any` unless there is a documented reason and it cannot be avoided.

## 4. Async Consistency

- All FastAPI routes and database calls are `async`. Never block the event loop with synchronous I/O.
- CPU-bound work (ML inference, OpenCV, Whisper) must be offloaded via `asyncio.get_event_loop().run_in_executor(...)`.
- Never mix sync and async improperly (e.g., calling `asyncio.run()` inside a running event loop).

## 5. Error Handling

- Never swallow exceptions silently. Always log the exception with context (`logger.exception(...)` or `logger.warning(...)`).
- All external API calls (Spotify, TMDb, yt-dlp) must have `try/except` with graceful degradation.
- Return meaningful HTTP status codes — never return `200` for an error.

## 6. Database Rules

- Structural schema changes **must** go through an Alembic migration (`alembic revision --autogenerate`).
- Never call `Base.metadata.create_all(...)` in production — only in development / test startup.
- Use PostgreSQL for structured metadata. Use MongoDB only for large document payloads (transcripts, full results).
- Always use parameterised queries. **Never** string-format SQL.

## 7. ML / AI Models

- Fine-tuned model checkpoints must not be committed to git. Store them in a dedicated model registry or cloud storage.
- Model loading is cached at the module level (single load per process). Never reload models per request.
- Document the model architecture, dataset, and training procedure before deploying a new model version.

## 8. API Design

- Follow REST conventions: `POST` for creation, `GET` for retrieval, correct HTTP status codes.
- All routes must be registered in `main.py`. A route file that is not imported does nothing.
- Rate limiting is applied globally via `api/middleware/rate_limiter.py`. Never bypass it.
- Paginated endpoints must support `page` and `page_size` query parameters with sensible defaults and maximums.

## 9. Dependencies

- Do not add new Python packages without updating `backend/requirements.txt`.
- Pin exact versions for reproducibility (`package==x.y.z`), not open ranges in production.
- Prefer packages already in the stack before introducing new ones.

## 10. Testing

- Every new service function must have at least one unit test in `tests/unit/`.
- Every new API endpoint must have at least one integration test in `tests/integration/`.
- Tests must be runnable without real external services (mock Spotify, TMDb, DB).
- CI must pass before merging any branch.

## 11. Documentation

- Complex logic must have docstrings. Public functions must always have a short docstring.
- `build.md` must be kept up to date whenever a feature is completed or a module is added.
- `README.md` is the primary user-facing setup guide. Keep it accurate.

## 12. Security

- Use `bcrypt` for password hashing. Never use MD5 or SHA-1 for passwords.
- JWT tokens must be validated for both signature and token type (`access` vs `refresh`).
- CORS origins are controlled by `ALLOWED_ORIGINS` in the config. Never set `allow_origins=["*"]` in production.
- All file paths from user input must be sanitised — never use them directly in `os.path` or `subprocess`.
