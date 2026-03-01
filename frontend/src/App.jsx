import { useState, useEffect, useRef, useCallback } from "react";

// ── CSS Variables + Global Styles injected as a style tag ──────────────────
const GLOBAL_STYLES = `
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:          #08080f;
    --surface:     #0f0f1c;
    --surface2:    #151528;
    --border:      rgba(255,255,255,0.07);
    --accent:      #00e5ff;
    --accent2:     #7c3aed;
    --accent3:     #f59e0b;
    --text:        #e8e8f0;
    --muted:       rgba(232,232,240,0.45);
    --danger:      #ff4d6d;
    --success:     #10b981;
    --radius:      12px;
    --radius-lg:   20px;
    --shadow:      0 8px 40px rgba(0,229,255,0.08);
  }

  html { font-size: 16px; scroll-behavior: smooth; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Syne', sans-serif;
    line-height: 1.6;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--accent2); border-radius: 3px; }

  /* Noise overlay */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.035'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 9999;
    opacity: 0.4;
  }

  /* Animations */
  @keyframes pulse-ring {
    0%   { transform: scale(0.85); opacity: 1; }
    70%  { transform: scale(1.4);  opacity: 0; }
    100% { transform: scale(1.4);  opacity: 0; }
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes shimmer {
    0%   { background-position: -1000px 0; }
    100% { background-position: 1000px 0; }
  }
  @keyframes glitch {
    0%,100% { clip-path: inset(0 0 98% 0); transform: translate(-2px); }
    20%      { clip-path: inset(33% 0 60% 0); transform: translate(2px); }
    40%      { clip-path: inset(70% 0 2% 0); transform: translate(-1px); }
    60%      { clip-path: inset(10% 0 85% 0); transform: translate(1px); }
    80%      { clip-path: inset(90% 0 1% 0); transform: translate(-2px); }
  }
  @keyframes scanline {
    from { background-position: 0 -100%; }
    to   { background-position: 0 100%; }
  }

  .fade-up { animation: fadeUp 0.5s ease forwards; }

  .skeleton {
    background: linear-gradient(90deg, var(--surface) 25%, var(--surface2) 50%, var(--surface) 75%);
    background-size: 1000px 100%;
    animation: shimmer 1.6s infinite;
    border-radius: var(--radius);
  }

  .mono { font-family: 'Space Mono', monospace; }
`;

// ── Utility functions ──────────────────────────────────────────────────────
const API = import.meta?.env?.VITE_API_URL || "http://localhost:8000";

function getAuthHeader() {
  const token = localStorage.getItem("yt_token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function apiPost(path, body, asBlob = false) {
  const res = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...getAuthHeader() },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return asBlob ? res.blob() : res.json();
}

async function apiGet(path) {
  const res = await fetch(`${API}${path}`, {
    headers: { ...getAuthHeader() },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

async function apiAuthPost(path, body) {
  const res = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function formatDuration(secs) {
  if (!secs) return "--";
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = secs % 60;
  return h > 0
    ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`
    : `${m}:${String(s).padStart(2, "0")}`;
}

function formatViews(n) {
  if (!n) return "--";
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`;
  return String(n);
}

// ── Category metadata ──────────────────────────────────────────────────────
const CATEGORY_META = {
  comedy: { icon: "🎭", color: "#f59e0b", label: "Comedy / Entertainment" },
  listicle: { icon: "📋", color: "#00e5ff", label: "Listicle / Ranking" },
  music: { icon: "🎵", color: "#a78bfa", label: "Music Compilation" },
  educational: { icon: "🎓", color: "#10b981", label: "Educational / Tutorial" },
  news: { icon: "📰", color: "#f97316", label: "News / Documentary" },
  review: { icon: "⭐", color: "#facc15", label: "Product Review" },
  gaming: { icon: "🎮", color: "#ec4899", label: "Gaming / Esports" },
  vlog: { icon: "📹", color: "#38bdf8", label: "Vlog / Lifestyle" },
  unknown: { icon: "❓", color: "#6b7280", label: "Unknown" },
};

const STATUS_STEPS = [
  { key: "queued", label: "Queued" },
  { key: "downloading", label: "Downloading" },
  { key: "extracting_frames", label: "Extracting Frames" },
  { key: "transcribing", label: "Transcribing Audio" },
  { key: "classifying", label: "Classifying" },
  { key: "extracting_info", label: "Extracting Info" },
  { key: "enriching", label: "Enriching Data" },
  { key: "complete", label: "Complete" },
];

// ── Sub-components ─────────────────────────────────────────────────────────

function GlowButton({ children, onClick, disabled, variant = "primary", style = {} }) {
  const base = {
    display: "inline-flex",
    alignItems: "center",
    gap: 8,
    padding: "12px 28px",
    borderRadius: "var(--radius)",
    border: "none",
    cursor: disabled ? "not-allowed" : "pointer",
    fontFamily: "'Syne', sans-serif",
    fontWeight: 700,
    fontSize: 15,
    transition: "all 0.2s",
    opacity: disabled ? 0.5 : 1,
    ...style,
  };

  const styles = {
    primary: {
      ...base,
      background: "linear-gradient(135deg, var(--accent), #0090aa)",
      color: "#08080f",
      boxShadow: "0 0 20px rgba(0,229,255,0.3)",
    },
    secondary: {
      ...base,
      background: "transparent",
      color: "var(--accent)",
      border: "1px solid var(--accent)",
      boxShadow: "none",
    },
    danger: {
      ...base,
      background: "linear-gradient(135deg, var(--danger), #c9184a)",
      color: "white",
    },
  };

  return (
    <button onClick={onClick} disabled={disabled} style={styles[variant]}>
      {children}
    </button>
  );
}

function Card({ children, style = {}, glow = false }) {
  return (
    <div style={{
      background: "var(--surface)",
      border: `1px solid ${glow ? "rgba(0,229,255,0.2)" : "var(--border)"}`,
      borderRadius: "var(--radius-lg)",
      padding: "24px",
      boxShadow: glow ? "0 0 40px rgba(0,229,255,0.06), inset 0 0 40px rgba(0,229,255,0.02)" : "none",
      ...style,
    }}>
      {children}
    </div>
  );
}

function Tag({ label, color = "var(--accent)" }) {
  return (
    <span style={{
      display: "inline-flex",
      alignItems: "center",
      padding: "3px 10px",
      background: `${color}18`,
      color,
      border: `1px solid ${color}40`,
      borderRadius: 20,
      fontSize: 12,
      fontWeight: 700,
      fontFamily: "'Space Mono', monospace",
      letterSpacing: "0.05em",
      textTransform: "uppercase",
    }}>
      {label}
    </span>
  );
}

function ProgressBar({ value, max = 1, color = "var(--accent)" }) {
  return (
    <div style={{
      width: "100%", height: 6, background: "var(--surface2)",
      borderRadius: 3, overflow: "hidden",
    }}>
      <div style={{
        height: "100%",
        width: `${Math.min(100, (value / max) * 100)}%`,
        background: `linear-gradient(90deg, ${color}, ${color}90)`,
        borderRadius: 3,
        transition: "width 0.6s ease",
      }} />
    </div>
  );
}

function Spinner({ size = 24, color = "var(--accent)" }) {
  return (
    <div style={{
      width: size, height: size,
      border: `2px solid ${color}30`,
      borderTopColor: color,
      borderRadius: "50%",
      animation: "spin 0.8s linear infinite",
      flexShrink: 0,
    }} />
  );
}

// ── Hero / URL Input ───────────────────────────────────────────────────────

function HeroInput({ onSubmit, loading }) {
  const [url, setUrl] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const inputRef = useRef(null);

  const handleSubmit = async () => {
    setError("");
    if (!url.trim()) { setError("Enter a YouTube URL"); return; }
    if (!url.includes("youtube.com") && !url.includes("youtu.be") && !url.includes("youtube.com/shorts/")) {
      setError("Must be a valid YouTube URL");
      return;
    }
    setSubmitting(true);
    try {
      await onSubmit(url.trim());
    } catch (e) {
      setError(e.message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div style={{ maxWidth: 760, margin: "0 auto", animation: "fadeUp 0.6s ease forwards" }}>
      {/* Hero text */}
      <div style={{ textAlign: "center", marginBottom: 48 }}>
        <div style={{
          display: "inline-flex", alignItems: "center", gap: 8,
          padding: "6px 16px", borderRadius: 20,
          background: "rgba(0,229,255,0.08)", border: "1px solid rgba(0,229,255,0.2)",
          fontSize: 12, fontFamily: "'Space Mono', monospace",
          color: "var(--accent)", letterSpacing: "0.1em",
          marginBottom: 24,
        }}>
          ◉ AI-POWERED · MULTI-MODAL · REAL-TIME
        </div>

        <h1 style={{
          fontSize: "clamp(36px, 6vw, 72px)",
          fontWeight: 800,
          lineHeight: 1.1,
          letterSpacing: "-0.02em",
          marginBottom: 20,
        }}>
          <span style={{ color: "var(--text)" }}>Understand any</span>
          <br />
          <span style={{
            background: "linear-gradient(135deg, var(--accent), var(--accent2))",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}>YouTube video</span>
          <br />
          <span style={{ color: "var(--text)" }}>in seconds.</span>
        </h1>

        <p style={{ color: "var(--muted)", fontSize: 18, maxWidth: 540, margin: "0 auto" }}>
          Paste any YouTube link. Our AI classifies the content and extracts
          structured information — transcripts, ranked lists, music tracks,
          movie metadata, and more.
        </p>
      </div>

      {/* Input */}
      <Card glow style={{ padding: 8 }}>
        <div style={{ display: "flex", gap: 8 }}>
          <div style={{ flex: 1, position: "relative" }}>
            <div style={{
              position: "absolute", left: 16, top: "50%", transform: "translateY(-50%)",
              color: "var(--muted)", fontSize: 20, pointerEvents: "none",
            }}>▶</div>
            <input
              ref={inputRef}
              value={url}
              onChange={e => setUrl(e.target.value)}
              onKeyDown={e => e.key === "Enter" && !(loading || submitting) && handleSubmit()}
              placeholder="https://youtube.com/watch?v=..."
              disabled={loading || submitting}
              style={{
                width: "100%", padding: "16px 16px 16px 48px",
                background: "var(--surface2)",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius)",
                color: "var(--text)",
                fontSize: 16,
                fontFamily: "'Space Mono', monospace",
                outline: "none",
                transition: "border-color 0.2s",
              }}
              onFocus={e => e.target.style.borderColor = "var(--accent)"}
              onBlur={e => e.target.style.borderColor = "var(--border)"}
            />
          </div>
          <GlowButton onClick={handleSubmit} disabled={loading || submitting}>
            {loading || submitting ? <Spinner size={18} color="#08080f" /> : "Analyse"}
          </GlowButton>
        </div>
        {error && (
          <div style={{ padding: "8px 8px 0", color: "var(--danger)", fontSize: 13, fontFamily: "'Space Mono', monospace" }}>
            ⚠ {error}
          </div>
        )}
      </Card>

      {/* Category chips */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8, justifyContent: "center", marginTop: 24 }}>
        {Object.entries(CATEGORY_META).filter(([k]) => k !== "unknown").map(([key, meta]) => (
          <span key={key} style={{
            padding: "4px 12px", borderRadius: 20,
            background: `${meta.color}12`,
            border: `1px solid ${meta.color}30`,
            color: meta.color, fontSize: 13,
          }}>
            {meta.icon} {meta.label}
          </span>
        ))}
      </div>
    </div>
  );
}

// ── Processing Status Tracker ──────────────────────────────────────────────

function ProcessingStatus({ status, analysisId }) {
  const currentStep = STATUS_STEPS.findIndex(s => s.key === status);
  const progress = Math.max(0, currentStep) / (STATUS_STEPS.length - 1);

  return (
    <Card glow style={{ animation: "fadeUp 0.5s ease" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 24 }}>
        <Spinner size={20} />
        <h3 style={{ fontWeight: 700 }}>Analysing your video…</h3>
        <Tag label={`${Math.round(progress * 100)}%`} color="var(--accent)" />
      </div>

      <ProgressBar value={progress} max={1} />

      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(4, 1fr)",
        gap: 12,
        marginTop: 20,
      }}>
        {STATUS_STEPS.map((step, i) => {
          const done = i < currentStep;
          const active = i === currentStep;
          return (
            <div key={step.key} style={{
              padding: "10px 14px",
              borderRadius: "var(--radius)",
              background: done ? "rgba(16,185,129,0.1)" : active ? "rgba(0,229,255,0.08)" : "var(--surface2)",
              border: `1px solid ${done ? "rgba(16,185,129,0.3)" : active ? "rgba(0,229,255,0.3)" : "var(--border)"}`,
              transition: "all 0.3s",
            }}>
              <div style={{ fontSize: 12, marginBottom: 2, color: done ? "var(--success)" : active ? "var(--accent)" : "var(--muted)" }}>
                {done ? "✓" : active ? "⟳" : "○"}
              </div>
              <div style={{
                fontSize: 12,
                fontFamily: "'Space Mono', monospace",
                color: done ? "var(--success)" : active ? "var(--accent)" : "var(--muted)",
                fontWeight: active ? 700 : 400,
              }}>
                {step.label}
              </div>
            </div>
          );
        })}
      </div>

      <div style={{ marginTop: 16, fontFamily: "'Space Mono', monospace", fontSize: 12, color: "var(--muted)" }}>
        Job ID: {analysisId}
      </div>
    </Card>
  );
}

// ── Video Meta Header ──────────────────────────────────────────────────────

function VideoHeader({ video, classification }) {
  const cat = CATEGORY_META[classification?.predicted_category] || CATEGORY_META.unknown;

  return (
    <div style={{ display: "flex", gap: 20, flexWrap: "wrap", marginBottom: 24 }}>
      {video.thumbnail_url && (
        <div style={{ position: "relative", flexShrink: 0 }}>
          <img
            src={video.thumbnail_url}
            alt=""
            style={{
              width: 220, height: 124, objectFit: "cover",
              borderRadius: "var(--radius)",
              border: "1px solid var(--border)",
            }}
          />
          <div style={{
            position: "absolute", bottom: 8, right: 8,
            background: "rgba(0,0,0,0.8)", color: "white",
            fontSize: 12, padding: "2px 6px", borderRadius: 4,
            fontFamily: "'Space Mono', monospace",
          }}>
            {formatDuration(video.duration_secs)}
          </div>
        </div>
      )}

      <div style={{ flex: 1, minWidth: 0 }}>
        <h2 style={{
          fontSize: 20, fontWeight: 800, lineHeight: 1.3,
          marginBottom: 10, letterSpacing: "-0.01em",
        }}>
          {video.title}
        </h2>

        <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 12 }}>
          <Tag label={`${cat.icon} ${cat.label}`} color={cat.color} />
          <Tag label={`${Math.round((classification?.confidence || 0) * 100)}% confidence`} color={cat.color} />
          {video.language && <Tag label={video.language.toUpperCase()} color="var(--muted)" />}
        </div>

        <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
          {[
            { label: "Views", value: formatViews(video.view_count) },
            { label: "Channel", value: video.channel_name },
            { label: "Duration", value: formatDuration(video.duration_secs) },
            { label: "Uploaded", value: video.upload_date },
          ].map(item => (
            <div key={item.label}>
              <div style={{ fontSize: 11, color: "var(--muted)", fontFamily: "'Space Mono', monospace", textTransform: "uppercase", letterSpacing: "0.1em" }}>
                {item.label}
              </div>
              <div style={{ fontSize: 14, fontWeight: 600, marginTop: 2 }}>{item.value || "--"}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Classification Scores ──────────────────────────────────────────────────

function ClassificationScores({ allScores, modalityBreakdown }) {
  const sorted = Object.entries(allScores || {}).sort(([, a], [, b]) => b - a);

  return (
    <Card style={{ marginBottom: 20 }}>
      <h3 style={{ fontWeight: 700, marginBottom: 16, fontSize: 14, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.08em", fontFamily: "'Space Mono', monospace" }}>
        Classification Scores
      </h3>
      <div style={{ display: "grid", gap: 10 }}>
        {sorted.map(([cat, score]) => {
          const meta = CATEGORY_META[cat] || CATEGORY_META.unknown;
          return (
            <div key={cat} style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <div style={{ width: 100, fontSize: 13, color: "var(--muted)" }}>
                {meta.icon} {cat}
              </div>
              <div style={{ flex: 1 }}>
                <ProgressBar value={score} max={1} color={meta.color} />
              </div>
              <div style={{ width: 48, textAlign: "right", fontSize: 13, fontFamily: "'Space Mono', monospace", color: meta.color }}>
                {(score * 100).toFixed(1)}%
              </div>
            </div>
          );
        })}
      </div>

      {modalityBreakdown && Object.keys(modalityBreakdown).length > 0 && (
        <div style={{ marginTop: 16, padding: "12px 16px", background: "var(--surface2)", borderRadius: "var(--radius)" }}>
          <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 8, fontFamily: "'Space Mono', monospace", textTransform: "uppercase" }}>
            Modality Breakdown (top class)
          </div>
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
            {Object.entries(modalityBreakdown).map(([mod, val]) => (
              <div key={mod} style={{ fontSize: 13 }}>
                <span style={{ color: "var(--muted)" }}>{mod}:</span>{" "}
                <span style={{ color: "var(--accent)", fontFamily: "'Space Mono', monospace" }}>
                  {(val * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
}

// ── Transcript Panel ───────────────────────────────────────────────────────

function TranscriptPanel({ transcription }) {
  const [expanded, setExpanded] = useState(false);
  if (!transcription?.full_text) return null;

  const preview = transcription.full_text.slice(0, 600);

  return (
    <Card style={{ marginBottom: 20 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <h3 style={{ fontWeight: 700 }}>Transcript</h3>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <Tag label={`${transcription.word_count?.toLocaleString() || 0} words`} />
          <Tag label={transcription.language?.toUpperCase()} color="var(--accent2)" />
        </div>
      </div>

      <div style={{
        fontFamily: "'Space Mono', monospace",
        fontSize: 13,
        lineHeight: 1.8,
        color: "var(--muted)",
        maxHeight: expanded ? "none" : 120,
        overflow: "hidden",
        position: "relative",
      }}>
        {expanded ? transcription.full_text : preview + (transcription.full_text.length > 600 ? "…" : "")}
        {!expanded && transcription.full_text.length > 600 && (
          <div style={{
            position: "absolute", bottom: 0, left: 0, right: 0, height: 40,
            background: "linear-gradient(transparent, var(--surface))",
          }} />
        )}
      </div>

      {transcription.full_text.length > 600 && (
        <button onClick={() => setExpanded(!expanded)} style={{
          marginTop: 10, background: "none", border: "none",
          color: "var(--accent)", cursor: "pointer", fontSize: 13,
          fontFamily: "'Space Mono', monospace",
        }}>
          {expanded ? "↑ Show less" : "↓ Read full transcript"}
        </button>
      )}
    </Card>
  );
}

// ── Listicle Output ────────────────────────────────────────────────────────

function ListicleOutput({ output }) {
  const items = output?.items || [];

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <h3 style={{ fontWeight: 800, fontSize: 18 }}>{output?.list_title}</h3>
        <Tag label={`${items.length} items`} />
      </div>

      <div style={{ display: "grid", gap: 12 }}>
        {items.map((item, i) => (
          <Card key={i} style={{ padding: 16, display: "flex", gap: 16 }}>
            <div style={{
              width: 48, height: 48, borderRadius: "var(--radius)", flexShrink: 0,
              background: "linear-gradient(135deg, var(--accent2), var(--accent))",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontWeight: 800, fontSize: 18, color: "white",
            }}>
              {item.rank}
            </div>

            {item.poster_url && (
              <img
                src={item.poster_url}
                alt={item.title}
                style={{ width: 40, height: 60, objectFit: "cover", borderRadius: 6, flexShrink: 0 }}
              />
            )}

            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontWeight: 700, marginBottom: 4 }}>{item.title}</div>
              {item.year && <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 6 }}>{item.year}</div>}
              {item.description && (
                <div style={{ fontSize: 13, color: "var(--muted)", lineHeight: 1.5, marginBottom: 8 }}>
                  {item.description.slice(0, 120)}{item.description.length > 120 ? "…" : ""}
                </div>
              )}
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {item.tmdb_rating && (
                  <Tag label={`⭐ ${item.tmdb_rating}`} color="var(--accent3)" />
                )}
                {item.streaming?.flatrate?.slice(0, 3).map(p => (
                  <Tag key={p} label={p} color="var(--success)" />
                ))}
                {item.imdb_url && (
                  <a href={item.imdb_url} target="_blank" rel="noopener noreferrer" style={{ color: "var(--accent)", fontSize: 12, textDecoration: "none" }}>
                    IMDb →
                  </a>
                )}
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}

// ── Music Output ───────────────────────────────────────────────────────────

function MusicOutput({ output }) {
  const tracks = output?.tracks || [];

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <h3 style={{ fontWeight: 800, fontSize: 18 }}>Tracks Found</h3>
        <div style={{ display: "flex", gap: 8 }}>
          <Tag label={`${tracks.length} tracks`} color="var(--accent2)" />
          {output?.spotify_playlist_url && (
            <a href={output.spotify_playlist_url} target="_blank" rel="noopener noreferrer">
              <Tag label="Open Spotify Playlist" color="var(--success)" />
            </a>
          )}
        </div>
      </div>

      <div style={{ display: "grid", gap: 8 }}>
        {tracks.map((track, i) => (
          <div key={i} style={{
            display: "flex", alignItems: "center", gap: 16,
            padding: "12px 16px",
            background: "var(--surface)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius)",
            transition: "border-color 0.2s",
          }}>
            <div style={{
              width: 32, height: 32, borderRadius: "50%",
              background: `hsl(${(i * 47) % 360}, 70%, 50%)`,
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 13, color: "white", fontWeight: 700, flexShrink: 0,
            }}>
              {track.rank || i + 1}
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontWeight: 600, fontSize: 14, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {track.title}
              </div>
              <div style={{ fontSize: 12, color: "var(--muted)" }}>
                {track.artist} {track.year ? `· ${track.year}` : ""}
              </div>
            </div>
            <div style={{ display: "flex", gap: 8, flexShrink: 0 }}>
              {track.spotify?.found && (
                <a
                  href={track.spotify.spotify_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{
                    padding: "4px 12px", borderRadius: 20,
                    background: "rgba(29,185,84,0.15)",
                    border: "1px solid rgba(29,185,84,0.3)",
                    color: "#1db954", fontSize: 12, textDecoration: "none",
                    fontFamily: "'Space Mono', monospace",
                  }}
                >
                  ♫ Spotify
                </a>
              )}
              {track.spotify?.preview_url && (
                <audio controls src={track.spotify.preview_url} style={{ height: 24 }} />
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Educational Output ─────────────────────────────────────────────────────

function EducationalOutput({ output }) {
  const [openChapter, setOpenChapter] = useState(null);

  return (
    <div>
      <div style={{ marginBottom: 20 }}>
        <h3 style={{ fontWeight: 800, fontSize: 18, marginBottom: 8 }}>Summary</h3>
        <p style={{ color: "var(--muted)", lineHeight: 1.7 }}>{output?.summary?.slice(0, 500)}</p>
      </div>

      {output?.key_concepts?.length > 0 && (
        <div style={{ marginBottom: 20 }}>
          <h4 style={{ fontWeight: 700, marginBottom: 10, fontSize: 14 }}>Key Concepts</h4>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {output.key_concepts.slice(0, 12).map(c => (
              <Tag key={c} label={c} color="var(--success)" />
            ))}
          </div>
        </div>
      )}

      {output?.chapters?.length > 0 && (
        <div>
          <h4 style={{ fontWeight: 700, marginBottom: 12, fontSize: 14 }}>Chapters</h4>
          <div style={{ display: "grid", gap: 8 }}>
            {output.chapters.map((ch, i) => (
              <div key={i} style={{
                border: "1px solid var(--border)", borderRadius: "var(--radius)",
                overflow: "hidden",
              }}>
                <button onClick={() => setOpenChapter(openChapter === i ? null : i)} style={{
                  width: "100%", padding: "12px 16px",
                  background: openChapter === i ? "var(--surface2)" : "var(--surface)",
                  border: "none", cursor: "pointer", textAlign: "left",
                  display: "flex", alignItems: "center", gap: 12,
                }}>
                  <div style={{
                    width: 28, height: 28, borderRadius: "50%",
                    background: "linear-gradient(135deg, var(--success), #059669)",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 12, color: "white", fontWeight: 700, flexShrink: 0,
                  }}>
                    {ch.index}
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontWeight: 600, color: "var(--text)", fontSize: 14 }}>{ch.title}</div>
                    <div style={{ fontSize: 12, color: "var(--muted)", fontFamily: "'Space Mono', monospace" }}>
                      {formatDuration(Math.round(ch.start_secs))}
                      {ch.end_secs ? ` – ${formatDuration(Math.round(ch.end_secs))}` : ""}
                    </div>
                  </div>
                  <span style={{ color: "var(--muted)" }}>{openChapter === i ? "▲" : "▼"}</span>
                </button>
                {openChapter === i && (
                  <div style={{ padding: "12px 16px", background: "var(--surface2)" }}>
                    <p style={{ fontSize: 13, color: "var(--muted)", lineHeight: 1.6, marginBottom: 8 }}>
                      {ch.summary || "No summary available."}
                    </p>
                    {ch.key_concepts?.length > 0 && (
                      <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                        {ch.key_concepts.map(c => <Tag key={c} label={c} color="var(--accent2)" />)}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Generic Output ─────────────────────────────────────────────────────────

function GenericOutput({ output }) {
  return (
    <div>
      <div style={{ marginBottom: 20 }}>
        <h3 style={{ fontWeight: 700, marginBottom: 8 }}>Summary</h3>
        <p style={{ color: "var(--muted)", lineHeight: 1.7 }}>{output?.summary}</p>
      </div>
      {output?.key_points?.length > 0 && (
        <div style={{ marginBottom: 20 }}>
          <h4 style={{ fontWeight: 700, marginBottom: 10 }}>Key Topics</h4>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {output.key_points.slice(0, 16).map(p => (
              <Tag key={p} label={p} color="var(--accent)" />
            ))}
          </div>
        </div>
      )}
      {output?.named_entities && (
        <div>
          <h4 style={{ fontWeight: 700, marginBottom: 10 }}>Entities Detected</h4>
          {Object.entries(output.named_entities).slice(0, 6).map(([type, ents]) => (
            <div key={type} style={{ marginBottom: 10 }}>
              <div style={{ fontSize: 11, color: "var(--muted)", fontFamily: "'Space Mono', monospace", marginBottom: 6 }}>
                {type}
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {ents.slice(0, 8).map(e => <Tag key={e} label={e} color="var(--accent3)" />)}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Shopping Output ────────────────────────────────────────────────

function ShoppingOutput({ output }) {
  const products = output?.products || [];
  const brands = output?.brand_mentions || [];

  if (!products.length) {
    return (
      <div style={{ textAlign: "center", padding: 40, color: "var(--muted)" }}>
        🛍️ No shoppable products detected in this video.
      </div>
    );
  }

  return (
    <div>
      {/* Summary bar */}
      <div style={{
        display: "flex", justifyContent: "space-between", alignItems: "center",
        marginBottom: 20, padding: "12px 16px",
        background: "var(--surface2)", borderRadius: "var(--radius)",
        border: "1px solid var(--border)",
      }}>
        <span style={{ fontWeight: 700, fontSize: 14 }}>
          🛍️ {products.length} Product{products.length !== 1 ? "s" : ""} detected
        </span>
        {brands.length > 0 && (
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            {brands.map(b => <Tag key={b} label={b} color="var(--accent2)" />)}
          </div>
        )}
      </div>

      {/* Product cards grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 16 }}>
        {products.map((p, i) => (
          <Card key={i} style={{ padding: 16, display: "flex", flexDirection: "column", gap: 10 }}>
            {/* Header */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
              <div style={{ fontWeight: 700, fontSize: 15, lineHeight: 1.3 }}>{p.name}</div>
              <div style={{
                fontSize: 10, fontFamily: "'Space Mono', monospace",
                padding: "2px 8px", borderRadius: 20,
                background: p.detection_source === "yolo" ? "rgba(99,202,183,0.15)" : "rgba(155,135,245,0.15)",
                color: p.detection_source === "yolo" ? "var(--success)" : "var(--accent3)",
                letterSpacing: "0.05em",
              }}>
                {p.detection_source === "yolo" ? "VISION" : "NLP"}
              </div>
            </div>

            {/* Meta tags */}
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              {p.category && <Tag label={p.category} color="var(--accent)" />}
              {p.brand && <Tag label={p.brand} color="var(--accent2)" />}
            </div>

            {/* Confidence bar */}
            {p.confidence != null && (
              <div>
                <div style={{
                  fontSize: 11, color: "var(--muted)", marginBottom: 4,
                  fontFamily: "'Space Mono', monospace"
                }}>
                  Confidence: {Math.round(p.confidence * 100)}%
                </div>
                <div style={{ height: 4, borderRadius: 2, background: "var(--border)" }}>
                  <div style={{
                    height: "100%", borderRadius: 2,
                    width: `${Math.round(p.confidence * 100)}%`,
                    background: "linear-gradient(90deg, var(--accent), var(--success))",
                    transition: "width 0.6s ease",
                  }} />
                </div>
              </div>
            )}

            {/* Frame count */}
            {p.frame_timestamps?.length > 0 && (
              <div style={{ fontSize: 12, color: "var(--muted)" }}>
                Seen in {p.frame_timestamps.length} frame{p.frame_timestamps.length !== 1 ? "s" : ""}
              </div>
            )}

            {/* Shop link */}
            {p.search_url && (
              <a
                href={p.search_url}
                target="_blank"
                rel="noopener noreferrer"
                style={{ marginTop: 4, textDecoration: "none" }}
              >
                <GlowButton style={{
                  width: "100%", fontSize: 12, padding: "8px 0",
                  justifyContent: "center", display: "flex"
                }}>
                  🛒 Shop on Google
                </GlowButton>
              </a>
            )}
          </Card>
        ))}
      </div>
    </div>
  );
}

// ── Full Result View ───────────────────────────────────────────────────────

function ResultView({ result, onReset }) {
  const [tab, setTab] = useState("output");
  const cat = result?.classification?.predicted_category;
  const output = result?.output;

  const renderOutput = () => {
    switch (cat) {
      case "listicle": return <ListicleOutput output={output} />;
      case "music": return <MusicOutput output={output} />;
      case "educational": return <EducationalOutput output={output} />;
      case "comedy": return <ComedyOutput output={output} />;
      case "shopping": return <ShoppingOutput output={output} />;
      default: return <GenericOutput output={output} />;
    }
  };

  return (
    <div style={{ animation: "fadeUp 0.5s ease" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
        <h2 style={{ fontWeight: 800, fontSize: 22 }}>Analysis Complete</h2>
        <div style={{ display: "flex", gap: 10 }}>
          <Tag label={`${result?.processing_time_secs?.toFixed(1)}s`} color="var(--muted)" />
          <GlowButton variant="secondary" onClick={onReset} style={{ padding: "8px 16px", fontSize: 13 }}>
            ← New Analysis
          </GlowButton>
        </div>
      </div>

      <VideoHeader video={result?.video} classification={result?.classification} />

      {/* Tabs */}
      <div style={{ display: "flex", gap: 4, marginBottom: 20, borderBottom: "1px solid var(--border)", paddingBottom: 0 }}>
        {["output", "classification", "transcript"].map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: "10px 20px",
            background: "none",
            border: "none",
            borderBottom: `2px solid ${tab === t ? "var(--accent)" : "transparent"}`,
            color: tab === t ? "var(--accent)" : "var(--muted)",
            cursor: "pointer",
            fontFamily: "'Space Mono', monospace",
            fontSize: 13,
            fontWeight: tab === t ? 700 : 400,
            letterSpacing: "0.05em",
            textTransform: "uppercase",
            transition: "all 0.2s",
          }}>
            {t}
          </button>
        ))}
      </div>

      {tab === "output" && renderOutput()}
      {tab === "classification" && (
        <ClassificationScores
          allScores={result?.classification?.all_scores}
          modalityBreakdown={result?.classification?.modality_breakdown}
        />
      )}
      {tab === "transcript" && (
        <TranscriptPanel transcription={result?.transcription} />
      )}

      {/* Export buttons — POST request, triggers Blob download */}
      <div style={{ display: "flex", gap: 10, marginTop: 24, paddingTop: 24, borderTop: "1px solid var(--border)" }}>
        <span style={{ color: "var(--muted)", fontSize: 13, alignSelf: "center" }}>Export:</span>
        {["json", "csv", "pdf"].map(fmt => (
          <ExportButton key={fmt} fmt={fmt} analysisId={result?.analysis_id} />
        ))}
      </div>
    </div>
  );
}

// ── Export Button (POST → Blob download) ─────────────────────────────────

function ExportButton({ fmt, analysisId }) {
  const [loading, setLoading] = useState(false);

  const handleExport = async () => {
    if (!analysisId) return;
    setLoading(true);
    try {
      const blob = await apiPost(
        "/api/v1/analyses/export",
        { analysis_id: analysisId, format: fmt },
        true, // asBlob
      );
      const ext = fmt === "pdf" ? "pdf" : fmt === "csv" ? "csv" : "json";
      triggerDownload(blob, `analysis-${analysisId.slice(0, 8)}.${ext}`);
    } catch (e) {
      console.error("Export failed:", e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <GlowButton
      variant="secondary"
      style={{ padding: "8px 16px", fontSize: 13 }}
      onClick={handleExport}
      disabled={loading || !analysisId}
    >
      {loading ? <Spinner size={14} color="var(--accent)" /> : fmt.toUpperCase()}
    </GlowButton>
  );
}

// ── Comedy Output Component ───────────────────────────────────────────────

function ComedyOutput({ output }) {
  if (!output) return <GenericOutput output={output} />;
  const punchlines = output.punchlines || [];
  const comedians = output.key_comedians || output.comedians || [];
  const topics = output.topics || [];
  const sentimentArc = output.sentiment_arc || [];

  return (
    <div style={{ display: "grid", gap: 20 }}>
      {/* Punchlines */}
      {punchlines.length > 0 && (
        <Card>
          <h3 style={{ fontWeight: 700, marginBottom: 16, display: "flex", alignItems: "center", gap: 8 }}>
            🎭 Key Punchlines
          </h3>
          <div style={{ display: "grid", gap: 10 }}>
            {punchlines.map((line, i) => (
              <div key={i} style={{
                padding: "12px 16px",
                background: "var(--surface2)",
                borderRadius: "var(--radius)",
                borderLeft: "3px solid var(--accent3)",
                fontStyle: "italic",
                color: "var(--text)",
                fontSize: 15,
                lineHeight: 1.5,
              }}>
                <span style={{ color: "var(--accent3)", fontStyle: "normal", marginRight: 8 }}>"</span>
                {typeof line === "string" ? line : line.text || JSON.stringify(line)}
                <span style={{ color: "var(--accent3)", fontStyle: "normal", marginLeft: 8 }}>"</span>
                {typeof line === "object" && line.timestamp && (
                  <span style={{ marginLeft: 12, fontSize: 12, color: "var(--muted)", fontFamily: "'Space Mono', monospace", fontStyle: "normal" }}>
                    @ {line.timestamp}
                  </span>
                )}
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Sentiment Arc */}
      {sentimentArc.length > 0 && (
        <Card>
          <h3 style={{ fontWeight: 700, marginBottom: 16 }}>📈 Sentiment Arc</h3>
          <div style={{ display: "flex", alignItems: "flex-end", gap: 4, height: 80 }}>
            {sentimentArc.map((v, i) => {
              const norm = Math.max(0, Math.min(1, (v + 1) / 2)); // -1..1 → 0..1
              const color = norm > 0.6 ? "var(--success)" : norm < 0.4 ? "var(--danger)" : "var(--accent3)";
              return (
                <div key={i} title={`Segment ${i + 1}: ${v.toFixed ? v.toFixed(2) : v}`} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
                  <div style={{
                    width: "100%",
                    height: `${Math.round(norm * 70) + 10}px`,
                    background: color,
                    borderRadius: "3px 3px 0 0",
                    opacity: 0.8,
                    transition: "height 0.3s",
                  }} />
                </div>
              );
            })}
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--muted)", fontFamily: "'Space Mono', monospace", marginTop: 6 }}>
            <span>Start</span><span>End</span>
          </div>
        </Card>
      )}

      {/* Comedians & Topics */}
      <div style={{ display: "grid", gridTemplateColumns: comedians.length && topics.length ? "1fr 1fr" : "1fr", gap: 16 }}>
        {comedians.length > 0 && (
          <Card>
            <h3 style={{ fontWeight: 700, marginBottom: 12 }}>🎤 Comedians / Hosts</h3>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {comedians.map((c, i) => <Tag key={i} label={c} color="var(--accent3)" />)}
            </div>
          </Card>
        )}
        {topics.length > 0 && (
          <Card>
            <h3 style={{ fontWeight: 700, marginBottom: 12 }}>💬 Topics</h3>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {topics.map((t, i) => <Tag key={i} label={t} color="var(--accent2)" />)}
            </div>
          </Card>
        )}
      </div>

      {(!punchlines.length && !comedians.length && !topics.length) && (
        <GenericOutput output={output} />
      )}
    </div>
  );
}

// ── Analytics Dashboard ───────────────────────────────────────────────────

function AnalyticsDashboard() {
  const [summary, setSummary] = useState(null);
  const [recent, setRecent] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      apiGet("/api/v1/analytics/summary"),
      apiGet("/api/v1/analyses/?page_size=10"),
    ])
      .then(([s, r]) => { setSummary(s); setRecent(r.items || []); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return (
    <div style={{ display: "grid", gap: 20 }}>
      {[200, 120, 300].map((h, i) => <div key={i} className="skeleton" style={{ height: h }} />)}
    </div>
  );

  if (error) return (
    <Card>
      <div style={{ color: "var(--danger)", fontFamily: "'Space Mono', monospace", fontSize: 13 }}>⚠ {error}</div>
    </Card>
  );

  const cats = summary?.category_breakdown || {};
  const maxCat = Math.max(1, ...Object.values(cats));

  return (
    <div style={{ animation: "fadeUp 0.5s ease" }}>
      <h2 style={{ fontWeight: 800, fontSize: 22, marginBottom: 28 }}>📊 Analytics Dashboard</h2>

      {/* KPI row */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))", gap: 16, marginBottom: 28 }}>
        {[
          { label: "Total Analyses", value: summary?.total_analyses ?? "—", color: "var(--accent)" },
          { label: "Completed", value: summary?.completed_analyses ?? "—", color: "var(--success)" },
          { label: "Failed", value: summary?.failed_analyses ?? "—", color: "var(--danger)" },
          { label: "Avg Time (s)", value: summary?.avg_processing_time_secs != null ? summary.avg_processing_time_secs.toFixed(1) : "—", color: "var(--accent3)" },
        ].map((kpi, i) => (
          <Card key={i} glow style={{ textAlign: "center" }}>
            <div style={{ fontSize: 32, fontWeight: 800, color: kpi.color, fontFamily: "'Space Mono', monospace" }}>
              {kpi.value}
            </div>
            <div style={{ fontSize: 12, color: "var(--muted)", marginTop: 6, textTransform: "uppercase", letterSpacing: "0.08em" }}>
              {kpi.label}
            </div>
          </Card>
        ))}
      </div>

      {/* Category breakdown */}
      {Object.keys(cats).length > 0 && (
        <Card style={{ marginBottom: 28 }}>
          <h3 style={{ fontWeight: 700, marginBottom: 20 }}>Category Breakdown</h3>
          <div style={{ display: "grid", gap: 12 }}>
            {Object.entries(cats).sort(([, a], [, b]) => b - a).map(([cat, count]) => {
              const meta = CATEGORY_META[cat] || CATEGORY_META.unknown;
              return (
                <div key={cat} style={{ display: "grid", gridTemplateColumns: "130px 1fr 48px", alignItems: "center", gap: 12 }}>
                  <div style={{ fontSize: 13, color: meta.color, fontWeight: 600 }}>{meta.icon} {cat}</div>
                  <div style={{ background: "var(--surface2)", borderRadius: 4, height: 10, overflow: "hidden" }}>
                    <div style={{
                      height: "100%",
                      width: `${(count / maxCat) * 100}%`,
                      background: `linear-gradient(90deg, ${meta.color}, ${meta.color}80)`,
                      borderRadius: 4,
                      transition: "width 0.6s ease",
                    }} />
                  </div>
                  <div style={{ fontSize: 13, color: "var(--muted)", fontFamily: "'Space Mono', monospace", textAlign: "right" }}>{count}</div>
                </div>
              );
            })}
          </div>
        </Card>
      )}

      {/* Recent analyses table */}
      {recent.length > 0 && (
        <Card>
          <h3 style={{ fontWeight: 700, marginBottom: 16 }}>Recent Analyses</h3>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13, fontFamily: "'Space Mono', monospace" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  {["ID", "Status", "Category", "Duration"].map(h => (
                    <th key={h} style={{ padding: "8px 12px", textAlign: "left", color: "var(--muted)", fontWeight: 700, fontSize: 11, textTransform: "uppercase", letterSpacing: "0.08em" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {recent.slice(0, 10).map((row, i) => (
                  <tr key={i} style={{ borderBottom: "1px solid var(--border)", transition: "background 0.15s" }}
                    onMouseEnter={e => e.currentTarget.style.background = "var(--surface2)"}
                    onMouseLeave={e => e.currentTarget.style.background = "transparent"}
                  >
                    <td style={{ padding: "10px 12px", color: "var(--muted)" }}>{(row.analysis_id || row.id || "").slice(0, 12)}…</td>
                    <td style={{ padding: "10px 12px" }}>
                      <Tag label={row.status || "—"} color={row.status === "complete" ? "var(--success)" : row.status === "failed" ? "var(--danger)" : "var(--accent)"} />
                    </td>
                    <td style={{ padding: "10px 12px", color: "var(--text)" }}>{row.category || "—"}</td>
                    <td style={{ padding: "10px 12px", color: "var(--muted)" }}>{row.processing_time_secs != null ? `${row.processing_time_secs.toFixed(1)}s` : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}

// ── Auth Modal ────────────────────────────────────────────────────────────

function AuthModal({ onClose, onAuth }) {
  const [tab, setTab] = useState("login"); // login | register
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    setError("");
    if (!email || !password) { setError("Email and password required"); return; }
    setLoading(true);
    try {
      if (tab === "register") {
        await apiAuthPost("/api/v1/auth/register", { email, password, display_name: displayName });
        setTab("login");
        setError("");
        setPassword("");
        return;
      }
      const data = await apiAuthPost("/api/v1/auth/login", { email, password });
      localStorage.setItem("yt_token", data.access_token);
      onAuth({ email, display_name: data.display_name || email.split("@")[0] });
      onClose();
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const inputStyle = {
    width: "100%", padding: "12px 14px",
    background: "var(--surface2)", border: "1px solid var(--border)",
    borderRadius: "var(--radius)", color: "var(--text)",
    fontSize: 14, fontFamily: "'Space Mono', monospace", outline: "none",
  };

  return (
    <div style={{
      position: "fixed", inset: 0, zIndex: 1000,
      display: "flex", alignItems: "center", justifyContent: "center",
      background: "rgba(8,8,15,0.85)", backdropFilter: "blur(8px)",
    }} onClick={onClose}>
      <div style={{
        background: "var(--surface)", border: "1px solid rgba(0,229,255,0.2)",
        borderRadius: "var(--radius-lg)", padding: 32, width: 400, maxWidth: "90vw",
        boxShadow: "0 0 60px rgba(0,229,255,0.1)",
        animation: "fadeUp 0.3s ease",
      }} onClick={e => e.stopPropagation()}>
        {/* Tab switcher */}
        <div style={{ display: "flex", gap: 4, marginBottom: 24, background: "var(--surface2)", borderRadius: "var(--radius)", padding: 4 }}>
          {["login", "register"].map(t => (
            <button key={t} onClick={() => { setTab(t); setError(""); }} style={{
              flex: 1, padding: "8px", border: "none", borderRadius: "var(--radius)",
              background: tab === t ? "var(--accent)" : "transparent",
              color: tab === t ? "#08080f" : "var(--muted)",
              fontFamily: "'Syne', sans-serif", fontWeight: 700, fontSize: 13,
              cursor: "pointer", transition: "all 0.2s",
            }}>
              {t === "login" ? "Log In" : "Register"}
            </button>
          ))}
        </div>

        <div style={{ display: "grid", gap: 12 }}>
          {tab === "register" && (
            <input
              placeholder="Display name"
              value={displayName}
              onChange={e => setDisplayName(e.target.value)}
              style={inputStyle}
            />
          )}
          <input
            placeholder="Email"
            type="email"
            value={email}
            onChange={e => setEmail(e.target.value)}
            style={inputStyle}
          />
          <input
            placeholder="Password"
            type="password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            onKeyDown={e => e.key === "Enter" && handleSubmit()}
            style={inputStyle}
          />

          {error && (
            <div style={{ color: "var(--danger)", fontSize: 13, fontFamily: "'Space Mono', monospace" }}>
              ⚠ {error}
            </div>
          )}

          <GlowButton onClick={handleSubmit} disabled={loading} style={{ width: "100%", justifyContent: "center" }}>
            {loading ? <Spinner size={18} color="#08080f" /> : (tab === "login" ? "Log In" : "Create Account")}
          </GlowButton>
        </div>

        {tab === "register" && (
          <p style={{ marginTop: 16, textAlign: "center", fontSize: 13, color: "var(--muted)" }}>
            Already have an account?{" "}
            <button onClick={() => setTab("login")} style={{ background: "none", border: "none", color: "var(--accent)", cursor: "pointer", fontSize: 13, fontFamily: "'Syne', sans-serif" }}>Log in</button>
          </p>
        )}
      </div>
    </div>
  );
}

// ── History Panel ──────────────────────────────────────────────────────────

function HistoryPanel({ onSelect }) {
  const [analyses, setAnalyses] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    apiGet("/api/v1/analyses/?page_size=10")
      .then(d => setAnalyses(d.items || []))
      .catch(() => { })
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="skeleton" style={{ height: 160 }} />;
  if (!analyses.length) return null;

  return (
    <div style={{ marginTop: 40 }}>
      <h3 style={{ fontWeight: 700, marginBottom: 16, fontSize: 14, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em", fontFamily: "'Space Mono', monospace" }}>
        Recent Analyses
      </h3>
      <div style={{ display: "grid", gap: 8 }}>
        {analyses.map(a => (
          <button key={a.analysis_id} onClick={() => onSelect(a.analysis_id)} style={{
            padding: "12px 16px", background: "var(--surface)",
            border: "1px solid var(--border)", borderRadius: "var(--radius)",
            cursor: "pointer", textAlign: "left", width: "100%",
            display: "flex", alignItems: "center", justifyContent: "space-between",
            transition: "border-color 0.2s",
          }}
            onMouseEnter={e => e.currentTarget.style.borderColor = "var(--accent)"}
            onMouseLeave={e => e.currentTarget.style.borderColor = "var(--border)"}
          >
            <span style={{ fontFamily: "'Space Mono', monospace", fontSize: 13, color: "var(--muted)" }}>
              {a.analysis_id.slice(0, 12)}…
            </span>
            <Tag
              label={a.status}
              color={a.status === "complete" ? "var(--success)" : a.status === "failed" ? "var(--danger)" : "var(--accent)"}
            />
          </button>
        ))}
      </div>
    </div>
  );
}

// ── Stats / Feature Highlights ─────────────────────────────────────────────

function FeatureGrid() {
  const features = [
    {
      icon: "🧠",
      title: "Multi-Modal AI",
      desc: "Combines EfficientNet vision, BERT text, and Whisper audio for 85%+ classification accuracy.",
    },
    {
      icon: "📊",
      title: "Structured Extraction",
      desc: "Ranked lists from listicle videos, track metadata from music compilations, chapters from tutorials.",
    },
    {
      icon: "🎵",
      title: "Spotify Integration",
      desc: "Auto-generate playable Spotify playlists from music compilation videos in one click.",
    },
    {
      icon: "🎬",
      title: "TMDb Enrichment",
      desc: "Movie/TV metadata, ratings, streaming availability from Netflix, Prime, Disney+ and more.",
    },
    {
      icon: "📝",
      title: "Whisper Transcription",
      desc: "Accurate speech-to-text in 20+ languages with speaker-aware timestamped segments.",
    },
    {
      icon: "⚡",
      title: "Async Pipeline",
      desc: "Celery + RabbitMQ task queue ensures sub-3-minute analysis for 10-minute videos.",
    },
  ];

  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))",
      gap: 16, marginTop: 64,
    }}>
      {features.map((f, i) => (
        <Card key={i} style={{ animation: `fadeUp 0.5s ease ${i * 0.07}s both` }}>
          <div style={{ fontSize: 28, marginBottom: 10 }}>{f.icon}</div>
          <h4 style={{ fontWeight: 700, marginBottom: 6 }}>{f.title}</h4>
          <p style={{ color: "var(--muted)", fontSize: 14, lineHeight: 1.6 }}>{f.desc}</p>
        </Card>
      ))}
    </div>
  );
}

// ── Main App ───────────────────────────────────────────────────────────────

export default function App() {
  const [phase, setPhase] = useState("idle"); // idle | processing | done | analytics
  const [analysisId, setAnalysisId] = useState(null);
  const [jobStatus, setJobStatus] = useState("queued");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [user, setUser] = useState(() => {
    try {
      const raw = localStorage.getItem("yt_user");
      return raw ? JSON.parse(raw) : null;
    } catch { return null; }
  });
  const [showAuth, setShowAuth] = useState(false);
  const pollRef = useRef(null);

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  const startPolling = useCallback((id) => {
    stopPolling();
    pollRef.current = setInterval(async () => {
      try {
        const status = await apiGet(`/api/v1/analyses/${id}/status`);
        setJobStatus(status.status);

        if (status.status === "complete") {
          stopPolling();
          const res = await apiGet(`/api/v1/analyses/${id}/result`);
          setResult(res);
          setPhase("done");
        } else if (status.status === "failed") {
          stopPolling();
          setError(status.error_message || "Analysis failed");
          setPhase("idle");
        }
      } catch (e) {
        console.warn("Poll error:", e);
      }
    }, 5000);
  }, []);

  const handleSubmit = async (url) => {
    setError(null);
    setPhase("processing");
    setJobStatus("queued");

    const job = await apiPost("/api/v1/analyses/", { url, force_reanalysis: false });
    const id = job.analysis_id;
    setAnalysisId(id);

    if (job.status === "complete") {
      const res = await apiGet(`/api/v1/analyses/${id}/result`);
      setResult(res);
      setPhase("done");
    } else {
      startPolling(id);
    }
  };

  const handleReset = () => {
    stopPolling();
    setPhase("idle");
    setAnalysisId(null);
    setResult(null);
    setError(null);
    setJobStatus("queued");
  };

  const handleSelectExisting = async (id) => {
    try {
      const res = await apiGet(`/api/v1/analyses/${id}/result`);
      setResult(res);
      setAnalysisId(id);
      setPhase("done");
    } catch (e) {
      setError(e.message);
    }
  };

  const handleAuth = (userData) => {
    setUser(userData);
    localStorage.setItem("yt_user", JSON.stringify(userData));
  };

  const handleLogout = () => {
    localStorage.removeItem("yt_token");
    localStorage.removeItem("yt_user");
    setUser(null);
  };

  useEffect(() => () => stopPolling(), []);

  return (
    <>
      <style>{GLOBAL_STYLES}</style>

      {/* Background glow orbs */}
      <div style={{ position: "fixed", inset: 0, pointerEvents: "none", overflow: "hidden", zIndex: 0 }}>
        <div style={{
          position: "absolute", width: 600, height: 600,
          borderRadius: "50%", top: -200, left: "20%",
          background: "radial-gradient(circle, rgba(0,229,255,0.06) 0%, transparent 70%)",
        }} />
        <div style={{
          position: "absolute", width: 500, height: 500,
          borderRadius: "50%", bottom: 0, right: "10%",
          background: "radial-gradient(circle, rgba(124,58,237,0.06) 0%, transparent 70%)",
        }} />
      </div>

      <div style={{ position: "relative", zIndex: 1 }}>
        {/* Nav */}
        <nav style={{
          display: "flex", alignItems: "center", justifyContent: "space-between",
          padding: "16px 40px",
          borderBottom: "1px solid var(--border)",
          backdropFilter: "blur(12px)",
          position: "sticky", top: 0, zIndex: 100,
          background: "rgba(8,8,15,0.8)",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <button onClick={handleReset} style={{
              display: "flex", alignItems: "center", gap: 10,
              background: "none", border: "none", cursor: "pointer", padding: 0,
            }}>
              <div style={{
                width: 32, height: 32, borderRadius: 8,
                background: "linear-gradient(135deg, var(--accent), var(--accent2))",
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 16,
              }}>▶</div>
              <span style={{ fontWeight: 800, fontSize: 16, letterSpacing: "-0.02em", color: "var(--text)" }}>
                YT Classifier
              </span>
            </button>
            <Tag label="AI" color="var(--accent)" />
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 20, fontSize: 14 }}>
            <button
              onClick={() => setPhase(p => p === "analytics" ? "idle" : "analytics")}
              style={{
                background: "none", border: "none",
                color: phase === "analytics" ? "var(--accent)" : "var(--muted)",
                cursor: "pointer", fontFamily: "'Syne', sans-serif", fontSize: 14, fontWeight: phase === "analytics" ? 700 : 400,
                transition: "color 0.2s",
              }}
            >
              📊 Analytics
            </button>
            <a href="/api/docs" target="_blank" rel="noopener noreferrer" style={{ color: "var(--muted)", textDecoration: "none" }}>API Docs</a>
            <a href="https://github.com" target="_blank" rel="noopener noreferrer" style={{ color: "var(--muted)", textDecoration: "none" }}>GitHub</a>
            {user ? (
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <span style={{ fontSize: 13, color: "var(--accent)", fontFamily: "'Space Mono', monospace" }}>
                  {user.display_name || user.email}
                </span>
                <GlowButton variant="secondary" style={{ padding: "6px 14px", fontSize: 12 }} onClick={handleLogout}>
                  Log out
                </GlowButton>
              </div>
            ) : (
              <GlowButton style={{ padding: "8px 18px", fontSize: 13 }} onClick={() => setShowAuth(true)}>
                Log in
              </GlowButton>
            )}
          </div>
        </nav>

        {/* Main content */}
        <main style={{ maxWidth: 900, margin: "0 auto", padding: "64px 24px 120px" }}>
          {phase === "analytics" && <AnalyticsDashboard />}

          {phase === "idle" && (
            <>
              <HeroInput onSubmit={handleSubmit} loading={false} />
              {error && (
                <div style={{
                  marginTop: 20, padding: 16, borderRadius: "var(--radius)",
                  background: "rgba(255,77,109,0.1)", border: "1px solid rgba(255,77,109,0.3)",
                  color: "var(--danger)", fontFamily: "'Space Mono', monospace", fontSize: 13,
                }}>
                  ⚠ {error}
                </div>
              )}
              <FeatureGrid />
              <HistoryPanel onSelect={handleSelectExisting} />
            </>
          )}

          {phase === "processing" && (
            <ProcessingStatus status={jobStatus} analysisId={analysisId} />
          )}

          {phase === "done" && result && (
            <ResultView result={result} onReset={handleReset} />
          )}


        </main>

        {/* Footer */}
        <footer style={{
          textAlign: "center", padding: "32px",
          borderTop: "1px solid var(--border)",
          color: "var(--muted)", fontSize: 13,
          fontFamily: "'Space Mono', monospace",
        }}>
          Built with FastAPI · Celery · Whisper · EfficientNet · BERT · spaCy · TMDb · Spotify
        </footer>
      </div>

      {showAuth && <AuthModal onClose={() => setShowAuth(false)} onAuth={handleAuth} />}
    </>
  );
}
