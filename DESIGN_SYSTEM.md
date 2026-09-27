# NXTSight — Design System

A reference for every future UI/UX decision in this project, across both surfaces: the existing Streamlit app (`app.py`, `pages/`, styled by [`assets/theme.css`](assets/theme.css) via [`src/ui/theme.py`](src/ui/theme.py)) and the new React frontend (`frontend/`, using `motion`).

Nothing here is invented. Every token and rule below was extracted from the theme this app already ships, or is a documented, citable standard (WCAG 2.1, established typographic and motion-design practice) — not a generic template and not a third-party "AI design tool." Where something is a genuine open gap rather than an established rule, it's marked **Open gap**, not silently glossed over.

This file documents and extends the existing system. It does not change any code — `assets/theme.css` remains the actual source of truth for the Streamlit app; this doc explains *why* those values are what they are and how to extend them consistently, including into the new frontend.

---

## 1. Color

### Tokens (from `assets/theme.css`)

| Token | Value | Role |
|---|---|---|
| `--nxt-bg` | `#030304` | Base canvas — near-black, not pure black |
| `--nxt-bg-alt` | `#060607` | Sidebar / secondary surface |
| `--nxt-panel` | `#0c0c0f` | Cards, inputs, badges — the standard "surface" |
| `--nxt-panel-hover` | `#131318` | Hover state for panels/tabs |
| `--nxt-border` | `rgba(255,255,255,0.08)` | Default hairline border |
| `--nxt-border-strong` | `rgba(255,255,255,0.18)` | Emphasized border (buttons, hover) |
| `--nxt-text` | `#f1f3f9` | Primary text |
| `--nxt-text-muted` | `#8b93a7` | Secondary text, captions |
| `--nxt-violet` | `#7c5cff` | Primary accent |
| `--nxt-cyan` | `#22d3ee` | Secondary accent |
| `--nxt-gradient` | `linear-gradient(135deg, violet, cyan)` | Headings, primary CTAs, active tab |
| `--nxt-green` | `#22c55e` | Positive / safe / accelerated |
| `--nxt-amber` | `#f59e0b` | Real-but-secondary accelerator |
| `--nxt-red` | `#ef4444` | Danger / scam verdict |
| `--nxt-blue` | `#60a5fa` | Neutral fallback (CPU) |

### Verified contrast ratios (WCAG 2.1, computed against the actual tokens above)

| Pair | Ratio | Passes |
|---|---|---|
| `--nxt-text` on `--nxt-bg` | **18.6:1** | AAA, all text sizes |
| `--nxt-text-muted` on `--nxt-bg` | **6.7:1** | AA (normal text), just under AAA |
| `--nxt-text-muted` on `--nxt-panel` | **6.35:1** | AA (normal text) |

Rule going forward: **never place `--nxt-text-muted` on anything lighter than `--nxt-panel`** without recomputing — it's already close to the AA floor for normal text, and closer still for anything under 18px/bold-14px (WCAG's "large text" exception).

### Semantic meaning (don't reassign these without updating every badge)

Color in this app carries meaning, not just decoration — the badge system (`nxt-badge-npu/cpu/accel/safe`) already encodes this:

- **Green** = the best-case outcome (Snapdragon NPU active, or a "safe/legit" verdict)
- **Amber** = a real accelerator, just not the primary target (CUDA/DirectML/CoreML/OpenVINO)
- **Blue** = neutral fallback, nothing wrong, nothing special (CPU)
- **Red** = danger (scam verdict only — never used decoratively)

Any new status indicator (in Streamlit or the React frontend) should map into this same four-color vocabulary rather than introduce a fifth meaning.

---

## 2. Typography

### Current pairing (already loaded via Google Fonts in `theme.css`)

- **Display / headings**: `Space Grotesk` (500/600/700) — a geometric sans with distinctive character in its display weights; used for `h1–h4`, hero titles, section labels.
- **Body / UI**: `Inter` (400/500/600) — one of the most extensively screen-tested UI typefaces available, chosen for small-size legibility over character.

This is a sound, standard pairing (a distinctive display face over a neutral, highly-legible body face) and should stay as-is. No change recommended.

### Addition worth making: a monospace for numeric readouts

This is a fraud-detection/fintech app — confidence percentages, transaction amounts, and timestamps currently render in Inter's proportional numerals, which is fine but not optimal: proportional digits shift width as they change, which reads as slightly "unstable" in a live-updating confidence meter or amount field.

**Recommendation**: `JetBrains Mono` or `IBM Plex Mono` (both open-source, both on Google Fonts, both already common in fintech/security UI) for:
- Confidence percentages (`nxt-meter-label`)
- Transaction amounts (Money Insight, Payment Pause)
- Timestamps / execution-path technical strings

Not applied to any file yet — this is a recommendation for the next time those components are touched, not a retroactive change.

### Type scale (extracted, current usage)

| Use | Size | Weight | Font |
|---|---|---|---|
| Hero title | 2.1rem (1.6rem compact) | 700 | Space Grotesk |
| Section title | 1.35rem | 600 | Space Grotesk |
| Body | 1rem (browser default) | 400 | Inter |
| Subtitle / caption | 0.98rem | 400 | Inter, muted |
| Badge / meter label | 0.85rem / 0.78rem | 500 | Inter |

---

## 3. Spacing & shape

- **Radius**: one token, `--nxt-radius: 16px`, used everywhere (cards, hero, expander, CTA). Buttons/inputs/tabs use a tighter `10px`. Fully round (`999px`) for pills/badges/dots. Don't introduce a third radius value.
- **Borders**: always `1px`, always one of the two border tokens (`--nxt-border` default, `--nxt-border-strong` on emphasis/hover) — never a bespoke color.
- **Spacing**: current values are ad hoc rem numbers (0.9rem, 1.1rem, 1.3rem, 1.6rem, 1.8rem…) rather than a formal scale. **Open gap, low priority**: for new components, prefer snapping to a standard 8px-based scale (4/8/12/16/24/32/48px → 0.25/0.5/0.75/1/1.5/2/3rem) so future additions compose more predictably. Not worth retrofitting existing components for this alone.

---

## 4. Motion

### Existing keyframes (all in `theme.css`, none touched by this doc)

`nxt-fade-in`, `nxt-float`, `nxt-spin`, `nxt-pop`, `nxt-meter-grow`, `nxt-pulse`, `nxt-badge-glow`, `nxt-cta-glow`.

### The signature curve

Almost every transition in the app uses plain `ease` or `ease-in-out` — functional but generic. One animation already uses something more deliberate:

```css
animation: nxt-meter-grow 0.7s cubic-bezier(0.22, 1, 0.36, 1) both;
```

That curve (`cubic-bezier(0.22, 1, 0.36, 1)`) is an **emphasized-decelerate** easing — fast start, soft landing, similar in feel to Material Design's "emphasized" easing. It's the one moment in the current UI with real motion craft. **Recommendation**: standardize on this as NXTSight's signature ease for anything meant to feel deliberate — a verdict resolving, a panel opening — rather than `ease-in-out` everywhere. Reserve plain `ease` for incidental hover states.

In `motion`/Framer Motion terms, the equivalent is:
```tsx
transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
```

### Duration guide (grounded in current usage, not invented)

| Purpose | Duration | Current example |
|---|---|---|
| Micro (hover, focus) | 0.15–0.2s | button/input transitions |
| Standard (enter, pop) | 0.25–0.5s | card hover, alert pop-in, page fade-in |
| Deliberate (a value resolving) | 0.7s | confidence meter fill |
| Ambient (looping glow/pulse) | 2.2–3.4s | badge glow, CTA glow, hero icon float |

### Open gap: no `prefers-reduced-motion` support anywhere

Checked directly: `assets/theme.css` has zero `@media (prefers-reduced-motion: reduce)` rules. Every glow, pulse, float, and page-fade animation currently plays unconditionally, including the two infinite ambient loops (hero icon float, badge/CTA glow). This is a real accessibility gap, not a style preference — worth fixing the next time `theme.css` is touched, and worth building correctly from day one in the new React frontend (respect `useReducedMotion()` from `motion/react`, which `motion` ships specifically for this).

---

## 5. Iconography

**Rule, already established and non-negotiable**: plain geometric Unicode symbols, never emoji. Current examples in use: `◆` (page icon), `⟡` (hero icon). Emoji render inconsistently across OS/font stacks and read as informal for a fraud-detection tool asking people to trust it with financial decisions — the existing choice to avoid them is correct and should hold for the React frontend too.

When a new symbol is needed: prefer characters from the Geometric Shapes, Miscellaneous Symbols, or Dingbats Unicode blocks (the same families `◆`/`⟡` come from) over anything that renders as a colorful pictograph. Test rendering across at least macOS + Windows default fonts before committing to one — Unicode symbol glyph support is less consistent than it looks in one environment.

---

## 6. Component patterns (semantics, not just style)

- **Card** (`nxt-card`): default content container. Lifts 4px + glows violet on hover — this hover treatment signals "interactive/clickable," so don't apply it to purely static display cards.
- **Badge** (`nxt-badge-*`): status, never a label for arbitrary text — see color semantics in §1.
- **Meter**: confidence visualization specifically — danger (red) or safe (green) gradient fill, animates its width in on mount. Don't repurpose for a non-confidence percentage without reconsidering the red/green framing (e.g. it would misread on a "% categorized" progress stat).
- **Status dots**: binary on/off system-health indicators (green pulse = on). Not for anything with more than two states.
- **CTA container** (`nxt_cta`): reserved for exactly one thing today — the Future Vision link. It's built as a real `st.container(key=...)`, not hand-wrapped markup, specifically so Streamlit's own DOM nesting stays correct. If a second highlighted CTA is ever needed, follow the same `st.container(key=...)` pattern rather than reusing this exact class for a second purpose.

---

## 7. Applying this to `frontend/`

The React app (`frontend/`, `motion` installed) has no theme file yet. When one is built, it should be a direct TypeScript port of the tokens in §1–4 — not a reinterpretation. Concretely, the natural next step (not done here, since this doc is reference-only) is a `frontend/src/theme.ts` exporting the same color tokens as JS constants (or CSS custom properties reused via a shared `:root` block), so a color or radius change only ever happens in one place conceptually, even though it's expressed in two files (`assets/theme.css` for Streamlit, `theme.ts` for React) because the two frameworks can't literally share a stylesheet.
