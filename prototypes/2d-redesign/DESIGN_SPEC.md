# NeuroDrive design specification

The design system behind the prototype. Every value here is in
`src/theme/tokens.css`; the prototype's components use the token names, never raw values. Where a
rule exists for a reason, the reason is given, so a later change can be judged against it.

Contents: 1 Philosophy · 2 Colour · 3 Typography · 4 Space, radius, elevation · 5 Layout and
navigation · 6 Components and their states · 7 Track and cars · 8 Charts · 9 Network diagram ·
10 Themes · 11 Motion · 12 Small windows · 13 Accessibility · 14 Directions that were rejected

---

## 1. Philosophy

NeuroDrive is three things at once: a racing simulator, a research instrument and a place where
experiments are analysed. The design gives each its own character and makes them feel like rooms
of one building.

| View | Character | What that means in practice |
|---|---|---|
| Environment | A race seen from above, with a timing screen | The track is a place: asphalt, kerbs, grass. Everything else floats over it on dark instrument panels. |
| Network | A measuring instrument | A quiet surface, one diagram, exact numbers on demand, the explanation beside the thing explained. |
| Experiment | A lab notebook with good charts | Panels on a grid, every chart labelled with what it measures, tables under the pictures. |

Six rules hold everywhere.

1. **One colour, one meaning.** A hue is given one job for the whole app (§2.1). The current app's
   blue is the selection, the primary button, a positive weight and a positive sensitivity; here
   those are three different colours.
2. **Decoration never sits above data.** The track may be as rich as a game's, but rays, trails,
   cars and labels are always drawn over it, and nothing decorative is drawn where a car can be.
3. **The truth line.** Whatever looks like the edge of the road *is* the edge the simulation uses.
   Kerbs and sand are outside it.
4. **Numbers look like numbers.** Every figure is in the tabular number face, so columns align and
   a value does not jitter as its digits change.
5. **Nothing important is hidden.** No tabs inside the inspector, no information that exists only
   in a tooltip. Tooltips add precision; they are never the only place something is said.
6. **Python decides, the browser draws.** No part of the design asks the browser to compute a
   network output, a physics step or a statistic. Where the prototype shows something Python does
   not send yet, it is marked "proposed" and listed in `PRODUCTION_MAPPING.md` §6.

---

## 2. Colour

### 2.1 What each colour family means

| Family | Meaning | Used for |
|---|---|---|
| **Violet** ("sector purple") | *This is selected / this is where you are looking* | Selected car's reticle and name tag, selected neuron, pressed controls, primary button, chart cursor, the look-back moment, the what-if ghost, the plain trail, the ancestry line |
| **Blue ↔ orange** | *Sign: positive ↔ negative* | Weights, neuron values, contributions, sensitivity, steer right ↔ left, throttle ↔ brake, the trail and rays when an output is selected, a mutation raising ↔ lowering a weight |
| **Green / amber / red / grey** | *How a drive ended, or how close a wall is* | Finished / stalled / crashed / out of time; rays: clear / near / danger / no hit |
| **Blue ramp, one hue** | *How much, or which rank* | Rank band of a car, share of a generation in a heat-map cell |
| **Gold** | *Elite* | The ring round an elite car. Nothing else. |
| **Eight series hues** | *Which group* | Lines of a chart with several groups, in a fixed order |
| **Neutrals** | Everything else | Surfaces, text, borders, grids |

Why violet for selection: it is the one strong hue not already spoken for (blue and orange are
sign, green/amber/red are status), it stands out on asphalt, on grass and on both page themes, and
in motor-racing timing purple marks the fastest of all, so "the one that matters" already reads as
purple to anyone who has watched a timing screen.

The blue of "positive" and the blue of the rank ramp are deliberately one family: both say "more
of the good thing". They never meet on one mark: a car's body carries rank, a line carries sign.

### 2.2 Page colours

| Token | Light | Dark | Role |
|---|---|---|---|
| `--bg` | `#eceef2` | `#0b0d11` | Page behind the panels |
| `--surface` | `#ffffff` | `#14171d` | Panels, the chart surface |
| `--surface-2` | `#f4f5f8` | `#1a1e26` | Wells, banners, the inspector rail |
| `--surface-3` | `#e8ebf0` | `#232834` | Hover, bar tracks, segmented control |
| `--border` | `#d9dde4` | `#2a303b` | Hairlines |
| `--border-strong` | `#bcc3ce` | `#3d4554` | Control outlines, node rings |
| `--text` | `#14171c` (18.0:1) | `#e9ecf1` (15.2:1) | Primary text |
| `--text-2` | `#4a5261` (7.9:1) | `#a9b1bf` (8.3:1) | Secondary text |
| `--text-3` | `#6f7888` (4.5:1) | `#7d8797` (4.9:1) | Labels, ticks, notes |
| `--accent` | `#6b46e5` (5.8:1) | `#a790ff` (6.9:1) | Selection and focus |
| `--accent-hover` | `#5a37cf` | `#b9a6ff` | Primary button, hovered |
| `--accent-soft` | `rgba(107,70,229,.11)` | `rgba(167,144,255,.16)` | Selected row, halo, cursor column |
| `--on-accent` | `#ffffff` | `#14101f` | Text on the accent |
| `--pos` | `#2f6fe4` (4.7:1) | `#4f86ee` (5.1:1) | Positive |
| `--neg` | `#c8590f` (4.3:1) | `#d9742a` (5.5:1) | Negative |
| `--mid` | `#d9dde4` | `#3a4150` | The neutral middle of a diverging scale |
| `--ok` / `--warn` / `--bad` / `--idle` | `#178a3e` / `#a86a00` / `#c0392b` / `#6f7888` | `#3ecf74` / `#f5b83d` / `#ff6b5e` / `#8a93a3` | Status as text and icons (all ≥ 4.4:1) |
| `--ok-fill` / `--warn-fill` / `--bad-fill` / `--idle-fill` | `#3ccf91` / `#e9a400` / `#a82a22` / `#8a93a3` | `#2cac7d` / `#c08a12` / `#bb3b31` / `#69738a` | Status as filled cells and bars |
| `--grid` | `#e6e9ee` | `#232834` | Chart gridlines |
| `--axis` | `#bcc3ce` | `#3d4554` | Chart baselines |

Contrast ratios are against `--surface` of the same theme.

**Checked, not eyeballed.** The palette was run through the dataviz skill's validator
(`validate_palette.js`), the same tool the current app's palette was checked with:

- `--pos` / `--neg`: pass every check in both themes (colour-blind separation ΔE 27 or more).
- **Status fills: changed because of the check.** A first choice of green and red failed
  (ΔE 3.6 for red-green colour blindness: the outcome grid would have been unreadable). The final
  set separates finished from crashed by lightness as well as hue (finished is light, crashed is
  dark) and passes all pairs in both themes (worst pair ΔE 11.4 light, 10.3 dark). The grey is
  below the validator's chroma floor on purpose: "out of time" is the neutral outcome. The current
  app's `--ok` / `--bad` cell colours have the same weakness the first choice had.
- Status colours never appear alone: chips carry an icon and a word, track badges carry a glyph,
  the outcome grid has a legend, a tooltip and a table.

### 2.3 Series and ramps

Kept from the current app (it already uses the dataviz reference palette) and re-validated on the
new surfaces.

| Token | Light (on `#ffffff`) | Dark (on `#14171d`) |
|---|---|---|
| `--series-1` … `--series-8` | `#2a78d6` `#eb6834` `#1baf7a` `#eda100` `#e87ba4` `#008300` `#6250d6` `#e34948` | `#3987e5` `#d95926` `#199e70` `#c98500` `#d55181` `#008300` `#9085e9` `#e66767` |
| `--rank-1` … `--rank-5` (1 = most) | `#0d366b` `#184f95` `#256abf` `#3987e5` `#6da7ec` | `#cde2fb` `#9ec5f4` `#6da7ec` `#3987e5` `#256abf` |
| `--rank-0` (none) | `#e3ecf9` | `#1c2433` |

Rules: series colours are given in this fixed order and never cycled; a ninth group is not drawn
in a new colour. In light mode series 3, 4 and 5 are below 3:1 against white, so every chart with
them also has labels at the line ends and a table. The ramp's "most" end is always the one that
stands out from the surface, which is why the two columns run in opposite directions.

### 2.4 Colours that do not change with the theme

The track is dark asphalt by day and by night, so everything drawn on it, and everything that
floats over it, uses one set of values in both themes.

| Token | Value | Role |
|---|---|---|
| `--chrome-bg` / `--chrome-raised` / `--chrome-border` | `#12151b` / `#1c2029` / `#2a303b` | The top bar |
| `--chrome-text` / `--chrome-muted` | `#e9ecf1` (15.4:1) / `#8e97a6` (6.2:1) | Text on the top bar |
| `--hud-bg` | `rgba(15,18,24,.90)` | Floating panels over the track |
| `--hud-border` / `--hud-track` | `rgba(255,255,255,.12)` / `rgba(255,255,255,.14)` | Their outline; gauge tracks |
| `--hud-text` / `--hud-muted` | `#f1f3f6` (16.9:1) / `#a3acba` (8.2:1) | Text on them |
| `--car-rank-1` … `--car-rank-5` | `#cde2fb` `#9ec5f4` `#6da7ec` `#3987e5` `#256abf` | Car bodies by rank band, best first |
| `--car-elite` | `#f2c230` | Elite ring |
| `--ray-clear` / `--ray-near` / `--ray-danger` / `--ray-none` | `#4ad17f` / `#f5b83d` / `#ff6b5e` / `rgba(255,255,255,.34)` | Rays by distance: over half the range / under half / under a fifth / no hit |
| `--track-pos` / `--track-neg` | `#6aa2ff` / `#ff9a4a` | The sign pair on asphalt |
| `--track-select` | `#b9a6ff` | The selection colour on asphalt |

On-track colours are brighter than their page versions because they are thin lines on a textured
mid-dark surface. `--track-pos` / `--track-neg` are above the validator's lightness band for dark
surfaces for that reason; their contrast on day asphalt is 4.3:1 and 5.3:1 and their colour-blind
separation is ΔE 25. The rank ramp passes the validator's ordinal check on both asphalts (its
darkest step is 2.05:1 on day asphalt, 2.74:1 at night), and every car also has an outline.

### 2.5 The world

| Token | Day (light theme) | Night (dark theme) | Role |
|---|---|---|---|
| `--w-grass` / `--w-grass-2` | `#5c8650` / `#557c4a` | `#1b2a1f` / `#18261c` | Grass and its mowing stripes |
| `--w-verge` | `#729c62` | `#243828` | The mown band beside the road |
| `--w-sand` | `#c9ba8c` | `#4a4535` | Sand traps |
| `--w-asphalt` / `--w-asphalt-2` | `#383c43` / `#30343a` | `#25282e` / `#1e2126` | Road; the dark lip under its edge |
| `--w-line` | `#f3f4f1` (10.0:1 on asphalt) | `#c9ccc8` (9.1:1) | Edge lines, start checks |
| `--w-kerb-a` / `--w-kerb-b` | `#d8443b` / `#f3f4f1` | `#a8352f` / `#c9ccc8` | Kerb blocks |
| `--w-marking` | `rgba(255,255,255,.30)` | `rgba(255,255,255,.20)` | Centre dashes, arrows |
| `--w-shadow` | `rgba(8,12,10,.34)` | `rgba(0,0,0,.50)` | Car shadows |
| `--w-board` / `--w-board-ink` | `#f3f4f1` / `#1b1f27` | `#c9ccc8` / `#14171c` | Distance boards |
| `--w-vignette` | `rgba(10,24,12,.22)` | `rgba(0,0,0,.40)` | Corner darkening of the stage |

Grass is kept low in saturation so it never competes with the green of a clear ray. Rays only
ever lie on asphalt (they end at the road's edge), so green on green cannot happen.

---

## 3. Typography

One family in three cuts, bundled with the app (no network request): **IBM Plex**. It was drawn
for engineering documents and instrument panels, it has a condensed cut for labels and a mono cut
whose digits match the sans, and it does not look like a default.

| Token | Face | Used for |
|---|---|---|
| `--font-ui` | IBM Plex Sans 400 / 500 / 600 | Everything not listed below |
| `--font-label` | IBM Plex Sans Condensed 500 / 600 | Small capital labels, chips, axis labels, the brand, track name tags |
| `--font-num` | IBM Plex Mono 400 / 500, tabular figures | Every number: telemetry, table cells, ticks, values in the diagram, ray labels |

| Token | Size | Used for |
|---|---|---|
| `--fs-micro` | 10.5 px | Capital labels ("SPEED", "HIDDEN 1"), 0.09 em letter-spacing, weight 600 |
| `--fs-small` | 12 px | Tables, legends, notes, dense controls |
| `--fs-body` | 13 px | Default text, line height 1.45 |
| `--fs-title` | 15 px | View titles, weight 600 |
| `--fs-figure` | 20 px | Stat figures |
| `--fs-hero` | 30 px | The one large number of a panel (speed) |

Rules:
- A label is either a sentence-case phrase in `--font-ui` or a short capital label in
  `--font-label`. Capital labels are for names of quantities, never for sentences.
- A number is always in `--font-num`. Its unit follows in `--font-label` at 11 px, muted.
- Minus signs are the real minus (−), not a hyphen. A signed quantity shows its + as well.
- Text never takes a data colour. A coloured mark beside it carries identity.

---

## 4. Space, radius, elevation

| Token | Value | Used for |
|---|---|---|
| `--sp-1` … `--sp-6` | 4, 8, 12, 16, 24, 32 px | All gaps and paddings. Inside a control 4 or 8; between controls 8 or 12; panel padding 12; between panels 12; page padding 8. |
| `--r-1` | 4 px | Chips with corners, inputs, cells, bars |
| `--r-2` | 8 px | Buttons, panels, banners |
| `--r-3` | 12 px | Views, the drawer, the telemetry strip |
| pill | 999 px | Status chips, the connection badge |
| `--shadow-1` | light `0 1px 2px rgba(16,20,28,.05)`; dark `0 1px 2px rgba(0,0,0,.4)` | Panels at rest |
| `--shadow-2` | light `0 8px 24px rgba(16,20,28,.16), 0 1px 3px rgba(16,20,28,.10)`; dark `0 10px 28px rgba(0,0,0,.55), 0 1px 3px rgba(0,0,0,.5)` | Anything that floats: HUD panels, tooltips, the sticky generation bar |

There are two levels only: resting (a hairline border, almost no shadow) and floating. No blur, no
glass, no glow, no gradient fills on interface elements. The only gradients in the app are on the
canvas: the stage vignette, the fade-in of a ray near its car, and a headlight cone at night.

---

## 5. Layout and navigation

### 5.1 The shell

```
┌ top bar, 48 px, dark in both themes ─────────────────────────────────────────────┐
│ ▮▮▮ NEURODRIVE │ Environment  Network  Experiment │ Run › Generation › Start ›   │
│                │ (tabs, 3 px violet underline)    │ Car › Neuron › Moment        │
│                                                   │ Notes · Page|Split · ☀☾◐ · ● │
├──────────────────────────────────────────────────────────────────────────────────┤
│ ┌ view ──────────────────────────────────────────────┐ ┌ drawer, 340 px ───────┐ │
│ │ header 44 px: Title  the view's question  controls │ │ linked view, compact  │ │
│ │ body                                               │ │                       │ │
│ └────────────────────────────────────────────────────┘ └───────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────┘
```

- **Top bar.** Dark in both themes: it frames the app like an instrument's bezel and gives the
  light theme a fixed anchor. The brand mark is three slanted blocks (red, white, red: a kerb),
  the only place the kerb motif appears outside the track.
- **View tabs** are links (`#/environment`, `#/network`, `#/experiment`), as today.
- **The breadcrumb** shows the shared selection in the order of the concept document: Run,
  Generation (or Checkpoint), Start, Car, Neuron, Moment. Each part is a button that goes to the
  view where that part is chosen. An unset part reads "none" or "now" in the muted colour.
- **Each view's header** carries its title and its question ("What does the car do?", "Why did it
  do that?", "How did it learn to?"), then the controls that set the view up.
- **The drawer** is a panel of its own with a header. On the Environment page it has two tabs,
  Network and Standings; on the other pages it shows the track following the selected car.
  Collapsed, it is a 36 px rail with a vertical label.
- **Layouts.** Page (view + drawer) and Split (two views, 1.15 : 1). The current app's All layout
  and resizable panels are kept in production; a view adapts to its own width (§12), so it no
  longer matters which slot it is in.

### 5.2 Environment view

```
header:  Environment · question      Track ▾  Driver ▾  Generation  Start  Ghost
┌ stage (the canvas fills it) ───────────────────────────────────────────────────┐
│ [▶ Play ⏭ ↺ | 0.5× 1× 2× 4× 8×]                    [rays trail | Follow Whole − +]
│                                                                     [ minimap ] │
│                               the track                                         │
│ [status banner, when the selected car has stopped]                              │
│ [legend: rank ramp · elite · ghost · stopped · (sensitivity) · (what-if)]       │
│ [telemetry: car | speed | steer, throttle | lap | rays | fitness | step]        │
└─────────────────────────────────────────────────────────────────────────────────┘
look-back strip: steer and throttle over the last 60 s, with the scrub cursor
```

- The scenario is in the header because it is set once; transport is on the stage because it is
  used constantly.
- Whole-track view keeps 70 px free at the top and 138 px at the bottom, so the floating panels
  never cover the circuit.
- The long instruction paragraph is gone. Each instruction is a tooltip on the control it is
  about, plus one hint in the look-back strip.

### 5.3 Network view

```
header:  Network · question     ‹ Car picker ›     Lines show: Weights | Flow | Mutation | Change
┌ main (scrolls) ──────────────────────────────────┬ inspector rail, 372 px (scrolls) ─┐
│ lineage banner                                   │ [ Steer card ] [ Throttle card ]  │
│ (past banner, when looking back)                 │ headline: Steer = −0.08 · words   │
│ diagram                                          │ WHY        receipt + z + curve    │
│ read-out line                                    │ SENSITIVITY  seven bars + sentence│
│ legend                                           │ WHAT IF    sliders + table        │
└──────────────────────────────────────────────────┴───────────────────────────────────┘
```

### 5.4 Experiment view

Five tools, as today, as a segmented control with icons in the view header: Timeline, Compare
runs, Training, Toy lab, Datasets. Each tool is a 12-column grid of panels with 12 px gaps on the
page colour.

Timeline for a GA run:

```
[ run: picker · method · status · settings ]  [ 4 stats: generations · best · finishers · diversity ]   12
[ generation bar: number · slider · this generation's numbers | Fitness on ▾ · Replay · Race ]  sticky   12
[ fitness lines + band + checkpoints            ] 7     [ Driving Exam of the checkpoint ] 5
[ distribution heat map, same generation axis   ]
[ histogram ] 4      [ diversity ] 4      [ which starts are finished ] 4
[ outcome grid of the generation ] 7      [ ancestry of its champion ] 5
```

For a run of another method the GA panels are left out and the fitness panel shows that method's
own curve (loss, validation loss, best fitness), with its own objective badge.

Compare runs: the Driving Exam of every method across the top (12), then the run list (3) beside
one objective's chart and table (9).

Training: the launch form and run list (4) beside the watched run's live chart and exam table (8).

---

## 6. Components and their states

All of these are in `src/ui/controls.tsx` and `src/theme/base.css`.

### 6.1 Buttons

Height 30 px (28 on the stage), radius 8, 12.5 px text at weight 500, 12 px side padding; an
icon-only button is square. Icons are 16 px with a 1.5 px stroke.

| Variant | Rest | Hover | Pressed (mouse down) | Selected (`aria-pressed`) | Disabled |
|---|---|---|---|---|---|
| default | `--surface`, 1 px `--border-strong` | `--surface-3` | moves down 1 px | `--accent-soft` fill, `--accent` border and text | 45 % opacity, not-allowed cursor |
| primary | `--accent` fill, `--on-accent` text | `--accent-hover` | moves down 1 px | — | 45 % opacity |
| ghost | no border, `--text-2` | `--surface-3`, `--text` | moves down 1 px | as default | 45 % opacity |
| hud (on the stage and top bar) | transparent, `--hud-text` | `rgba(255,255,255,.12)` | moves down 1 px | `rgba(167,144,255,.26)`, white text | 45 % opacity |

One primary button per panel at most (Start training, Race generation).

**Focus** for every interactive element: a 2 px `--accent` outline, 2 px away from the element,
shown for keyboard focus only.

### 6.2 Segmented control

A `--surface-3` track with 2 px padding; options 26 px high. The selected option is a `--surface`
pill with a hairline ring and a 1 px shadow. On the stage: a `rgba(255,255,255,.08)` track and a
`rgba(255,255,255,.18)` selected option. Used for a choice among two to five named modes: speed,
camera, layout, theme, "Lines show", the Experiment tools.

### 6.3 Fields

A field is a capital label followed by its control. Selects and inputs are 30 px high, radius 8,
1 px `--border-strong`. Numbers typed into inputs are in the number face. A disabled field is at
50 % opacity and says why in its tooltip. Sliders take the accent colour.

### 6.4 Chips

20 px high, pill, condensed capitals, 11 px. A chip is a state, not a button.

| Kind | Look | Used for |
|---|---|---|
| run | neutral, with a small pulsing green dot | Running |
| ok | green text on a 15 % green wash, flag icon | Finished, completed |
| bad | red, cross icon | Crashed |
| warn | amber, pause icon | Stalled |
| idle | neutral, clock icon | Out of time, stopped |
| accent | violet | Held-out |
| plain | neutral | Method, split |

### 6.5 Panels, banners, tables

- **Panel:** `--surface`, 1 px `--border`, radius 8, 12 px padding. A panel has a title (13.5 px,
  weight 600) and may have a one-line note under it in `--text-3`.
- **Banner:** a `--surface-2` row with an icon, for context that belongs to the whole view
  (lineage, replay). The look-back banner uses `--accent-soft` with an accent border, because it
  changes what every number means.
- **Table:** capital-label headers, 6 × 10 px cells, hairline row rules, numbers right-aligned in
  the number face. Hovered row `--surface-2`; selected row `--accent-soft` with a 3 px accent bar
  at its left edge.

### 6.6 Stats, bars, legends

- **Stat:** a capital label, a figure, an optional unit and note. Three sizes: 15, 20, 30 px.
- **Value bar:** 8 px high on a `--surface-3` track. A signed bar grows from a centre tick: blue to
  the right, orange to the left. An unsigned bar grows from the left.
- **Legend:** a row of swatch-plus-words in 11.5 px `--text-2`. The swatch has the shape of the
  mark it explains: a line for lines, a square for cells, a ring for rings.

### 6.7 Tooltips

- **Hint** (on controls): a dark pill below the control after 350 ms. One short sentence.
- **Read-out** (on charts and the diagram): a dark card that follows the pointer, a capital-label
  title and rows of name and number. It is an HTML element, so it is sharp at any size (the
  current app learned this in task 5.8).
- Both are dark in both themes, so they read as "over" the interface.

### 6.8 Loading and empty states

- **Loading:** a shimmer block (`.skeleton`) in the place the content will take, never a spinner
  in the middle of nothing. Used while the recorded data loads; in production, while a scenario,
  a network or a run loads.
- **Empty or unavailable:** a centred block with a title in sentence case, one sentence saying
  why, and the one action that fixes it. Examples in the prototype: a car without a recorded
  network; a chart with nothing ticked; a what-if that cannot be asked while playing ("Pause to
  try it"); a live chart before the first metrics line.
- **Ended:** when the selected car stops, a banner on the stage says how, in words, with the same
  chip as the telemetry.

---

## 7. Track and cars

Everything in this section is drawing code in `src/views/environment/scene.ts` and
`trackArt.ts`. All of it is computed from the three polylines the server already sends
(`centerline`, `left`, `right`) plus `width`, `length` and `start_s0`.

### 7.1 Layers, bottom to top

| # | Layer | How it is drawn |
|---|---|---|
| 1 | Grass | Flat `--w-grass`, then mowing stripes (bands 9 m wide at 28° in `--w-grass-2`), then fine noise |
| 2 | Verge | The centreline stroked `width + 11 m` wide in `--w-verge`: 5.5 m of mown grass each side |
| 3 | Sand traps | `--w-sand` polygons outside bends tighter than 46 m radius, up to 7.5 m deep, tapering at both ends |
| 4 | Asphalt | A dark lip (`--w-asphalt-2`, 0.55 m beyond each edge), then the ring between the two edges in `--w-asphalt`, then noise |
| 5 | Rubber | The centreline stroked three times in black at 5 to 6 % opacity (5.6, 3.6, 1.9 m wide): a darker line where cars drive |
| 6 | Kerbs | Blocks 2.2 m long and 0.95 m wide, alternating `--w-kerb-a` and `--w-kerb-b`, **outside** the edge: on the inside of bends tighter than 85 m, and on the outside from mid-bend past the exit of bends tighter than 46 m |
| 7 | Edge lines | `--w-line`, 0.26 m wide (never under 1.4 px), exactly on `left` and `right` |
| 8 | Centre line | Dashes 2.6 m on, 5.4 m off in `--w-marking` (hidden below 1.4 px per metre) |
| 9 | Arrows | Double chevrons every 60 m on straights, in `--w-marking` (above 3 px per metre) |
| 10 | Start line | Two rows of twelve checks across the road at `start_s0` |
| 11 | Decision trail | See §7.4 |
| 12 | Sensor fan and rays | See §7.3 |
| 13 | What-if paths | The real action dashed white, the what-if action solid violet, both cased in dark |
| 14 | Cars | Worst rank first; then the ghost; then the selected car again on top; then the look-back car; then the what-if ghost |
| 15 | Labels | Distance boards, elite rings, status badges, the selection reticle and name tags |

Textures are two small seeded noise tiles made once per theme and repeated in world coordinates,
so the grain stays fixed to the ground when the camera moves. One asphalt tile covers 9 m, one
grass tile 14 m.

**Distance boards** stand 3.2 m outside the right edge at every 100 m of a lap, counted from the
start line. They tie the "Progress" figure in the telemetry to a place on the track.

### 7.2 Cars

The car's box is the one the simulation uses: 4.5 m long (wheelbase 2.5 m + 2 m, as the current
drawing code) and 2.0 m wide.

| Scale | Shape |
|---|---|
| 4.2 px per metre and above | An open-wheel race car: four tyres, front and rear wings in a darker shade of the body, a tapered body with a light stripe on the nose, a dark cockpit, a helmet, and a soft shadow offset 0.3 m right and 0.36 m down. Brake lights show when throttle is below −0.08. At night, running cars have a faint headlight cone. |
| below 4.2 | A dart: a rectangle with a pointed front, a 1 px dark outline |
| below about 2.2 | The dart is enlarged so it is never shorter than 10 px. This is the one place the picture is not to scale; it happens only at overview zooms. |

| State | Look |
|---|---|
| Running | Full colour |
| Stopped | 42 % opacity, and a round badge above it: red × crashed, green ✓ finished, amber Ⅱ stalled, grey ◷ out of time. Below 3.2 px per metre the badge becomes a small coloured dot (except on the selected car). |
| Selected | A four-part reticle ring (2.25 px `--track-select` over a 5 px dark casing, radius 3.5 m, never under 16 px), a violet helmet, and a name tag above ("CAR 47 · P1 · CHAMPION"). Drawn again on top of all other cars. |
| Elite | A 1.75 px gold ring |
| Ghost (an earlier champion) | A pale see-through body with a dashed white outline and a tag ("GHOST · GEN 17") |
| Look-back car | A violet see-through car where the selected car was, tagged with the step |
| What-if ghost | A violet see-through car with a dashed outline at the end of the what-if path, tagged "WHAT-IF · 2 s" |

**Body colour.** In population mode the body is the rank band (`--car-rank-1` … `-5`). For other
drivers (not in the prototype): one car is white with the accent helmet; up to ten cars take a
livery each from the eight dark-surface series colours plus white and graphite; more than ten cars
are all pale grey, the selected one white, because eleven or more liveries cannot be told apart
and identity then comes from the reticle, the name tag and the car picker.

### 7.3 Sensor rays

- A faint fan (white at 5.5 %) through the five ray ends shows the field of view.
- Each ray: a 2 px line, dark-cased, fading in over its first 26 px so the car stays readable; a
  3 px dot ringed in dark at the hit point; a label pill (number face, 11 px, on `--hud-bg`)
  pushed 12 px outward along the ray so neighbouring labels do not stack.
- Colour by distance: more than half the range `--ray-clear`, under half `--ray-near`, under a
  fifth `--ray-danger`. A ray that hits nothing is dashed in `--ray-none` and labelled "≥50 m".
- A selected input neuron's ray is 4.5 px; the others drop to 28 % opacity.
- With an output selected, the rays show sensitivity instead: `--track-pos` or `--track-neg`,
  1.2 to 6.2 px wide by size, labelled "s +0.57". The legend gains a line saying what the colours
  mean for that output.

### 7.4 Decision trail

The selected car's last 60 s. With no output selected: a 2.25 px line in `--track-select` over a
4.5 px dark casing. With an output selected: one segment per step in `--track-pos` or
`--track-neg`, 1.5 to 5.5 px wide and 45 to 100 % opaque by the size of the output there, over a
7.5 px casing. The casing is what makes it readable on asphalt, kerb and grass alike.

### 7.5 Telemetry, minimap, standings, look-back strip

- **Telemetry strip:** one floating panel along the bottom of the stage, in groups divided by
  hairlines: car (rank swatch, name, role, status chip) · speed (30 px, km/h, with m/s under it) ·
  steer and throttle (signed gauges with L/R and B/A ends, value, words) · lap (one bar per lap) ·
  rays (a fan glyph and five numbers) · fitness (F, finished and crashed starts, f on this start) ·
  step and time. When an earlier step is being looked at the strip gets a violet outline and its
  last label reads "Looking back".
- **Minimap:** 168 px wide, top right, under the camera controls: the circuit as a thick grey
  line, every car as a dot in its rank colour, the selected car as a larger violet dot. A dot can
  be clicked.
- **Standings** (drawer tab): four counts (running, finished, crashed, stopped) and a table in
  road order: position, rank swatch and car (gold ring for an elite, a crown for the champion, a
  status icon when stopped), a progress bar coloured by status, and F.
- **Look-back strip:** under the stage. Steer above, throttle below, one thin bar per remembered
  step, blue up and orange down, each row scaled to its own largest value (printed in its label).
  A violet cursor marks the step shown. Dragging scrubs; "Back to now" appears while looking back.

### 7.6 Camera

- **Follow:** 7.5 px per metre by default. The camera looks ahead of the car by
  `0.28 s × speed + 2 m` (at most 22 % of the stage's shorter side), so the road it is about to
  drive is on screen.
- **Whole track:** fitted inside the free area between the floating panels.
- **Free:** after a drag, or a wheel turn outside follow mode. A "Free view" note appears.
- A change of mode, of selected car, or into or out of look-back is a 450 ms eased move
  (position linear, zoom geometric), never a cut.
- North stays up. A heading-up camera was considered and left out: it would turn the minimap and
  the track's shape into something the user has to re-learn every corner.

---

## 8. Charts

All charts are SVG, drawn at the size they are shown (the chart measures its container), so text
is never scaled. The kit is `src/views/experiment/charts.tsx`.

### 8.1 Anatomy

```
Title (13.5 px, 600)                                         [ OBJECTIVE ↑ note ]  [controls]
one-line note in --text-3: how to read it
 1 ┤ ─────────────────────────────────────  gridlines: 1 px --grid, solid
   │                                  ● best    labels at the line ends, pushed apart ≥ 13 px
 0 ┴───────────────────────────────── baseline: 1 px --axis
   1     5     10    15    20   25   30         ticks: 10.5 px number face, --text-3
                   generation                   axis label: 11 px condensed, --text-3
legend (when there are 2 or more series or shades)
```

### 8.2 Marks

| Mark | Specification |
|---|---|
| Line | 2 px, round joins and caps |
| Band (middle half, q1 to q3) | The line's colour at 14 % opacity |
| Reference line | 1.5 px, dashed 4 4, `--text-3` |
| Per-seed line | 1 px at 35 % opacity |
| Bar | At most 24 px wide, 3 px rounded data end, square at the baseline, `--series-1` |
| Heat-map cell | A 1 px gap all round in the surface colour, no outline |
| Marker | Radius 4.5 with a 2 px ring in the surface colour |
| Checkpoint | A hollow ring: surface fill, 2 px stroke in the line's colour, radius 3.5 |
| Value labels | Only where they say something: the tallest bar, the ends of lines. Never one per point. |

### 8.3 The objective badge

Every chart carries a small badge at its top right saying what kind of number it shows.

| Badge | Text | Look |
|---|---|---|
| Loss | LOSS ↓ LOWER IS BETTER | Outlined |
| Fitness | FITNESS ↑ HIGHER IS BETTER | Outlined |
| Reward (Phase 10) | REWARD ↑ HIGHER IS BETTER | Outlined |
| Diversity | DIVERSITY · NOT AN OBJECTIVE | Outlined |
| Driving Exam | DRIVING EXAM ↑ THE COMMON YARDSTICK | **Filled**: the only filled badge |

Rules that follow from the concept document ("loss, fitness and reward are not interchangeable"):
1. A chart holds one objective. No chart has two y-axes.
2. Loss, fitness and reward are never given a shared colour or a shared axis across panels.
3. Different methods are put on one axis only in a Driving Exam chart.
4. The badge's tooltip says what the number can be compared with.

### 8.4 Interaction

- **Generation cursor:** a column in `--accent` at 13 % opacity across every chart that has
  generations on its x-axis. A click or a drag on any of them moves it. On grids, the selected
  column or cell gets a 2 px accent outline; a second, dashed outline in `--text` marks the start
  that is raced.
- **Hover:** a read-out card with every series' value at that x; on line charts a hairline
  crosshair and a marker on each line.
- **Cells and bars** show their exact count, share or outcome in the read-out; a cell of the
  outcome grid selects that car in every view.

### 8.5 Chart types used

| Chart | Encoding |
|---|---|
| Fitness per generation | Four lines in `--series-1` … `-4` (best, median, mean, worst), the middle half as a band on the median's colour, checkpoint rings on the best line, labels at the line ends |
| Distribution | Columns = generations, cells = fitness bands of 0.05, five shades of the ramp by the share of the generation (up to 5, 10, 20, 40 %, more); shares the x-axis of the chart above it |
| Histogram | Bars per fitness band for one generation |
| Diversity | One line in `--text-2` (it is not an objective, so it takes no series colour) |
| Starts map | Rows = starts, columns = generations, six shades by the share that finished (none, up to 20, 40, 60, 80 %, more) |
| Outcome grid | Rows = starts, columns = networks by rank, cells in the four status fills |
| Ancestry | The line of descent in `--accent` over the generation's best as a dashed reference; markers by origin: square = random, ring = elite copy, dot = mutated child, diamond = child of two parents |
| Compare (bands) | One colour per group: median line, middle-half band, optional per-seed lines; log y-axis for losses; labels at the line ends; a table of final values below |
| Exam strip | One row per method: a dot per seed, a thick tick at the median, a bar for the middle half; the row label is the method, so no legend is needed |
| Live chart | The run's line, a pulsing last marker, the axis not yet reached shaded in `--surface-2`, exam rings |

### 8.6 The Driving Exam card

Two blocks side by side: "Training track" on `--surface-2` and "Held-out tracks · never trained
on" on `--accent-soft`, each with its stats. Under them one row per exam track with a chip
(train / held-out), a small meter and count for finished starts, crashed, median progress and best
lap. A closing sentence says what the card is of.

### 8.7 Toy lab and Datasets (specified, not built)

Both keep every function they have and take on the kit:

- **Toy lab.** Loss curve: §8.2 line with the Loss badge and the checkpoint cursor. Prediction
  plot: targets as ringed dots, the network's curve as a 2 px line in `--series-1`; for XOR and
  two spirals the map is diverging (`--pos`, `--neg`, `--mid` at the middle). Gradient glow: the
  Network view's diagram (§9) with "Lines show" set to a fifth mode, Gradient, shaded by |dL/dW|.
- **Datasets.** Histograms: §8.2 bars; two datasets as paired bars in `--series-1` and
  `--series-2`; the dense-coverage stretches as a 3 px rule under the axis in `--text`. Track map:
  the §7 track at whole-track scale with layers 11 to 15 replaced by one dot per frame, on the
  diverging pair for steer and on the ramp for throttle. Episodes: the §6.5 table, status as chips.

---

## 9. Network diagram

`src/views/network/encode.ts` and `Diagram.tsx`.

- **Layout.** Layers are columns, evenly spaced. All layers share one row spacing (set by the
  widest layer, at most 30) so the hidden layers line up; the output layer is spaced 104 apart
  because each output carries a label, a value, a bar and a word.
- **Connections** are cubic curves that leave and arrive horizontally. They are drawn weakest
  first, so strong ones end up on top. Each has a 9 px invisible hit curve above it.
- **Neurons.** Radius 10 (outputs 13): a surface-coloured disc, a fill in `--pos` or `--neg` whose
  opacity is the size of the value (at least 4 %), a 1.25 px ring in `--border-strong`.

| "Lines show" | Colour | Width | Opacity |
|---|---|---|---|
| **Weights** | sign of the weight | 0.35 + 3.4 × min(1, \|w\|) | 0.10 + 0.75 × min(1, \|w\|) |
| **Flow** (proposed) | sign of weight × source value | 0.3 + 4.4 × s^1.3, s = size ÷ the layer's largest | 0.04 + 0.92 × s^1.5 |
| **Mutation** | blue raised, orange lowered; second-parent genes thin dashed grey | mutated: 1.2 + 3.6 × \|change\| | mutated 1; second parent 0.5; unchanged 0.05 |
| **Change** (checkpoint replay) | blue raised, orange lowered | as Mutation | unchanged weights fade out |

| State | Neuron | Connections |
|---|---|---|
| Rest | as above | as the table above |
| Hover | ring 2 px in `--text`; a read-out card follows the pointer | those touching it keep at least 45 % opacity; all others drop to a tenth of their own |
| Selected | ring 2.5 px in `--accent`, plus a dashed halo 6 px out on `--accent-soft` | as hover, kept until something else is selected |
| Selected output | as selected | additionally, the strongest routes back to the inputs stay lit at 75 % or more: the 4 largest contributions into the output, the 2 largest into each of those, the 2 largest into each of those |
| Selected connection | — | drawn in `--accent`, 1.6 px wider, with a soft accent shadow |
| Pointed at from the inspector | as hover, while the row is hovered | as hover |
| Bias mutated (Mutation mode) | a dashed ring 3.5 px outside | — |

- **Inputs** carry their name, their value and a 40 px bar (signed for the heading error), so a
  sensor reading can be read without hovering. **Outputs** carry their name, value, a signed bar
  and the meaning in words.
- **Column heads:** a capital label and, under it, the count and the activation function. Under
  each gap between columns: "W1: 112 weights + 16 biases".
- **Read-out line** under the diagram repeats the tooltip's content for whatever is hovered or
  selected, so the numbers can be read without holding the pointer still. For a neuron:
  Σ weight × input, bias, z, activation(z). For a connection: weight, source value, what it
  carries.
- **Below 560 px of view width** the diagram is replaced by the compact summary (input bars, a
  strip of cells per hidden layer, output bars with words), which the drawer already uses.

---

## 10. Themes

- Light and dark are both complete; neither is derived from the other by a filter. Every token in
  §2.2, §2.3 and §2.5 has a value chosen for each.
- The theme switch has three positions: light, dark, follow the system. It sets
  `data-theme` on `<html>`; all tokens hang off that attribute.
- **What changes with the theme:** page surfaces, text, borders, the accent's lightness, the sign
  pair's lightness, status colours, series steps, the direction of the ramp, and the world's
  brightness.
- **What does not change:** the top bar, the floating panels over the track, tooltips, and
  everything drawn on the asphalt (§2.4). The track is the same place by day and by night; only
  the light on it changes.
- The dark theme is not black: `#0b0d11` page, `#14171d` panels, both slightly blue, so the
  carbon-grey top bar and HUD panels still read as a separate material.

---

## 11. Motion

| Token | Duration | Used for |
|---|---|---|
| `--dur-1` | 120 ms | Hover and press on controls, table rows, inspector rows |
| `--dur-2` | 200 ms | Selection: lines dimming and lighting, the tab underline, theme colours |
| `--dur-3` | 450 ms | Camera moves between modes, cars and moments |

All use `--ease` = `cubic-bezier(0.2, 0.7, 0.2, 1)`: quick to start, soft to stop.

Rules:
1. **Data is never animated for effect.** A bar or a line shows the current number. Bars follow
   their value with a 90 ms linear step only so that 30 updates a second look continuous.
2. **Motion explains a change of view**, nothing else: the camera travels instead of cutting; the
   unrelated lines fade instead of vanishing.
3. Two things pulse, both meaning "live": the dot in a Running chip and the last point of a chart
   that is still growing (1.2 to 1.6 s, opacity only).
4. With `prefers-reduced-motion`, every transition and animation is cut to nothing.

---

## 12. Small windows

A view adapts to **its own width**, not the window's, wherever the browser allows (container
queries), so the same rules work in Page, Split and All layouts.

| Width | What changes |
|---|---|
| Window under 1560 px | Layout and theme switches show icons only; the connection badge shows its dot only |
| Window under 1300 px | The breadcrumb drops Start and Moment |
| Window under 1120 px | The breadcrumb is hidden |
| Stage under 1120 px | The telemetry strip drops the fitness group |
| Stage under 900 px | … and the rays group |
| Stage under 740 px | … and the lap and step groups |
| Network view under 940 px | The inspector goes below the diagram |
| Network view under 560 px | The diagram is replaced by the compact summary; "Lines show" is hidden |
| Window under 1180 px | Every Experiment panel takes the full width |
| Window under 860 px (phone) | The top bar wraps into two rows; the view title loses its question; the scenario fields, minimap and stage legend are hidden; the stage's two control bars stack; the drawer goes below the view; the generation bar stops being sticky |

On a phone the Environment view shows the track, transport, the car, its speed, steer and
throttle, and the look-back strip. Setting up a scenario is not offered on a phone in the
prototype; in production the scenario fields would go in a sheet opened from the header.

---

## 13. Accessibility

- Text contrast: primary and secondary text at least 7.8:1, labels at least 4.4:1, in both themes
  (§2.2).
- Colour is never the only carrier: status has icons and words; series have labels at the line
  ends and a table; sign has position (left or right of a centre tick) as well as colour; rank has
  the standings table; elites have a ring; the ghost has a dashed outline.
- The sign pair, the series order and the status fills pass the validator's colour-blind checks.
- Every button, field, tab and slider can be reached and operated from the keyboard; focus is
  always visible. Icon-only buttons have an accessible name and a tooltip.
- Neurons have `role="button"`, `aria-pressed` and a spoken name. Charts have an `aria-label`.
- Reduced motion is honoured (§11).
- Known gaps, shared with the current app: neurons and connections in the diagram, chart cells
  and cars on the canvas are reached by pointer only. The compact network's rows and cells and
  the inspector's output cards are real buttons, so they are the keyboard route to selecting a
  neuron. The Standings rows should become buttons in production so they are the keyboard route
  to selecting a car; in the prototype they are click-only.

---

## 14. Directions that were considered and rejected

| Direction | Why not |
|---|---|
| **Keep blue as the accent** | It is already "positive". The current app's single blue for four jobs is finding A2 of the audit. |
| **Racing red as the accent** | Red must stay free to mean "crashed". |
| **Amber or yellow as the accent** | Collides with "negative" orange and "stalled" amber. |
| **A neon, glowing "AI" look** (dark only, glows, gradients, glass panels) | Glow and blur lower contrast exactly where numbers are read; it would be dark-only in spirit, and the brief asks for both themes as equals. |
| **A light track in the light theme** (pale asphalt, as now) | Rank colours, rays and trails would need a second set of on-track colours, and the track would stop looking like asphalt. One dark road in both themes is simpler and more real. |
| **A rank ramp from gold to grey** | Gold next to "negative" orange and "stalled" amber is one warm hue too many; gold is kept for the elite ring alone. |
| **Photographic textures or sprite images** | Image assets to license, load and theme. Drawn shapes and seeded noise give enough surface, scale without blur and follow the tokens. |
| **A heading-up (rotating) camera** | Disorienting next to a north-up minimap and a north-up track map in Datasets. |
| **Tabs inside the inspector** (Why / Sensitivity / What-if) | Hides two thirds of the explanation at any moment. A scrolling rail shows all of it. |
| **A charting library** | The app's charts are its own SVG by decision (ADR-033). The kit needs about 200 lines; a library would add a dependency and a second visual language. |
| **A separate font for the brand** | A third family for one word. The condensed cut of the same family does it. |
