---
name: seatmap-to-figma
description: Convert a raster seatmap image (PNG/JPG of a stadium/arena/theatre seating plan) into a named, clickable, Figma-ready SVG - one vector path per section plus a grouped text layer. Use when the user wants to vectorise a seat map, turn a seatmap into Figma sections/clickables, trace seating blocks, or build an interactive venue map from an image. Works from a picture alone (no vector source, no OCR).
---

# Seatmap → Figma clickable sections

Turn a **raster seatmap** into a layered SVG that drops straight into Figma. The output
is one outer group `<g id="seatmap">` wrapping `<g id="sections">` - one `<path>` per
section, named by its id (e.g. `420`, `PA14`) - and `<g id="labels">` - one `<text>` per
id (sections grouped, labels grouped, both grouped together). Every path is an
independent clickable vector. Fills are neutral/monochrome (style them in Figma); the drawn price-category
colour is kept only as `data-cat-colour` metadata.

The engine (`scripts/seatmap.py`) is a plain Python CLI so **any** agent (Claude Code,
Codex) or a human can run it. It needs `opencv-python-headless` and `numpy`
(`pip install opencv-python-headless numpy`). **No OCR** - you (the agent) read the
labels with vision.

## Pipeline (agent-in-the-loop)

### 1. Extract - deterministic geometry + palette
```
python scripts/seatmap.py extract <seatmap.png> <workdir>
```
Produces in `<workdir>`:
- `meta.json` - image size and the price-category palette, auto-sampled from the
  legend swatches (top→bottom = category order). Falls back to colour clustering
  if no legend is found.
- `preview.png` - the seatmap with every detected section outlined (1400px wide).
- `labels.template.json` - the shape of the file you write next.

### 2. Read the labels (this is your job - vision)
Open `preview.png` **and** the original image. For **every printed section id**,
write an entry to `<workdir>/labels.json`:
```json
{ "labels": [ { "id": "420", "x": 0.83, "y": 0.61 }, ... ] }
```
- `id` - exactly as printed (`420`, `PA14`, `107A`, `PB13`).
- `x`, `y` - the centre of that id's text, as **fractions 0..1** of image
  width/height. The seed must fall **inside the coloured block**, so read the actual
  full-resolution image, not a thumbnail. Absolute pixel coords are also accepted.
- **Every id once.** For a section drawn in two colour bands but labelled once
  (e.g. a yellow cap over a blue body), place **one** seed on the block - watershed
  claims the whole block. If a hard stroke splits the bands and you want both
  captured, add a second entry with the **same id** on the other band (this mirrors
  the common Figma convention of two vectors sharing one name).

### 3. Build - seed-Voronoi → named SVG
```
python scripts/seatmap.py build <seatmap.png> <workdir>/labels.json <workdir>
```
Sections are cut from the **real drawn dividers** (per-colour connected pieces) so each
gets its TRUE shape wherever a divider is visible. A piece that still holds several
merged sections (faint dividers) is split by its geometry: **arc/ring pieces split by
angle around the bowl centre** (clean radial trapezoids, no diamonds); **grid blocks
split euclidean** (Lloyd-relaxed). Each region is fit to a small (~quad) polygon and
named by its seed's id; labels stay at the printed position. A tiny block you seeded is
never dropped by the size filter (a seed is proof the section is real). Produces:
- `seatmap_figma.svg` - the deliverable.
- `build_debug.png` - the map with placed sections outlined and seeds dotted.
  **Look at it.** Any `id` that outlines the wrong block = move that seed and rebuild.
  The command prints `placed/total`; if a seed produced no region it sat off a section.
- `uncovered.png` - **the coverage net; the single most important reliability step.**

### 3b. Coverage - let the engine tell you what you MISSED
The engine can only place sections you seed; an unseeded coloured block is invisible to
it and silently absent. So after every build it scans the coloured pixels **no seed
claimed** and prints one of:
```
COVERAGE: OK -- every coloured section-sized blob is claimed by a seed.
COVERAGE: 7 coloured blob(s) have NO seed -> work/uncovered.png
    area=  1650  #36fefe  at (0.424,0.379)   ...
```
If it is not OK, **open `uncovered.png`** (each miss is outlined red + numbered) and for
each one decide:
- a **section you forgot to label** → add a normal `labels.json` entry;
- a **second colour band of a section you did label** (very common: a fan block drawn
  as a magenta body **plus a thin cyan/blue front-row strip**, one printed label) → add
  an entry with the **same id** on the strip → two paths, one label;
- a genuine **rim/decoration** (a thin sliver hugging a section of the *same colour*
  that is already captured) → dismiss it.

Put the seed on a **solid interior pixel of the flagged blob**, not on the reported
centroid - a curved/thin strip's centroid can fall in a gap and snap to the wrong piece.
Rebuild and repeat until `COVERAGE: OK` (bar the odd dismissable rim). This loop is what
turns a good-looking 90% draft into a verified-complete one.

### 4. Into Figma
Drag `seatmap_figma.svg` onto the canvas (or File → Place). Sections import as
individually-named vectors; the labels come in as one `labels` group. No HTML,
no html.to.figma, no plugin.

## Tips
- The accuracy loop is: `build` → read the `COVERAGE` line → fix `labels.json` →
  rebuild. `build_debug.png` shows wrong-shaped/mis-seeded sections; `uncovered.png`
  shows missing ones. Both are cheap; don't ship until you've looked at both.
- **The engine traces the colour mask directly** - it does not invent geometry. Real
  structural notches (a stage overlapping a standing pen, a two-band boundary) are kept;
  only the printed number text is painted out first so it can't bite a false notch. If a
  shape is wrong, the seed or the mask is wrong - don't reach for shape-fitting.
- Isometric/3D maps are supported: the darker extruded side faces fall outside the
  colour threshold, so only the flat top face is traced (extrusion handled for free).
- Blocks are often **multi-band** (a coloured body + a thin front-row strip of another
  colour, one label). Seed each band with the **same id**; the coverage check will
  remind you of any band you skipped.
- To re-style by price tier later, read each path's `data-cat-colour`.
- **Stage / FOH:** seed it like a section with id `STAGE` (or `FOH`); its label is drawn
  **white**. `extract` reports any **solid dark blocks** it finds (`DARK BLOCKS:` line +
  `meta.json` `dark_blocks`) - those are almost always a black FOH box or a filled black
  stage, so seed one STAGE/FOH per reported block. A solid dark block IS traced into a
  dark `<path>`. But a **white, open-outlined stage** (a big white shape with a thin
  black border, e.g. an indoor-stadium centre stage) can't be traced from a raster - it
  merges with the surrounding white void - so seeding it places only the white label;
  draw the stage block itself in Figma during your edit pass (one-time, quick).
- **Workflow note:** if you'll edit the map in Figma before exporting, keep labels as
  live `<text>` (the default) so they stay editable, and let Figma outline them on
  export (enable "Outline text"). Use `--outline-text` only when going straight to the
  target app without a Figma edit pass.
- **Labels dropping on import?** Live `<text>` renders only where the viewer has the
  **font** and supports SVG `<text>`. First suspect is the font: the labels are emitted
  as `Inter` (falling back to Arial/sans-serif) because that's what renders in typical
  Figma-derived pipelines - if your target lacks Inter, set the group's `font-family`
  to one it has, keeping editable text. If the target can't be relied on for *any* font
  (or drops `<text>` entirely), use **`--outline-text`** - the bulletproof route, and
  the format the known-good hand-built exports actually use. It renders each label as a
  vector `<path id="Text_<id>">`: the `Text_` prefix means it never collides with the
  section path of the same id, and readers that hit-test sections by `<path>` id
  commonly skip ids containing `"Text"`, so outlined labels are ignored for clicks while
  still drawing. This is the safest choice for a seat-picker that selects by path id.
  The `<svg>` is always emitted with explicit `width`/`height` attributes (not only
  `viewBox`) - some readers parse dimensions by string-matching `width="…"` and crash
  without them. `python scripts/seatmap.py build map.png work/labels.json work
  --outline-text` renders each label as **vector `<path>` glyphs** instead (OpenCV's
  built-in font, letter holes preserved via `fill-rule="evenodd"`), so labels become
  geometry like the sections - font-independent and importable anywhere. Trade-off: the
  text is no longer editable as text in Figma (it's outlines). Use it for the
  import-into-my-software deliverable; keep the default `<text>` when you want editable
  labels in Figma.
