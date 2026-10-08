# seatmap-to-figma

Convert a **raster seatmap image** (PNG/JPG) into a **named, clickable, Figma-ready
SVG** - one vector path per seating section + a grouped text layer. No vector source
and no OCR required: a vision-capable agent reads the section labels; OpenCV does the
geometry.

Designed to be driven by **Claude Code** (via `SKILL.md`) or **Codex** (via this
README) or by hand.

## Install
```
pip install opencv-python-headless numpy
```

To use it as a **Claude Code skill**, put this folder in your skills directory so Claude
picks it up whenever you ask it to vectorise a seat map:
```
git clone https://github.com/NeedSomeFoodMr/seatmap-to-figma ~/.claude/skills/seatmap-to-figma
```
Any other agent, or a person, can follow the steps below directly.

## Use (3 steps)

**1. Extract** geometry + the price-category palette (auto-sampled from the legend):
```
python scripts/seatmap.py extract seatmap.png work
```
→ `work/preview.png` (sections outlined), `work/meta.json`, `work/labels.template.json`.

**2. Read the labels.** Look at `work/preview.png` and the original, then write
`work/labels.json` with one entry per printed section id. `x,y` are the centre of the
id text as fractions 0..1 (must land inside the coloured block):
```json
{ "labels": [
  { "id": "420",  "x": 0.83, "y": 0.61 },
  { "id": "PA14", "x": 0.37, "y": 0.31 }
] }
```
For a block drawn in two colour bands but labelled once, one seed captures the whole
block; add a second entry with the same `id` if a hard stroke splits the bands.

**3. Build** the SVG (per-colour pieces, geometry-split, named by seed id):
```
python scripts/seatmap.py build seatmap.png work/labels.json work
```
→ `work/seatmap_figma.svg` (the deliverable), `work/build_debug.png` (verify seat
placement; move any mis-seeded label and rerun), and `work/uncovered.png`.

**3b. Read the `COVERAGE` line - this is the reliability net.** The engine only places
sections you seed, so it also reports every coloured blob **no seed claimed**:
```
COVERAGE: OK -- every coloured section-sized blob is claimed by a seed.
COVERAGE: 7 coloured blob(s) have NO seed -> work/uncovered.png
```
If not OK, open `uncovered.png` (misses outlined red + numbered) and for each: add a
`labels.json` entry - with a **new id** if it's a section you missed, or the **same id**
as an existing section if it's that section's second colour band (e.g. a fan block's
thin cyan/blue front-row strip). Seed a **solid interior pixel of the blob** (a thin
strip's centroid can fall in a gap). Rebuild until `COVERAGE: OK` (bar the odd thin rim
of an already-captured strip, which is safe to ignore).

## Into Figma
Drag `seatmap_figma.svg` onto the canvas. The file is one outer group `<g id="seatmap">`
wrapping a `<g id="sections">` (one individually-named vector per section) and a
`<g id="labels">` (all the labels) - i.e. sections grouped, labels grouped, both grouped
together, mirroring the known-good hand-built maps. Fills are neutral `#E9E9EC`; the
drawn tier colour is preserved on each path as `data-cat-colour`.

### Labels not showing up on import?
Live `<text>` renders only where the viewer has the **font**. Labels are emitted as
`Inter` (fallback Arial/sans-serif) - the font that renders in typical Figma-derived
pipelines. **First fix:** if your target lacks Inter, change the labels group's
`font-family` to a font it has - labels stay editable text. **Bulletproof fix** (no font
needed at all, or the importer drops `<text>`): rebuild with `--outline-text`:
```
python scripts/seatmap.py build seatmap.png work/labels.json work --outline-text
```
Each label becomes a **vector `<path id="Text_<id>">`** (built-in font, letter holes via
`fill-rule="evenodd"`) - geometry like the sections, so it imports regardless of fonts.
The `Text_` prefix keeps label ids from colliding with the section of the same id, and
matches the convention where seat-pickers skip ids containing "Text" (so labels don't
become clickable phantom seats). This is the format the known-good hand-built exports
use. The text is then outlines (not editable as text in Figma); keep the default
`<text>` only when you specifically want editable labels and the target has the font.

## How it works
- **Palette**: the legend swatches (solid saturated rects, bottom-left) are sampled
  top→bottom to give the exact category colours and order. Fallback: k-means on
  saturated pixels.
- **Sections mask**: pixels near a palette colour, minus dark separator strokes and
  Canny edges, so abutting blocks are cut apart.
- **Trace the mask**: each per-colour connected piece is traced directly (printed
  numbers are painted out first so they can't bite a false notch). A piece still holding
  several merged sections is split by geometry - arc/ring pieces by angle around the
  bowl centre (radial trapezoids), grid blocks euclidean (Lloyd-relaxed) - then fit to a
  small (~quad) polygon → named `<path>`. Real notches (stage overlap, band boundaries)
  are preserved; the engine never invents or "corrects" geometry.
- **Coverage check**: after building, coloured pixels no seed claimed are reported and
  drawn to `uncovered.png`, so a section you forgot to seed can't be silently dropped.
- A section you seed is never removed by the size filter (the seed proves it's real), so
  tiny blocks and thin front-row bands survive.

## Limitations
- Seed accuracy drives correctness - read positions from the full-res image, and seed a
  solid interior pixel (not a thin strip's centroid).
- Isometric/3D maps trace less cleanly than flat top-down ones.
- Very thin rings/rims (< a few px) may need a seed nudge or are a safe coverage dismiss.

## Licence
MIT. See [LICENSE](LICENSE).
