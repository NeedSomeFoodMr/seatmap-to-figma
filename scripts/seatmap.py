#!/usr/bin/env python3
"""seatmap.py -- turn a raster seatmap PNG into named, clickable Figma-ready SVG.

Two-stage, agent-in-the-loop pipeline (works for Claude Code, Codex, or a human):

  1) extract  PNG  ->  palette (from legend) + section polygons + preview overlay
                       The agent then LOOKS at the preview and writes labels.json
                       (every printed section id + its position, as 0..1 fractions).

  2) build    PNG + labels.json  ->  seed-watershed that splits/merges/names every
                       section, then writes a monochrome, layered SVG:
                         <g id="sections"> one <path> per id  (colour bands merged)
                         <g id="labels">   one <text> per id
                       -> Figma: File > Place image / drag the .svg in. Each path is
                          a clickable vector named by section id; texts are one group.

Dependencies: opencv-python(-headless), numpy.  No OCR required.
"""
import cv2, numpy as np, json, os, argparse


# ----------------------------- shared helpers -----------------------------
def detect_legend_palette(img, min_area_frac=3e-5):
    """Legend swatches = solid saturated rects stacked in the bottom-left.
    Returns (palette_bgr[N,3], legend_box) top->bottom = category order."""
    H, W = img.shape[:2]
    x0, x1 = 0, int(0.10 * W); y0, y1 = int(0.75 * H), int(0.99 * H)
    reg = img[y0:y1, x0:x1]
    hsv = cv2.cvtColor(reg, cv2.COLOR_BGR2HSV)
    m = ((hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 90)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
    min_a = min_area_frac * W * H
    rows = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_a:
            continue
        rows.append((cent[i][1], reg[lab == i].mean(axis=0)))
    rows.sort(key=lambda r: r[0])
    pal = np.array([r[1] for r in rows]).astype(int) if rows else np.empty((0, 3), int)
    return pal, (x0, y0, x1, y1)


def kmeans_palette(img, k=14, s_min=90, v_min=120, merge_dist=45):
    """Fallback palette when no legend is found."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] > s_min) & (hsv[:, :, 2] > v_min)
    pts = img[mask].astype(np.float32)
    if len(pts) > 80000:
        pts = pts[np.random.choice(len(pts), 80000, replace=False)]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 1.0)
    _, lbl, centers = cv2.kmeans(pts, k, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(int)
    counts = np.bincount(lbl.flatten(), minlength=k)
    merged = []
    for i in np.argsort(-counts):
        c = centers[i]; b, g, r = int(c[0]), int(c[1]), int(c[2])
        if max(r, g, b) < 110 or max(r, g, b) - min(r, g, b) < 32:
            continue
        if all(np.linalg.norm(c - m) > merge_dist for m in merged):
            merged.append(c)
    return np.array(merged)


def section_mask(img, palette, legend_box):
    """Boolean mask of all seat-section pixels (any category) minus separators."""
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    coloured = (hsv[:, :, 1] > 55) & (hsv[:, :, 2] > 90)
    if legend_box:
        lx0, ly0, lx1, ly1 = legend_box
        coloured[ly0:ly1, lx0:lx1] = False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.dilate(cv2.Canny(gray, 40, 120), np.ones((3, 3), np.uint8))
    outline = (gray < 95) | (edges > 0)
    # nearest palette label + distance, computed per-colour to keep memory low
    imgi = img.astype(np.int32)
    best = np.full((H, W), 1 << 30, np.int32)
    lbl = np.zeros((H, W), np.int16)
    for ci, col in enumerate(palette):
        d = ((imgi - col.astype(np.int32)) ** 2).sum(2)
        upd = d < best; best[upd] = d[upd]; lbl[upd] = ci
    near = np.sqrt(best) < 120
    sect = coloured & near & (~outline)
    return sect, lbl


def inpaint_labels(img, legend_box):
    """Paint over the printed number labels with the surrounding block colour, so a
    number touching a block edge doesn't bite a notch out of the traced shape. Only
    dark text INTERIOR to a colour block is removed -- the white stage overlap and the
    block borders are left alone (so real structural notches survive)."""
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    coloured = ((hsv[:, :, 1] > 55) & (hsv[:, :, 2] > 90)).astype(np.uint8)
    # fill the block first (text is dark, so it's NOT in `coloured`); then dark pixels
    # inside the filled block interior are the number text.
    block = cv2.morphologyEx(coloured, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    interior = cv2.erode(block, np.ones((6, 6), np.uint8))
    text = ((gray < 110) & (interior > 0)).astype(np.uint8)
    if legend_box:
        lx0, ly0, lx1, ly1 = legend_box
        text[ly0:ly1, lx0:lx1] = 0
    text = cv2.dilate(text, np.ones((3, 3), np.uint8))
    return cv2.inpaint(img, text, 6, cv2.INPAINT_NS)


def hexcol(bgr):
    return '#%02x%02x%02x' % (int(bgr[2]), int(bgr[1]), int(bgr[0]))


def simplify(contour):
    """Light polygon approximation of a (convex) section outline -- clean straight
    edges between real corners; a parallelogram collapses to ~4-5 points, a short
    arc keeps a couple of points along its outer curve."""
    peri = cv2.arcLength(contour, True)
    p = cv2.approxPolyDP(contour, 0.008 * peri, True)
    return [[int(x), int(y)] for x, y in p.reshape(-1, 2)]


def poly_to_d(poly):
    """A closed SVG path `d` string from a list of [x, y] points."""
    return 'M' + ' L'.join(f'{x},{y}' for x, y in poly) + ' Z'


# ------------------------------- stage 1 ----------------------------------
def extract(png, outdir, preview_w=1400):
    img = cv2.imread(png)
    if img is None:
        raise SystemExit(f"cannot read {png}")
    H, W = img.shape[:2]
    os.makedirs(outdir, exist_ok=True)
    palette, legend_box = detect_legend_palette(img)
    source = 'legend'
    if len(palette) < 2:
        palette, legend_box, source = kmeans_palette(img), None, 'kmeans'
    sect, lblmap = section_mask(img, palette, legend_box)
    min_area = 2.5e-4 * W * H

    # per-category preview contours (so the agent can see what was found)
    overlay = img.copy()
    n_found = 0
    for ci in range(len(palette)):
        m = ((lblmap == ci) & sect).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < min_area:
                continue
            cv2.polylines(overlay, [c], True, (0, 0, 0), 2); n_found += 1

    # Solid dark blocks (a black FOH box or a filled black stage) are not a priced tier
    # so they never show as sections -- agents keep forgetting them. Detect and report
    # them as candidate STAGE/FOH seeds so they get labelled. (A white, open-outlined
    # stage cannot be detected here -- it merges with the surrounding white void -- so
    # look at the map for a "STAGE" that isn't a solid dark block and seed it by hand.)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ndk, dkl, dkst, dkc = cv2.connectedComponentsWithStats(
        (gray < 70).astype(np.uint8), 8)
    dark_blocks = []
    for j in range(1, ndk):
        x, y, w, h, a = dkst[j]
        if a > min_area and a / float(w * h) > 0.35:   # big and solid
            dark_blocks.append((round(dkc[j][0] / W, 3), round(dkc[j][1] / H, 3), int(a)))

    meta = dict(size=[W, H], preview_w=preview_w, palette_source=source,
                palette=[hexcol(c) for c in palette],
                legend_box=list(legend_box) if legend_box else None,
                dark_blocks=[{"x": x, "y": y, "area": a} for x, y, a in dark_blocks])
    json.dump(meta, open(os.path.join(outdir, 'meta.json'), 'w'), indent=2)
    cv2.imwrite(os.path.join(outdir, 'overlay_full.png'), overlay)
    scale = preview_w / W
    cv2.imwrite(os.path.join(outdir, 'preview.png'),
                cv2.resize(overlay, (preview_w, int(H * scale))))
    tmpl = {"note": "Fill labels for EVERY printed section id. x,y are 0..1 "
                    "fractions of image width/height, at the centre of the id text.",
            "labels": [{"id": "EXAMPLE_101", "x": 0.5, "y": 0.5}]}
    json.dump(tmpl, open(os.path.join(outdir, 'labels.template.json'), 'w'), indent=2)
    print(f"extract: {W}x{H}  palette={len(palette)} ({source})  preview contours~{n_found}")
    print(f"  -> {outdir}/preview.png   (read it, then write {outdir}/labels.json)")
    print(f"  -> template: {outdir}/labels.template.json")
    if dark_blocks:
        print(f"  DARK BLOCKS: {len(dark_blocks)} solid dark block(s) found -- likely "
              f"STAGE/FOH. Seed each with id STAGE or FOH (white label). frac x,y:")
        for x, y, a in sorted(dark_blocks, key=lambda b: -b[2]):
            print(f"    at ({x},{y})  area={a}")


# ------------------------------- stage 2 ----------------------------------
def build(png, labels_json, outdir, outline_text=False):
    img = cv2.imread(png)
    if img is None:
        raise SystemExit(f"cannot read {png}")
    H, W = img.shape[:2]
    os.makedirs(outdir, exist_ok=True)
    palette, legend_box = detect_legend_palette(img)
    if len(palette) < 2:
        palette, legend_box = kmeans_palette(img), None
    img = inpaint_labels(img, legend_box)   # remove number text before tracing
    sect, lblmap = section_mask(img, palette, legend_box)
    CY_, CX_ = np.argwhere(sect).mean(0)          # bowl centre (row, col)
    CX, CY = float(CX_), float(CY_)

    labels = json.load(open(labels_json))["labels"]
    for L in labels:  # raw clamped seed pixel (fractions 0..1 or absolute px)
        x = int(round(L["x"] * (W if L["x"] <= 1 else 1)))
        y = int(round(L["y"] * (H if L["y"] <= 1 else 1)))
        L["px"], L["py"] = min(max(x, 0), W - 1), min(max(y, 0), H - 1)

    min_area = 2.5e-4 * W * H

    # Stage / FOH is a solid dark block, not a priced tier, so it has no legend colour
    # and is handled apart from the palette pipeline: trace the dark blob at its seed
    # and emit it as a dark shape with a WHITE label. Seed it with id "STAGE" (or "FOH").
    stage_shapes, stage_texts, white_ids = [], [], set()
    is_stage = lambda L: str(L["id"]).strip().lower() in ("stage", "foh")
    if any(is_stage(L) for L in labels):
        dark = cv2.morphologyEx((cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 70)
                                .astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        _, dkl, dkst, _ = cv2.connectedComponentsWithStats(dark, 8)
        keep = []
        for L in labels:
            if is_stage(L):
                white_ids.add(L["id"])
                j = int(dkl[L["py"], L["px"]])
                # Only trace a SOLID dark block (a filled FOH box or a black stage).
                # A white stage drawn as a thin open outline has no fillable blob at its
                # seed -- tracing dark strokes yields a partial, wrong silhouette -- so we
                # emit only the (white) label and leave the block for the Figma edit pass.
                if j > 0 and dkst[j, 4] > min_area:
                    x0, y0, w0, h0, a0 = dkst[j]
                    if a0 / float(w0 * h0) > 0.35:      # solid, not a thin outline
                        cnts, _ = cv2.findContours((dkl == j).astype(np.uint8),
                                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if cnts:
                            poly = simplify(max(cnts, key=cv2.contourArea))
                            if len(poly) >= 3:
                                stage_shapes.append((L["id"], poly))
                stage_texts.append((L["id"], L["px"], L["py"]))
            else:
                keep.append(L)
        labels = keep

    # Real-divider pieces: per-colour connected sections. Where the drawn dividers
    # are visible, each section is its own piece with its TRUE shape.
    piece = -np.ones((H, W), np.int32); pid = 0
    percol = []  # cache raw per-colour masks for small-section recovery
    for ci in range(len(palette)):
        raw = ((lblmap == ci) & sect).astype(np.uint8)
        percol.append(raw)
        m = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        npc, ll, st, _ = cv2.connectedComponentsWithStats(m, 8)
        for j in range(1, npc):
            if st[j, 4] < min_area:
                continue
            piece[ll == j] = pid; pid += 1

    # Recover small user-seeded sections the area/open filter dropped: a seed sitting
    # on a coloured pixel with no piece gets its own piece from the raw same-colour
    # blob (no open), so a tiny real block (e.g. an arc end-cap) is never lost. A seed
    # is ground truth that the section exists, so it overrides the noise-area floor.
    for L in labels:
        x, y = L["px"], L["py"]
        if sect[y, x] and piece[y, x] < 0:
            ci = int(lblmap[y, x])
            npc, ll, _, _ = cv2.connectedComponentsWithStats(percol[ci], 8)
            j = ll[y, x]
            if j > 0 and bool((piece[ll == j] < 0).all()):
                piece[ll == j] = pid; pid += 1
    pys, pxs = np.where(piece >= 0)

    for L in labels:  # seed on a gap/stroke -> nearest section pixel
        x, y = L["px"], L["py"]
        if piece[y, x] < 0 and len(pxs):
            j = ((pxs - x) ** 2 + (pys - y) ** 2).argmin()
            L["px"], L["py"] = int(pxs[j]), int(pys[j])

    from collections import defaultdict
    by_piece = defaultdict(list)
    for i, L in enumerate(labels):
        by_piece[int(piece[L["py"], L["px"]])].append(i)

    # A piece with one label keeps its true shape; a piece with several merged
    # sections is split by its geometry: ARC pieces (seeds at ~one radius from the
    # bowl centre) split by ANGLE -> clean radial trapezoids (no diamonds); GRID
    # blocks split euclidean (Lloyd-relaxed).
    region_of = -np.ones((H, W), np.int32)
    for p, idxs in by_piece.items():
        ys, xs = np.where(piece == p)
        if len(idxs) == 1:
            region_of[ys, xs] = idxs[0]; continue
        sp = np.array([[labels[i]["px"], labels[i]["py"]] for i in idxs], float)
        r = np.hypot(sp[:, 0] - CX, sp[:, 1] - CY)
        if (r.max() - r.min()) < 0.45 * r.mean():   # arc-like
            a_s = np.arctan2(sp[:, 1] - CY, sp[:, 0] - CX)
            a_p = np.arctan2(ys - CY, xs - CX)
            nn = np.abs(np.angle(np.exp(1j * (a_p[:, None] - a_s[None])))).argmin(1)
        else:                                        # grid block -> Lloyd
            P = np.stack([xs, ys], 1)
            for _ in range(3):
                nn = (((P[:, None] - sp[None]) ** 2).sum(2)).argmin(1)
                for j in range(len(idxs)):
                    s = nn == j
                    if s.any():
                        sp[j] = [xs[s].mean(), ys[s].mean()]
            nn = (((P[:, None] - sp[None]) ** 2).sum(2)).argmin(1)
        for j, i in enumerate(idxs):
            region_of[ys[nn == j], xs[nn == j]] = i

    sections, texts = [], []
    dbg = img.copy()
    for i, L in enumerate(labels):
        region = (region_of == i).astype(np.uint8)
        area = int(region.sum())
        if area < 150:            # noise floor only; a seeded section is real
            continue
        # TRACE THE MASK. The colour segmentation already has the true section
        # outline; a light open + gaussian only removes pixel-edge jaggedness. We do
        # NOT hull/fill/parallelogram-snap it -- those "corrections" deviate from a
        # mask that is already right (e.g. they'd chamfer or over-fill a standing
        # pen's real stage-overlap notch). RETR_EXTERNAL already ignores the interior
        # number hole. Only a genuinely merged same-colour blob was split upstream.
        # Small sections get gentler smoothing so open+blur can't erase them.
        k, sig = (5, 2) if area > 1500 else (3, 1)
        proc = cv2.morphologyEx(region, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        proc = (cv2.GaussianBlur(proc * 255, (0, 0), sig) > 128).astype(np.uint8)
        cnts, _ = cv2.findContours(proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:              # smoothing erased a thin one -> trace it raw
            cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        poly = simplify(c)
        if len(poly) < 3:
            continue
        cat = int(np.bincount(lblmap[region > 0], minlength=len(palette)).argmax())
        sections.append((L["id"], poly, hexcol(palette[cat])))
        texts.append((L["id"], L["px"], L["py"]))
        cv2.polylines(dbg, [np.array(poly)], True, (0, 0, 0), 3)
        cv2.circle(dbg, (L["px"], L["py"]), 6, (0, 0, 255), -1)

    # Structure mirrors the known-good hand-built maps: one outer group wrapping a
    # sections group and a labels group ("sections grouped, texts grouped, both grouped
    # together"). Every <g> imports into Figma as a Group.
    # width/height as explicit attributes (not just viewBox): some seatmap readers
    # parse dimensions by string-matching width="..."/height="..." and break without them.
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}">']
    svg.append('<g id="seatmap">')
    svg.append('<g id="sections">')
    for sid, poly in stage_shapes:   # dark stage/FOH block(s), drawn first
        svg.append(f'<path id="{esc(sid)}" d="{poly_to_d(poly)}" fill="#111827"/>')
    for sid, poly, ccol in sections:
        svg.append(f'<path id="{esc(sid)}" data-cat-colour="{ccol}" d="{poly_to_d(poly)}" '
                   f'fill="#E9E9EC" stroke="#8A8A94" stroke-width="2"/>')
    svg.append('</g>')
    texts = texts + stage_texts      # stage labels ride along, coloured white below
    if outline_text:
        # Labels as vector outlines (paths) -- font-independent, render anywhere. Named
        # "Text_<id>" (NOT the bare section id) so they (a) never collide with the
        # section path of the same id and (b) are skipped by readers that ignore ids
        # containing "Text" -- this mirrors the known-good hand-built exports.
        svg.append('<g id="labels" fill="#222" fill-rule="evenodd">')
        seen = set()
        for sid, x, y in texts:
            if sid in seen:
                continue
            seen.add(sid)
            d = label_to_path(str(sid), x, y)
            if d:
                fill = ' fill="#FFFFFF"' if sid in white_ids else ''
                svg.append(f'<path id="Text_{esc(sid)}"{fill} d="{d}"/>')
    else:
        # Labels as live text -- editable, but depends on the viewer having the font
        # and supporting <text>. If your importer drops the labels, rebuild with
        # --outline-text. The layer id is prefixed "Text_" (not the bare section id) so
        # a seat-picker that recolours any id'd element on hover/click leaves the labels
        # alone (they commonly skip ids containing "Text") -- otherwise a label sharing
        # its section's id gets repainted (e.g. to white) and disappears on hover.
        svg.append('<g id="labels" font-family="Inter, Arial, sans-serif" '
                   'font-size="26" fill="#222" text-anchor="middle">')
        seen = set()
        for sid, x, y in texts:
            if sid in seen:      # one label per id (bands share an id)
                continue
            seen.add(sid)
            fill = ' fill="#FFFFFF"' if sid in white_ids else ''
            svg.append(f'<text id="Text_{esc(sid)}"{fill} x="{x}" y="{y+9}">'
                       f'{esc(sid)}</text>')
    svg.append('</g>')          # close labels
    svg.append('</g></svg>')    # close seatmap wrapper
    out = os.path.join(outdir, 'seatmap_figma.svg')
    open(out, 'w', encoding='utf-8').write(''.join(svg))
    sc = 1400 / W
    cv2.imwrite(os.path.join(outdir, 'build_debug.png'),
                cv2.resize(dbg, (1400, int(H * sc))))
    print(f"build: {len(sections)}/{len(labels)} sections placed -> {out}")
    print(f"  verify: {outdir}/build_debug.png")
    if len(sections) < len(labels):
        missing = len(labels) - len(sections)
        print(f"  {missing} seed(s) produced no region (seed off a section?) -- check preview.")

    # COVERAGE CHECK -- the reliability net. Look at the coloured pixels the build left
    # UNASSIGNED (region_of < 0) and flag any blob big enough to be a section: it is
    # either a section you forgot to label, or a second colour band of one you did (add
    # a seed with the SAME id -> two paths, one label). Working from unassigned pixels
    # directly -- not a re-segmentation -- means it can't be fooled by morphology
    # fragmenting an already-claimed strip, and a section traced onto the wrong blob is
    # caught because its true blob stays unassigned. Nothing coloured escapes review.
    cov_min = 0.45 * min_area   # catch small second-bands, not just full sections
    uncov = (sect & (region_of < 0)).astype(np.uint8)
    uncov = cv2.morphologyEx(uncov, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ncc, ll, st, cen = cv2.connectedComponentsWithStats(uncov, 8)
    over = img.copy(); k = 0; report = []
    for j in range(1, ncc):
        if st[j, 4] < cov_min:
            continue
        blob = ll == j
        ci = int(np.bincount(lblmap[blob], minlength=len(palette)).argmax())
        k += 1
        report.append((int(st[j, 4]), round(cen[j][0] / W, 3),
                       round(cen[j][1] / H, 3), hexcol(palette[ci])))
        cs, _ = cv2.findContours(blob.astype(np.uint8),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(over, cs, -1, (0, 0, 255), 4)
        cx, cy = int(cen[j][0]), int(cen[j][1])
        cv2.putText(over, str(k), (cx - 8, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
        cv2.putText(over, str(k), (cx - 8, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    if k:
        cv2.imwrite(os.path.join(outdir, 'uncovered.png'),
                    cv2.resize(over, (1400, int(H * sc))))
        print(f"  COVERAGE: {k} coloured blob(s) have NO seed -> {outdir}/uncovered.png")
        print(f"  (each = a section you missed, or a 2nd colour band; add a seed, "
              f"same id for a band). frac x,y,colour:")
        for a, fx, fy, col in sorted(report, reverse=True):
            print(f"    area={a:6d}  {col}  at ({fx},{fy})")
    else:
        print("  COVERAGE: OK -- every coloured section-sized blob is claimed by a seed.")


def esc(s):
    return (str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            .replace('"', '&quot;'))


def label_to_path(text, cx, cy, px_h=26, ss=4):
    """Outline a label string into a single SVG path `d` (letter holes preserved via
    even-odd), centred on (cx, cy) at ~px_h pixels tall. Uses only OpenCV's built-in
    vector font -- no external font file or extra dependency -- so the labels become
    geometry that renders in any importer, independent of installed fonts or <text>
    support. Returns '' if nothing rendered."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    H0 = int(px_h * ss)
    thick = max(2, int(round(H0 / 9)))
    fs = cv2.getFontScaleFromHeight(font, H0, thick)
    (tw, th), base = cv2.getTextSize(text, font, fs, thick)
    pad = thick * 3 + 6
    canvas = np.zeros((th + base + pad * 2, tw + pad * 2), np.uint8)
    cv2.putText(canvas, text, (pad, th + pad), font, fs, 255, thick, cv2.LINE_AA)
    ys, xs = np.where(canvas > 127)
    if len(xs) == 0:
        return ''
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    gcx, gcy = (x0 + x1) / 2.0, (y0 + y1) / 2.0     # glyph-ink centre in canvas px
    cnts, _ = cv2.findContours(canvas, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    d = []
    for c in cnts:
        c = cv2.approxPolyDP(c, 0.6, True)           # de-jag the AA edge, keep curves
        if len(c) < 3:
            continue
        pts = []
        for p in c.reshape(-1, 2):
            X = cx + (p[0] - gcx) / ss               # canvas px -> map px, ink-centred
            Y = cy + (p[1] - gcy) / ss
            pts.append(f'{X:.1f},{Y:.1f}')
        d.append('M' + ' L'.join(pts) + ' Z')
    return ' '.join(d)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('extract'); e.add_argument('png'); e.add_argument('outdir')
    b = sub.add_parser('build'); b.add_argument('png'); b.add_argument('labels')
    b.add_argument('outdir')
    b.add_argument('--outline-text', action='store_true',
                   help='emit labels as vector paths instead of <text> -- '
                        'font-independent, imports into any software')
    a = ap.parse_args()
    if a.cmd == 'extract':
        extract(a.png, a.outdir)
    else:
        build(a.png, a.labels, a.outdir, outline_text=a.outline_text)
