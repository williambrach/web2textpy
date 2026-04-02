"""web2text.py — Python port of Web2Text alignment-based labeling pipeline.

Given paired (raw_html, clean_text), aligns clean text back onto DOM nodes
and labels each as content or boilerplate.

Based on: Vogels et al., "Web2Text: Deep Structured Boilerplate Removal" (ECIR 2018)
Original Scala: https://github.com/dalab/web2text
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from typing import NamedTuple

from lxml import etree
from lxml.html import document_fromstring

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GAPCHAR = "\u25a1"  # □

SKIP_TAGS = frozenset({
    "script", "style", "head", "noscript", "iframe", "img", "input",
    "br", "hr", "meta", "title", "video", "select", "textarea",
    "link", "object", "embed", "applet", "param", "svg",
})

BLOCK_TAGS = frozenset({
    "address", "article", "aside", "blockquote", "body", "center",
    "dd", "div", "dl", "dt", "fieldset", "figcaption", "figure",
    "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header",
    "li", "main", "nav", "ol", "p", "pre", "section", "table",
    "tbody", "td", "tfoot", "th", "thead", "tr", "ul",
})

# Characters treated as equivalent in alignment (apostrophes, quotes, ?)
_QUOTE_CHARS = frozenset("?'\u2018\u2019\u201a\u201b`\u0027")

# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Word-internal ? acting as apostrophe: letter?letter (e.g., it?s, don?t)
_QUESTION_APOS_RE = re.compile(r"(?<=\w)\?(?=\w)")


def _fix_mojibake(s: str) -> str:
    """Fix CP1252 mojibake: text that was UTF-8 but decoded as CP1252.

    Tries to re-encode as CP1252 and decode as UTF-8. If the result is
    shorter (mojibake expands chars), it's a genuine fix.
    """
    try:
        fixed = s.encode("cp1252").decode("utf-8")
        if len(fixed) < len(s):  # mojibake always inflates length
            return fixed
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    return s


def normalize_text(s: str) -> str:
    """NFC-normalize, fix mojibake, collapse whitespace, replace NBSP, strip."""
    s = _fix_mojibake(s)
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\u00a0", " ")
    return _WS_RE.sub(" ", s).strip()


def _normalize_for_eval(s: str) -> str:
    """Extra normalization applied only during evaluation.

    Normalizes smart quotes and ?-as-apostrophe so that ground truth
    quirks don't penalize correct extraction.
    """
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = _QUESTION_APOS_RE.sub("'", s)
    return s


# ---------------------------------------------------------------------------
# CDOM construction
# ---------------------------------------------------------------------------

def _remove_preserving_tail(el: etree._Element) -> None:
    """Remove *el* from its parent without losing its tail text."""
    parent = el.getparent()
    if parent is None:
        return
    tail = el.tail
    if tail:
        prev = el.getprevious()
        if prev is not None:
            prev.tail = (prev.tail or "") + tail
        else:
            parent.text = (parent.text or "") + tail
    parent.remove(el)


_XML_DECL_RE = re.compile(r"<\?xml[^\n]*(?:\?>|\n|$)", re.IGNORECASE)


def build_cdom(html_str: str) -> etree._Element:
    """Parse HTML and build a Collapsed DOM tree.

    Removes non-content elements, empty nodes, and collapses single-child chains.
    """
    # Strip XML declaration that lxml.html rejects
    html_str = _XML_DECL_RE.sub("", html_str, count=1)

    doc = document_fromstring(html_str)
    try:
        body = doc.body
    except IndexError:
        body = None
    if body is None:
        body = doc

    # --- Strip XML-invalid control characters from all text/tail/attributes ---
    for el in doc.iter():
        if el.text:
            el.text = _CTRL_RE.sub("", el.text)
        if el.tail:
            el.tail = _CTRL_RE.sub("", el.tail)
        if isinstance(el.tag, str):
            for attr, val in el.attrib.items():
                cleaned = _CTRL_RE.sub("", val)
                if cleaned != val:
                    el.attrib[attr] = cleaned

    # --- Remove comments ---
    for c in body.iter():
        if callable(c.tag):  # Comments, PIs
            _remove_preserving_tail(c)

    # --- Remove skip-tag elements ---
    to_remove = [el for el in body.iter() if isinstance(el.tag, str) and el.tag in SKIP_TAGS]
    for el in to_remove:
        _remove_preserving_tail(el)

    # --- Remove empty text (set .text / .tail to None if whitespace-only) ---
    for el in body.iter():
        if not isinstance(el.tag, str):
            continue
        if el.text and not el.text.strip():
            el.text = None
        if el.tail and not el.tail.strip():
            el.tail = None

    # --- Remove empty leaf elements (bottom-up) ---
    changed = True
    while changed:
        changed = False
        for el in list(body.iter()):
            if not isinstance(el.tag, str):
                continue
            if el is body:
                continue
            if len(el) == 0 and not (el.text and el.text.strip()):
                _remove_preserving_tail(el)
                changed = True

    # --- Collapse single-child chains (bottom-up via post-order) ---
    def _collapse(el: etree._Element) -> None:
        for child in list(el):
            if isinstance(child.tag, str):
                _collapse(child)
        if len(el) == 1 and not (el.text and el.text.strip()):
            child = el[0]
            if not isinstance(child.tag, str):
                return
            # Merge child into parent: adopt child's children and text
            el.text = child.text
            # Move grandchildren up
            grandchildren = list(child)
            for gc in grandchildren:
                child.remove(gc)
                el.append(gc)
            # Preserve child.tail: append it to the last grandchild's tail,
            # or to el.text if there are no grandchildren
            if child.tail and child.tail.strip():
                if grandchildren:
                    last_gc = grandchildren[-1]
                    last_gc.tail = (last_gc.tail or "") + child.tail
                else:
                    el.text = (el.text or "") + child.tail
            el.remove(child)

    _collapse(body)

    return body


# ---------------------------------------------------------------------------
# Leaf extraction
# ---------------------------------------------------------------------------

def extract_leaves(tree: etree._Element) -> list[tuple[etree._Element, str]]:
    """Extract ordered text blocks from the CDOM tree.

    In lxml, text lives in two places: el.text (before first child) and
    child.tail (after a child element). Both must be captured, mirroring
    how Jsoup represents text as separate TextNode children.

    For leaf elements (no children), we capture el.text_content() as usual.
    For internal elements, we capture el.text as a separate text block,
    and each child's .tail as a separate text block.
    """
    leaves: list[tuple[etree._Element, str]] = []

    def _walk(el: etree._Element) -> None:
        if not isinstance(el.tag, str):
            return
        if el.get("data-synthetic"):
            return  # already processed
        if len(el) == 0:
            # Leaf element — capture all its text
            text = normalize_text(el.text_content())
            if text:
                el.set("data-leaf-id", str(len(leaves)))
                leaves.append((el, text))
        else:
            # Snapshot original children BEFORE any tree mutations
            original_children = list(el)

            # Internal element — capture .text (text before first child)
            if el.text and el.text.strip():
                text = normalize_text(el.text)
                if text:
                    span = etree.SubElement(el, "span")
                    span.set("data-synthetic", "1")
                    span.text = el.text
                    el.text = None
                    el.insert(0, span)
                    span.set("data-leaf-id", str(len(leaves)))
                    leaves.append((span, text))

            # Recurse into original children only
            for child in original_children:
                _walk(child)
                # Capture child.tail (text after this child, before next sibling)
                if child.tail and child.tail.strip():
                    tail_text = normalize_text(child.tail)
                    if tail_text:
                        span = etree.SubElement(el, "span")
                        span.set("data-synthetic", "1")
                        span.text = child.tail
                        child.tail = None
                        idx = list(el).index(child)
                        el.insert(idx + 1, span)
                        span.set("data-leaf-id", str(len(leaves)))
                        leaves.append((span, tail_text))

    _walk(tree)
    return leaves


# ---------------------------------------------------------------------------
# Anchor-based alignment  (port of Scala find1to1mathches)
# ---------------------------------------------------------------------------

class Segment(NamedTuple):
    src_start: int
    src_end: int
    cln_start: int
    cln_end: int
    matched: bool


def _find_anchors(source: str, cleaned: str, k: int = 10) -> list[Segment]:
    """Find unique k-char anchors shared between *source* and *cleaned*.

    Returns a list of Segments covering the full source/cleaned range.
    Matched segments are definite alignments; open segments need DP.
    """
    n, m = len(source), len(cleaned)

    if n < k or m < k:
        return [Segment(0, n, 0, m, False)]

    # Build substring → [positions] map for source (skip substrings with GAPCHAR)
    source_map: dict[str, list[int]] = defaultdict(list)
    for i in range(n + 1 - k):
        sub = source[i : i + k]
        if GAPCHAR not in sub:
            source_map[sub].append(i)

    # Trimmed source maps for safety check (strip whitespace + GAPCHAR)
    def _trim_filter(c: str) -> bool:
        return not c.isspace() and c != GAPCHAR

    trimmed_source = "".join(c for c in source if _trim_filter(c))
    # Pre-compute occurrence counts for trimmed substrings of lengths 1..k
    trimmed_maps: list[dict[str, int]] = []
    for kk in range(1, k + 1):
        counts: dict[str, int] = defaultdict(int)
        for i in range(len(trimmed_source) + 1 - kk):
            counts[trimmed_source[i : i + kk]] += 1
        trimmed_maps.append(counts)

    def _equal_enough(c1: str, c2: str) -> bool:
        if c1.isspace() and c2.isspace():
            return True
        if c1.upper() == c2.upper():
            return True
        # Treat ? as equivalent to apostrophe variants (l3s-gn1 ground truth quirk)
        if c1 in _QUOTE_CHARS and c2 in _QUOTE_CHARS:
            return True
        return False

    segments: list[Segment] = []
    # Track last matched/open segment end positions
    last_src_start, last_src_end = 0, 0
    last_cln_start, last_cln_end = 0, 0
    last_is_init = True  # first sentinel

    i = 0
    while i < m + 1 - k:
        subs = cleaned[i : i + k]

        # Trimmed substring for safety check
        trimmed_subs = "".join(c for c in subs if _trim_filter(c))

        match_locs = source_map.get(subs, [])
        trimmed_count = (
            trimmed_maps[len(trimmed_subs) - 1].get(trimmed_subs, 0)
            if trimmed_subs
            else 0
        )

        if len(match_locs) == 1 and trimmed_count == 1:
            src_pos = match_locs[0]

            # Extend right
            extra_right = 0
            while (
                i + k + extra_right < m
                and src_pos + k + extra_right < n
                and _equal_enough(cleaned[i + k + extra_right], source[src_pos + k + extra_right])
            ):
                extra_right += 1

            # Extend left
            extra_left = 0
            while (
                i - extra_left > 0
                and src_pos - extra_left > 0
                and src_pos - extra_left >= last_src_end + 1
                and _equal_enough(cleaned[i - 1 - extra_left], source[src_pos - 1 - extra_left])
            ):
                extra_left += 1

            if src_pos <= last_src_start and not last_is_init:
                # Collision: new match is before previous — discard last 2 segments
                if len(segments) >= 2:
                    segments.pop()
                    segments.pop()
                    if segments:
                        prev = segments[-1]
                        last_src_start, last_src_end = prev.src_start, prev.src_end
                        last_cln_start, last_cln_end = prev.cln_start, prev.cln_end
                    else:
                        last_src_start = last_src_end = 0
                        last_cln_start = last_cln_end = 0
                        last_is_init = True
                i += 1

            elif src_pos < last_src_end:
                # Overlap — skip this anchor
                i += 1

            else:
                last_is_init = False
                # Shorten left extension if it overlaps with previous segment
                while src_pos - extra_left < last_src_end:
                    extra_left -= 1

                match_src_start = src_pos - extra_left
                match_src_end = src_pos + k + extra_right
                match_cln_start = i - extra_left
                match_cln_end = i + k + extra_right

                # Insert open segment before this match (if there's a gap)
                if match_src_start > last_src_end or match_cln_start > last_cln_end:
                    segments.append(
                        Segment(last_src_end, match_src_start, last_cln_end, match_cln_start, False)
                    )

                # Insert matched segment
                seg = Segment(match_src_start, match_src_end, match_cln_start, match_cln_end, True)
                segments.append(seg)
                last_src_start, last_src_end = seg.src_start, seg.src_end
                last_cln_start, last_cln_end = seg.cln_start, seg.cln_end

                i += k + extra_right
        else:
            i += 1

    # Append final open segment if source/cleaned has remaining chars
    if segments:
        last = segments[-1]
        if last.src_end < n or last.cln_end < m:
            segments.append(Segment(last.src_end, n, last.cln_end, m, False))
    else:
        segments.append(Segment(0, n, 0, m, False))

    return segments


# ---------------------------------------------------------------------------
# Dynamic-programming alignment  (port of Scala dpalignment)
# ---------------------------------------------------------------------------

# Decision enum
_MATCH, _SKIP_SRC, _SKIP_CLN = 0, 1, 2


def _dp_align(source: str, cleaned: str) -> str:
    """Align *cleaned* against *source* using DP with affine gap penalties.

    Returns a string of len(source) with GAPCHAR at non-content positions.
    """
    n, m = len(source), len(cleaned)

    if n == 0:
        return ""
    if m == 0:
        return GAPCHAR * n

    # Size guard — avoid quadratic blowup on huge unmatched segments.
    # For large segments, align in chunks rather than giving up entirely.
    if n * m > 10_000_000:
        # Chunk the source into manageable pieces and align each
        chunk_size = max(1, 10_000_000 // m)
        result_parts: list[str] = []
        cln_pos = 0
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            src_chunk = source[start:end]
            # Estimate how much clean text corresponds to this chunk
            src_fraction = (end - start) / n
            overshoot_factor = 1.5
            cln_chunk_size = int(m * src_fraction * overshoot_factor)
            cln_end = min(cln_pos + cln_chunk_size, m)
            cln_chunk = cleaned[cln_pos:cln_end]
            chunk_result = _dp_align(src_chunk, cln_chunk)
            # Advance clean pointer by how many chars were consumed
            consumed = sum(1 for c in chunk_result if c != GAPCHAR)
            cln_pos += consumed
            result_parts.append(chunk_result)
        return "".join(result_parts)

    # Score matrices (2-row rolling for space efficiency)
    S = [[0] * (m + 1) for _ in range(2)]
    G = [[True] * (m + 1) for _ in range(2)]
    D = [[_SKIP_SRC] * (m + 1) for _ in range(n + 1)]

    # Initialize first row
    for j in range(m + 1):
        S[0][j] = -j
        G[0][j] = True
        D[0][j] = _SKIP_CLN
    D[0][0] = -1  # sentinel

    for i in range(1, n + 1):
        ci = i % 2
        pi = (i - 1) % 2
        S[ci][0] = 0
        G[ci][0] = True

        for j in range(1, m + 1):
            sc = source[i - 1]
            cc = cleaned[j - 1]

            # SkipClean score
            skip_cln_score = S[ci][j - 1] + (0 if cc.isspace() else -6)

            # SkipSource score
            skip_src_score = S[pi][j] + (0 if G[pi][j] else -2)

            if skip_cln_score > skip_src_score:
                best_score = skip_cln_score
                best_dec = _SKIP_CLN
            else:
                best_score = skip_src_score
                best_dec = _SKIP_SRC

            # Match (only if chars are compatible)
            chars_compatible = (
                sc.upper() == cc.upper()
                or (sc.isspace() and cc.isspace())
                or (sc in _QUOTE_CHARS and cc in _QUOTE_CHARS)
            )
            if chars_compatible:
                if sc.isalnum() and sc == cc:
                    match_score = S[pi][j - 1] + 3
                else:
                    match_score = S[pi][j - 1] + 1
                if match_score > best_score:
                    best_score = match_score
                    best_dec = _MATCH

            D[i][j] = best_dec
            S[ci][j] = best_score

            if best_dec == _MATCH:
                G[ci][j] = False
            elif best_dec == _SKIP_SRC:
                G[ci][j] = True
            else:  # _SKIP_CLN
                G[ci][j] = G[ci][j - 1]

    # Backtrack
    result: list[str] = []
    i, j = n, m
    while i > 0 or j > 0:
        d = D[i][j]
        if d == _MATCH:
            result.append(source[i - 1])
            i -= 1
            j -= 1
        elif d == _SKIP_CLN:
            j -= 1
        else:  # _SKIP_SRC
            result.append(GAPCHAR)
            i -= 1

    result.reverse()
    assert len(result) == n, f"DP output length {len(result)} != source length {n}"
    return "".join(result)


# ---------------------------------------------------------------------------
# Alignment orchestrator
# ---------------------------------------------------------------------------

def align(
    leaves: list[tuple[etree._Element, str]], clean_text: str
) -> dict[int, float]:
    """Align leaf texts against *clean_text* and return per-leaf content scores.

    Returns {leaf_index: fraction_of_chars_matched} (0.0–1.0).
    """
    if not leaves or not clean_text.strip():
        return {i: 0.0 for i in range(len(leaves))}

    # Build source by concatenating leaf texts with space separators
    parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    pos = 0
    for idx, (el, text) in enumerate(leaves):
        offsets.append((pos, pos + len(text)))
        parts.append(text)
        pos += len(text)
        if idx < len(leaves) - 1:
            parts.append(" ")
            pos += 1
    source = "".join(parts)

    cleaned = normalize_text(clean_text)
    if not cleaned:
        return {i: 0.0 for i in range(len(leaves))}

    # Phase 1: anchor matching
    segments = _find_anchors(source, cleaned, k=10)

    # Phase 2: DP on open segments, pass-through on matched segments
    aligned_parts: list[str] = []
    for seg in segments:
        src_slice = source[seg.src_start : seg.src_end]
        cln_slice = cleaned[seg.cln_start : seg.cln_end]
        if seg.matched:
            aligned_parts.append(cln_slice)
        else:
            aligned_parts.append(_dp_align(src_slice, cln_slice))
    aligned = "".join(aligned_parts)

    # Extract per-leaf scores
    scores: dict[int, float] = {}
    for i, (start, end) in enumerate(offsets):
        if end <= len(aligned):
            substr = aligned[start:end]
            n_matched = sum(1 for c in substr if c != GAPCHAR)
        else:
            n_matched = 0
        n_total = end - start
        scores[i] = n_matched / n_total if n_total > 0 else 0.0

    # Fallback pass: leaves with score 0 that have substantial text might
    # have been missed due to ordering differences between DOM and clean text.
    # Try direct substring matching against the full clean text.
    cleaned_lower = cleaned.lower()
    for i, (el, text) in enumerate(leaves):
        if scores[i] > 0.0 or len(text) < 20:
            continue
        leaf_lower = text.lower()
        if leaf_lower in cleaned_lower:
            scores[i] = 1.0
        elif len(text) >= 50:
            # Try matching 50-char chunks to estimate content fraction
            chunk_size = 50
            matched_chars = 0
            for c in range(0, len(text) - chunk_size + 1, chunk_size):
                if text[c : c + chunk_size].lower() in cleaned_lower:
                    matched_chars += chunk_size
            scores[i] = matched_chars / len(text) if len(text) > 0 else 0.0

    return scores


# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------

def label_nodes(
    tree: etree._Element,
    scores: dict[int, float],
    threshold: float = 0.667,
) -> etree._Element:
    """Label CDOM nodes as content or boilerplate based on alignment scores.

    Threshold 0.667 matches the original Scala's 2/3 rule.
    """
    # Label leaves
    for el in tree.iter():
        if not isinstance(el.tag, str):
            continue
        leaf_id = el.get("data-leaf-id")
        if leaf_id is not None:
            score = scores.get(int(leaf_id), 0.0)
            el.set("data-label", "content" if score > threshold else "boilerplate")

    # Propagate upward: internal node is content if any child is content
    def _propagate(el: etree._Element) -> bool:
        if not isinstance(el.tag, str):
            return False
        if el.get("data-label"):
            return el.get("data-label") == "content"
        has_content = any(_propagate(child) for child in el)
        el.set("data-label", "content" if has_content else "boilerplate")
        return has_content

    _propagate(tree)
    return tree


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text(tree: etree._Element) -> str:
    """Reconstruct clean text from content-labeled leaf nodes."""
    parts: list[str] = []
    for el in tree.iter():
        if not isinstance(el.tag, str):
            continue
        if el.get("data-label") == "content" and len(el) == 0:
            text = normalize_text(el.text_content())
            if text:
                parts.append(text)
    return "\n".join(parts)


def extract_text_from_labeled_html(labeled_html: str) -> str:
    """Extract content text from an already-labeled HTML string.

    Accepts HTML where elements carry ``data-label="content"`` or
    ``data-label="boilerplate"`` attributes (e.g. the ``labeled_html``
    column produced by :func:`label_original_html`) and returns only
    the text of content-labeled leaf nodes.
    """
    doc = document_fromstring(labeled_html)
    return extract_text(doc)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(extracted: str, ground_truth: str) -> dict:
    """Compute quality metrics: token F1, ROUGE-1, BLEU, CHRF.

    Both inputs are normalized for fair comparison (smart quotes, ?-apostrophes).
    ROUGE-L is skipped as it's O(n*m) and very slow on long texts.
    """
    try:
        import sacrebleu
        from rouge_score import rouge_scorer
    except ImportError:
        raise ImportError(
            "evaluate() requires optional dependencies. "
            "Install them with: uv add 'web2textpy[eval]'"
        )

    # Normalize both sides for fair comparison
    extracted = _normalize_for_eval(extracted)
    ground_truth = _normalize_for_eval(ground_truth)

    # Token-level F1 (multiset)
    ext_tokens = Counter(extracted.lower().split())
    gt_tokens = Counter(ground_truth.lower().split())
    overlap = sum((ext_tokens & gt_tokens).values())
    ext_total = sum(ext_tokens.values())
    gt_total = sum(gt_tokens.values())
    precision = overlap / ext_total if ext_total else 0.0
    recall = overlap / gt_total if gt_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # ROUGE-1 only (ROUGE-L is O(n*m) and very slow on long texts)
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=False)
    rouge = scorer.score(ground_truth, extracted)

    # BLEU & CHRF
    bleu = sacrebleu.corpus_bleu([extracted], [[ground_truth]])
    chrf = sacrebleu.corpus_chrf([extracted], [[ground_truth]])

    return {
        "token_f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "rouge1_f": round(rouge["rouge1"].fmeasure, 4),
        "bleu": round(bleu.score, 2),
        "chrf": round(chrf.score, 2),
    }


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    html_str: str, clean_text: str
) -> tuple[etree._Element, str, dict]:
    """Run the full Web2Text alignment pipeline.

    Returns (labeled_tree, extracted_text, metrics).
    """
    tree = build_cdom(html_str)
    leaves = extract_leaves(tree)
    scores = align(leaves, clean_text)
    tree = label_nodes(tree, scores)
    extracted = extract_text(tree)
    metrics = evaluate(extracted, clean_text)
    return tree, extracted, metrics


# ---------------------------------------------------------------------------
# Label original HTML
# ---------------------------------------------------------------------------

def label_original_html(
    html_str: str, clean_text: str, threshold: float = 0.667,
) -> tuple[str, str, dict]:
    """Run pipeline and return the *original* HTML with data-label attributes.

    Unlike run_pipeline (which returns the collapsed CDOM), this preserves
    the full original document structure and annotates every element with
    data-label="content" or data-label="boilerplate".

    Returns (labeled_html_string, extracted_text, metrics).
    """
    import copy

    # --- Parse & clean (same steps as build_cdom) ---
    cleaned_str = _XML_DECL_RE.sub("", html_str, count=1)
    doc = document_fromstring(cleaned_str)

    for el in doc.iter():
        if el.text:
            el.text = _CTRL_RE.sub("", el.text)
        if el.tail:
            el.tail = _CTRL_RE.sub("", el.tail)
        if isinstance(el.tag, str):
            for attr, val in el.attrib.items():
                c = _CTRL_RE.sub("", val)
                if c != val:
                    el.attrib[attr] = c

    try:
        body = doc.body
    except IndexError:
        body = None
    if body is None:
        body = doc

    # --- Assign stable IDs before any tree modifications ---
    _id = 0
    for el in body.iter():
        if isinstance(el.tag, str):
            el.set("data-orig-id", str(_id))
            _id += 1

    # --- Deep-copy: this is the "original" we will label at the end ---
    orig_doc = copy.deepcopy(doc)

    # --- CDOM construction (mirrors build_cdom logic) ---
    # Remove comments
    for c in body.iter():
        if callable(c.tag):
            _remove_preserving_tail(c)

    # Remove skip-tag elements
    for el in [e for e in body.iter() if isinstance(e.tag, str) and e.tag in SKIP_TAGS]:
        _remove_preserving_tail(el)

    # Remove empty text
    for el in body.iter():
        if not isinstance(el.tag, str):
            continue
        if el.text and not el.text.strip():
            el.text = None
        if el.tail and not el.tail.strip():
            el.tail = None

    # Remove empty leaf elements
    changed = True
    while changed:
        changed = False
        for el in list(body.iter()):
            if not isinstance(el.tag, str) or el is body:
                continue
            if len(el) == 0 and not (el.text and el.text.strip()):
                _remove_preserving_tail(el)
                changed = True

    # Collapse single-child chains (tracking merged orig-ids)
    def _collapse_tracking(el: etree._Element) -> None:
        for child in list(el):
            if isinstance(child.tag, str):
                _collapse_tracking(child)
        if len(el) == 1 and not (el.text and el.text.strip()):
            child = el[0]
            if not isinstance(child.tag, str):
                return
            # Track which orig-ids get absorbed
            child_oid = child.get("data-orig-id", "")
            child_merged = child.get("data-merged-ids", "")
            existing = el.get("data-merged-ids", "")
            all_ids = [existing, child_oid, child_merged]
            non_empty_ids = filter(None, all_ids)
            merged = ",".join(non_empty_ids)
            if merged:
                el.set("data-merged-ids", merged)
            el.text = child.text
            grandchildren = list(child)
            for gc in grandchildren:
                child.remove(gc)
                el.append(gc)
            if child.tail and child.tail.strip():
                if grandchildren:
                    grandchildren[-1].tail = (grandchildren[-1].tail or "") + child.tail
                else:
                    el.text = (el.text or "") + child.tail
            el.remove(child)

    _collapse_tracking(body)

    # --- Run alignment pipeline on the CDOM ---
    leaves = extract_leaves(body)
    scores = align(leaves, clean_text)
    label_nodes(body, scores, threshold)
    extracted = extract_text(body)
    metrics = evaluate(extracted, clean_text)

    # --- Build orig-id → label mapping ---
    label_map: dict[str, str] = {}
    for el in body.iter():
        if not isinstance(el.tag, str):
            continue
        label = el.get("data-label")
        if not label:
            continue
        oid = el.get("data-orig-id")
        if oid:
            label_map[oid] = label
        merged = el.get("data-merged-ids")
        if merged:
            for mid in merged.split(","):
                if mid:
                    label_map[mid] = label

    # --- Apply labels to original doc ---
    try:
        orig_body = orig_doc.body
    except IndexError:
        orig_body = None
    if orig_body is None:
        orig_body = orig_doc

    for el in orig_body.iter():
        if not isinstance(el.tag, str):
            continue
        oid = el.get("data-orig-id")
        if oid and oid in label_map:
            el.set("data-label", label_map[oid])
        elif isinstance(el.tag, str) and el.tag in SKIP_TAGS:
            el.set("data-label", "boilerplate")
        # Clean up tracking attributes
        for attr in ("data-orig-id",):
            if attr in el.attrib:
                del el.attrib[attr]

    # Propagate: unlabeled nodes inherit from children
    def _propagate(el: etree._Element) -> bool:
        if not isinstance(el.tag, str):
            return False
        if el.get("data-label"):
            return el.get("data-label") == "content"
        has_content = any(_propagate(child) for child in el)
        el.set("data-label", "content" if has_content else "boilerplate")
        return has_content

    _propagate(orig_body)

    # Label <head> as boilerplate
    try:
        head = orig_doc.head
        if head is not None and not head.get("data-label"):
            head.set("data-label", "boilerplate")
    except Exception:
        pass

    labeled_html = etree.tostring(orig_doc, encoding="unicode", method="html")
    return labeled_html, extracted, metrics
