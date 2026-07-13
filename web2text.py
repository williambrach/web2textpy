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
    """Reconstruct clean text from content-labeled *leaf* nodes.

    Pipeline-internal: this is correct only on a tree that has gone through
    :func:`extract_leaves`, which splits internal-node text and child tails into
    synthetic ``<span>`` leaves. On such a tree every text block is a leaf, so
    the ``len(el) == 0`` rule captures all content. Do NOT call this directly on
    a freshly parsed labeled document — internal-node prose is not a leaf there
    and would be dropped; use :func:`extract_text_direct` /
    :func:`extract_text_from_labeled_html` instead.
    """
    parts: list[str] = []
    for el in tree.iter():
        if not isinstance(el.tag, str):
            continue
        if el.get("data-label") == "content" and len(el) == 0:
            text = normalize_text(el.text_content())
            if text:
                parts.append(text)
    return "\n".join(parts)


def _append_own_text(parts: list[str], el: etree._Element) -> None:
    """Append a content node's OWN text -- ``el.text`` then ``el.tail`` -- to *parts*.

    ``el.text`` is the text before the node's first child; ``el.tail`` is the
    text after the node, in the parent's flow. Each is stripped, normalized, and
    skipped when empty. No subtree recursion: descendants contribute their own
    text only when they too are content-labeled and visited in their own right.
    """
    for segment in (el.text, el.tail):
        if segment and segment.strip():
            text = normalize_text(segment)
            if text:
                parts.append(text)


def extract_text_direct(tree: etree._Element) -> str:
    """Reconstruct content text from an already-labeled tree using direct node text.

    Emits each content-labeled node's own ``el.text`` + ``el.tail`` in document
    order, without recursing into subtrees. Unlike :func:`extract_text` (leaf
    only, pipeline-internal), this captures article prose living on internal /
    inline nodes: in ``<p>before <a>link</a> after.</p>`` the ``before`` is
    ``p.text`` and ``after.`` is ``a.tail``, both recovered. Boilerplate-labeled
    descendants of a content node are excluded.
    """
    parts: list[str] = []
    for el in tree.iter():
        if not isinstance(el.tag, str):
            continue
        if el.get("data-label") == "content":
            _append_own_text(parts, el)
    return "\n".join(parts)


def extract_text_from_labeled_html(labeled_html: str) -> str:
    """Extract content text from an already-labeled HTML string.

    Accepts HTML where elements carry ``data-label="content"`` or
    ``data-label="boilerplate"`` attributes (e.g. the ``labeled_html`` column
    produced by :func:`label_original_html`) and returns the text of
    content-labeled nodes via :func:`extract_text_direct` -- i.e. each content
    node's own ``el.text`` + ``el.tail`` in document order. (Previously this
    emitted only content-labeled *leaf* nodes, which dropped the substantial
    fraction of article prose that lives on internal/inline DOM nodes.)
    """
    doc = document_fromstring(labeled_html)
    return extract_text_direct(doc)


def extract_text_from_nodes(nodes: list[etree._Element], labels: list[int]) -> str:
    """Direct content text from parallel lists of DOM nodes and 0/1 labels.

    For callers that already hold classified lxml elements (e.g. a GNN node
    classifier) rather than a labeled HTML string. Emits each content-labeled
    (``label == 1``) node's own ``el.text`` + ``el.tail`` in list order, matching
    :func:`extract_text_direct`. ``nodes`` and ``labels`` must be equal length
    and in document order.
    """
    if not nodes:
        return ""
    parts: list[str] = []
    for el, label in zip(nodes, labels, strict=True):
        if isinstance(el.tag, str) and label == 1:
            _append_own_text(parts, el)
    return "\n".join(parts)


# Inline-level tags whose text stays merged with the surrounding prose (never a
# block boundary). Used by the doc-adaptive recovery below.
INLINE_TAGS = frozenset({
    "a", "b", "i", "em", "strong", "span", "u", "small", "sub", "sup", "mark",
    "abbr", "cite", "code", "q", "s", "time", "label", "bdi", "bdo", "wbr",
    "font", "tt", "kbd", "var",
})


def _inline_recovery_parts(el: etree._Element) -> list[str]:
    """Text of a content node *el* recovering inline-descendant text + all child tails.

    Emits ``el.text``, then for every descendant: the text of INLINE children
    (recursively) and the ``.tail`` of EVERY child (inline and block — the
    block-child tails carry the prose on container-only-labeled docs), then
    ``el.tail``. Block children's own ``.text`` is reached when they are content
    nodes in their own right.
    """
    parts: list[str] = []
    if el.text and el.text.strip():
        parts.append(el.text.strip())

    def walk(node: etree._Element) -> None:
        for ch in node:
            if not isinstance(ch.tag, str):
                if ch.tail and ch.tail.strip():
                    parts.append(ch.tail.strip())
                continue
            if ch.tag in INLINE_TAGS:
                if ch.text and ch.text.strip():
                    parts.append(ch.text.strip())
                walk(ch)
            if ch.tail and ch.tail.strip():
                parts.append(ch.tail.strip())

    walk(el)
    if el.tail and el.tail.strip():
        parts.append(el.tail.strip())
    return parts


def _content_subtree_words(nodes: list[etree._Element], labels: list[int]) -> int:
    """Total itertext word count of the maximal content subtrees (content roots)."""
    id2idx = {id(n): i for i, n in enumerate(nodes)}

    def has_content_ancestor(i: int) -> bool:
        p = nodes[i].getparent()
        while p is not None:
            j = id2idx.get(id(p))
            if j is not None and labels[j] == 1:
                return True
            p = p.getparent()
        return False

    total = 0
    for i, lab in enumerate(labels):
        if lab == 1 and not has_content_ancestor(i):
            total += len("".join(nodes[i].itertext()).split())
    return total


def extract_text_from_nodes_adaptive(
    nodes: list[etree._Element], labels: list[int], threshold: float = 0.40
) -> str:
    """Doc-adaptive content text: precise own-text by default, recover when labels are coarse.

    Per document, compares the words :func:`extract_text_from_nodes` (variant D)
    would emit against the total word mass of the content subtrees. If that ratio
    is below ``threshold`` the labels mark only *containers* (the prose lives in
    unlabeled descendants), so this recurses to recover it
    (:func:`_inline_recovery_parts`); otherwise it returns the precise variant-D
    text. The per-document routing keeps precision high on fine-labeled pages
    while rescuing the recall-collapsed container-only pages.

    Same ``(nodes, labels)`` contract as :func:`extract_text_from_nodes`; needs
    no gold and no model change (a binary node classifier's labels suffice).
    """
    if not nodes:
        return ""
    d_parts: list[str] = []
    for el, label in zip(nodes, labels, strict=True):
        if isinstance(el.tag, str) and label == 1:
            for seg in (el.text, el.tail):
                if seg and seg.strip():
                    d_parts.append(seg.strip())
    d_words = sum(len(s.split()) for s in d_parts)
    subtree_words = _content_subtree_words(nodes, labels)
    ratio = d_words / subtree_words if subtree_words > 0 else 1.0
    if ratio < threshold:
        parts: list[str] = []
        for el, label in zip(nodes, labels, strict=True):
            if isinstance(el.tag, str) and label == 1:
                parts.extend(_inline_recovery_parts(el))
        return "\n".join(parts)
    return "\n".join(d_parts)


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

def disambiguate_content_by_gold(
    root: etree._Element, clean_text: str, min_tokens: int = 4, frac: float = 0.8
) -> int:
    """Gold-guided own-text label repair (build-time): flip prose nodes to content.

    The CDOM->original label transfer in :func:`label_original_html` lands content
    labels on empty container nodes while the prose-bearing original node is left
    boilerplate/unlabeled (measured: ~71% of the worst docs' gold-text mass ends up
    on boilerplate-labeled originals). This pass repairs that: for every node NOT
    already labeled ``content``, it tokenizes the node's OWN text (``el.text`` +
    ``el.tail``) and re-labels it ``content`` when it has at least ``min_tokens``
    tokens and at least ``frac`` of them appear in the gold token set. The
    ``min_tokens`` guard avoids spuriously flipping short generic strings (e.g.
    a lone "By") that happen to be gold members.

    Gold-guided is acceptable here because the whole dataset is a pseudo-label set
    built FROM the gold (``align`` is gold-guided too); the rule only re-labels
    nodes whose own text the gold already contains, injecting no out-of-document
    signal, and the GNN never sees the gold at inference. Returns #nodes flipped.

    Run AFTER the apply-to-original loop and BEFORE the upward _propagate so the
    newly-content nodes propagate to their containers.
    """
    gold_tokens = set(_normalize_for_eval(clean_text).lower().split())
    if not gold_tokens:
        return 0
    flipped = 0
    for el in root.iter():
        if not isinstance(el.tag, str) or el.get("data-label") == "content":
            continue
        own = (el.text or "") + " " + (el.tail or "")
        toks = _normalize_for_eval(own).lower().split()
        if len(toks) < min_tokens:
            continue
        if sum(1 for t in toks if t in gold_tokens) / len(toks) >= frac:
            el.set("data-label", "content")
            flipped += 1
    return flipped


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

    # Gold-guided own-text disambiguation: rescue prose-bearing nodes the
    # CDOM->original label transfer left as boilerplate/unlabeled. Runs before
    # _propagate so the newly-content nodes bubble up to their containers.
    disambiguate_content_by_gold(orig_body, clean_text)

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
