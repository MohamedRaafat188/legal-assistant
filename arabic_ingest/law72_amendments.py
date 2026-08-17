"""The amendment map for Law 72/2017, and the engine that splices it in.

Fourteen amendments, all read off the two Gazette PDFs and cross-checked
against the rasterized pages:

  Law 141/2019 (4)   adds a final paragraph to articles 12 and 48, adds
                     clause 14 to article 74, and adds article 91 مكررًا.
  Law 160/2023 (10)  replaces the الحوافز الخاصة definition inside article 1,
                     replaces articles 9, 13, 17, 20 and 34 whole, replaces
                     parts of articles 11, 12, 14 and 40, and adds article
                     11 مكررًا.

No article is repealed by either law, so the ملغاة branch never fires here.
It is implemented anyway, because the next law merged into this corpus may
well need it.

How an amended article reads
----------------------------
Following the requested convention, the consolidated text keeps the original
wording *and* the new wording, separated by a note naming the amending law,
so a lawyer can see what the provision said before and after:

  whole-article replacement
      <original text>
      استبدلت بالقانون رقم (١٦٠) لسنة (٢٠٢٣).
      <new text>

  partial replacement
      <untouched leading part>
      <original wording of the replaced part>
      استبدلت <part> بالقانون رقم (١٦٠) لسنة (٢٠٢٣).
      <new wording of that part>
      <untouched trailing part>

  addition
      <article text>
      مضافة بالقانون رقم (١٤١) لسنة (٢٠١٩).

  repeal
      <original text>
      ملغاة بالقانون رقم (N) لسنة (YYYY).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

AR = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


def ar(n: int) -> str:
    return str(n).translate(AR)


# --- amendment kinds --------------------------------------------------------
ADD_ARTICLE = "add_article"           # a brand-new article
REPLACE_ARTICLE = "replace_article"   # the whole article is substituted
REPLACE_PART = "replace_part"         # one paragraph/clause is substituted
APPEND_PART = "append_part"           # a paragraph/clause is added to an article
REPEAL = "repeal"                     # the article is struck out


@dataclass
class Amendment:
    law_number: int
    law_year: int
    kind: str
    target: int                       # article number in the base law
    target_suffix: str | None = None  # "مكرر" for the bis articles
    part_label: str | None = None     # e.g. "الفقرة الأولى - البند ١"
    anchor: str | None = None         # locates the replaced part in the original
    new_text: str = ""
    note_override: str | None = None

    def note(self) -> str:
        """The Arabic annotation inserted into the consolidated text."""
        if self.note_override:
            return self.note_override
        law = f"بالقانون رقم ({ar(self.law_number)}) لسنة ({ar(self.law_year)})"
        if self.kind in (ADD_ARTICLE, APPEND_PART):
            if not self.part_label:
                return f"مضافة {law}."
            # Agree with the part's gender: البند is masculine, الفقرة feminine.
            added = "مضاف" if self.part_label.startswith("البند") else "مضافة"
            return f"{self.part_label} {added} {law}."
        if self.kind == REPEAL:
            return f"ملغاة {law}."
        if self.kind == REPLACE_ARTICLE:
            return f"استبدلت {law}."
        return f"استبدلت {self.part_label} {law}."


@dataclass
class MergedArticle:
    number: int
    suffix: str | None
    text: str
    amendments: list[dict] = field(default_factory=list)
    status: str = "active"            # "active" | "repealed"


# --- splicing ---------------------------------------------------------------
def _normalize(s: str) -> str:
    """Whitespace/orthography-insensitive key for locating an anchor."""
    s = re.sub(r"\s+", " ", s)
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ى", "ي"), ("ة", "ه")):
        s = s.replace(a, b)
    return re.sub(r"[ً-ْ]", "", s).strip()


def find_anchor(body: str, anchor: str) -> tuple[int, int]:
    """Locate ``anchor`` inside ``body``, tolerant of wrapping and hamza style.

    Returns the (start, end) character offsets in the ORIGINAL ``body``.
    Raises if the anchor is absent or ambiguous -- silently splicing into the
    wrong place would corrupt an article, so this must fail loudly.
    """
    norm_body = _normalize(body)
    norm_anchor = _normalize(anchor)
    hits = [m.start() for m in re.finditer(re.escape(norm_anchor), norm_body)]
    if not hits:
        raise ValueError(f"anchor not found: {anchor[:60]!r}")
    if len(hits) > 1:
        raise ValueError(f"anchor is ambiguous ({len(hits)} matches): {anchor[:60]!r}")

    # Map the normalized offset back onto the original string by walking both
    # in step -- normalization only collapses whitespace and folds letters
    # one-for-one, so a running index stays aligned.
    target_start, target_end = hits[0], hits[0] + len(norm_anchor)
    orig_start = orig_end = None
    j = 0  # index into norm_body
    prev_space = False
    for i, ch in enumerate(body):
        if j == target_start and orig_start is None:
            orig_start = i
        if j == target_end and orig_end is None:
            orig_end = i
        if ch.isspace():
            if prev_space:
                continue
            prev_space = True
            j += 1
            continue
        prev_space = False
        if re.match(r"[ً-ْ]", ch):
            continue
        j += 1
    if orig_end is None:
        orig_end = len(body)
    if orig_start is None:
        raise ValueError(f"could not map anchor back to source: {anchor[:60]!r}")
    return orig_start, orig_end


def apply_amendments(
    number: int, body: str, amendments: list[Amendment], suffix: str | None = None
) -> MergedArticle:
    """Weave every amendment for one article into its consolidated text."""
    records: list[dict] = []
    status = "active"
    text = body.strip()

    # Part-level edits first (they anchor against the ORIGINAL wording, so they
    # must run before any whole-article note is appended), latest law last.
    part_edits = [a for a in amendments if a.kind in (REPLACE_PART,)]
    part_edits.sort(key=lambda a: (a.law_year, a.law_number))
    # Splice from the end backwards so earlier offsets stay valid.
    spans = []
    for amd in part_edits:
        start, end = find_anchor(text, amd.anchor or "")
        spans.append((start, end, amd))
    spans.sort(key=lambda s: s[0], reverse=True)
    for start, end, amd in spans:
        original = text[start:end].strip()
        replacement = f"{original}\n{amd.note()}\n{amd.new_text.strip()}"
        text = text[:start] + replacement + text[end:]
        records.append({
            "law_number": amd.law_number, "law_year": amd.law_year,
            "kind": amd.kind, "part": amd.part_label, "note": amd.note(),
        })

    for amd in sorted(
        (a for a in amendments if a.kind not in (REPLACE_PART,)),
        key=lambda a: (a.law_year, a.law_number),
    ):
        if amd.kind == REPLACE_ARTICLE:
            text = f"{text}\n{amd.note()}\n{amd.new_text.strip()}"
        elif amd.kind == APPEND_PART:
            text = f"{text}\n{amd.new_text.strip()}\n{amd.note()}"
        elif amd.kind == ADD_ARTICLE:
            text = f"{amd.new_text.strip()}\n{amd.note()}"
        elif amd.kind == REPEAL:
            text = f"{text}\n{amd.note()}"
            status = "repealed"
        records.append({
            "law_number": amd.law_number, "law_year": amd.law_year,
            "kind": amd.kind, "part": amd.part_label, "note": amd.note(),
        })

    return MergedArticle(number, suffix, text.strip(), records, status)
