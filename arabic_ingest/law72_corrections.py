"""Reviewed typo table for the Law 72/2017 source PDF.

The 72/2017 PDF in `data/` is an unofficial *retyped* copy, not a Gazette
scan. Unlike the two amendment PDFs (which are clean official texts), it
carries roughly eighty word-level defects: transposed letters, dropped
letters, and pairs of words run together with no space. Every one of them was
confirmed against the rasterized page, so they are defects of the source
document itself, not of extraction.

This table follows the precedent set by `arabic_ingest/glyph_repair.py`: a
hand-audited, explicitly enumerated correction set rather than a heuristic.
Nothing here is guessed -- each entry is a word whose intended reading is
unambiguous from the surrounding legal sentence, and the anchored entries
carry enough context that they cannot fire anywhere else.

What is deliberately NOT corrected:
  * Hamza-seat inconsistency (الأستثمار / الإستثمار / الاستثمار). The source
    is inconsistent, the retrieval copy folds hamza anyway, and "fixing" it
    would be a style edit, not a defect repair.
  * Foreign trade terms transliterated into Arabic ("سيف" = CIF,
    "فوب" = FOB). These are correct legal usage.
  * Substantive errors that cannot be repaired without the official text --
    see SUBSTANTIVE_DEFECTS. Those are reported, never silently rewritten.
"""
from __future__ import annotations

import re

# --- words run together (missing space) -------------------------------------
MERGED_WORDS: dict[str, str] = {
    "القانونالضوابط": "القانون الضوابط",
    "مجلسالوزراء": "مجلس الوزراء",
    "أومشروعاتالمشاركة": "أو مشروعات المشاركة",
    "تفصيليةمحددا": "تفصيلية محددا",
    "ولاتتقيد": "ولا تتقيد",
    "الاتزيد": "ألا تزيد",
    "الايجاوز": "ألا يجاوز",
    "لا قامة": "لإقامة",
    "أ وفساد": "أو فساد",
    "ار بيان": "أى بيان",
    "ابدا الراى": "إبداء الرأى",
}

# --- single-word letter defects ---------------------------------------------
# Each key is a token that appears nowhere in correct Arabic; the value is the
# reading the sentence requires. Applied on whole-token boundaries only.
WORD_FIXES: dict[str, str] = {
    # dropped or transposed letters
    "الاأئتمانية": "الائتمانية",
    "الاقنون": "القانون",
    "القاتون": "القانون",
    "القانوان": "القانون",
    "القوانيين": "القوانين",
    "الغنتاجية": "الإنتاجية",
    "الكتاليف": "التكاليف",
    "التاليف": "التكاليف",
    "اللاتزام": "الالتزام",
    "اللشخص": "الشخص",
    "المؤسيي": "المؤسسي",
    "المنطورة": "المنظورة",
    "المنقطة": "المنطقة",
    "المرافثق": "المرافق",
    "المشرعات": "المشروعات",
    "المللية": "الملكية",
    "الممختص": "المختص",
    "الوزارت": "الوزارات",
    "الوزراة": "الوزارة",
    "الوزرءا": "الوزراء",
    "الرمسية": "الرسمية",
    "بالمتلك": "بالتملك",
    "بذء": "بدء",
    "ترتكتب": "ترتكب",
    "تلشريعات": "التشريعات",
    "جمروكية": "جمركية",
    "زفقا": "وفقا",
    "صدورة": "صدوره",
    "طبثا": "طبقا",
    "غخطار": "إخطار",
    "غدارة": "إدارة",
    "قمية": "قيمة",
    "معاكملته": "معاملته",
    "مصاردتها": "مصادرتها",
    "ومستحضرارات": "ومستحضرات",
    "الأسخاص": "الأشخاص",
    "للمشىروعات": "للمشروعات",
    "والإعفااءت": "والإعفاءات",
    "إلية": "إليه",
    "هذذه": "هذه",
    "هذة": "هذه",
    "والاستمثارية": "والاستثمارية",
    "والزير": "والوزير",
    "ووضائفهم": "ووظائفهم",
    "وتوفيير": "وتوفير",
    "ويجمتع": "ويجتمع",
    "تاريه": "تاريخ",
    "تقدين": "تقديم",
    "للقواعج": "للقواعد",
    "مكاتنب": "مكاتب",
    "محالفة": "مخالفة",
    "صاحية": "صاحبة",
    "الباين": "البابين",
    "السوطاء": "الوسطاء",
    "الزاحم": "التزاحم",
    "خالة": "حالة",
    "انشظة": "أنشطة",
    "بالحوافظ": "بالحوافز",
    "الوارة": "الواردة",
    "الإعفااءات": "الإعفاءات",
    "حاجه": "حاجة",
    "وآالية": "وآلية",
    "اتعباهم": "أتعابهم",
    "الإدراية": "الإدارية",
    "تبنية": "تبينه",
    "واتخذا": "واتخاذ",
    "للتويج": "للترويج",
    "وادارتة": "وإدارته",
    "وردو": "ورود",
    "المشا": "المشار",
    "لمهيمنة": "المهيمنة",
    "واسترددها": "واستردادها",
    "الإحصائات": "الإحصاءات",
    "وضوابطة": "وضوابطه",
    "فقا": "وفقا",
}

# --- defects needing sentence context to disambiguate -----------------------
# The bare token is legitimate elsewhere (or too short to key on safely), so
# these carry enough of the surrounding clause to fire exactly once.
# Matching is whitespace-flexible: the source wraps mid-phrase, so a fixed
# space would miss the occurrences that straddle a line break.
ANCHORED_FIXES: list[tuple[str, str, str]] = [
    # (phrase to find, replacement, what changed)
    ("الاكثر منت تاريخ", "الاكثر من تاريخ", "منت -> من"),
    ("هلال مدة لا تجاوز", "خلال مدة لا تجاوز", "هلال -> خلال"),
    ("مركز مستقب للتحكيم", "مركز مستقل للتحكيم", "مستقب -> مستقل"),
    ("رقم ١٠٨ لسمة", "رقم ١٠٨ لسنة", "لسمة -> لسنة"),
    ("فسه عقد البيع", "فسخ عقد البيع", "فسه -> فسخ"),
    ("ودة سريانه", "ومدة سريانه", "ودة -> ومدة"),
    ("و الإ يجوز رفع دعوى", "ولا يجوز رفع دعوى", "و الإ -> ولا"),
    ("المناطق الرحة", "المناطق الحرة", "الرحة -> الحرة"),
    ("إىل ادخل البلاد", "إلى داخل البلاد", "إىل ادخل -> إلى داخل"),
    ("خروجها من المنطقة الحرة غىل داخل البلاد",
     "خروجها من المنطقة الحرة إلى داخل البلاد", "غىل -> إلى"),
    ("واستثناء منذ ذلك", "واستثناء من ذلك", "منذ -> من"),
    ("تكفل تطبق مبادئ", "تكفل تطبيق مبادئ", "تطبق -> تطبيق"),
    ("قرارا ببناء على عرض", "قرارا بناء على عرض", "ببناء -> بناء"),
    ("وتبين اللائحتين التنفيذية", "وتبين اللائحة التنفيذية", "اللائحتين -> اللائحة"),
    ("للشركات والمنشآت الخاصة لأحكام هذا القانون",
     "للشركات والمنشآت الخاضعة لأحكام هذا القانون", "الخاصة -> الخاضعة"),
    ("بالحوافز المنصوص فى المواد", "بالحوافز المنصوص عليها فى المواد", "restore dropped عليها"),
    ("المواد )-١١-١٠ (١٢", "المواد (١٠-١١-١٢)", "malformed bracketed article list"),
    ("الاستثمارى (٨٠)%) من رأس المال", "الاستثمارى (٨٠%) من رأس المال", "stray closing paren"),
    ("للقطاعين(، أ)و(ب)", "للقطاعين (أ) و(ب)", "mirrored paren with stray comma"),
    ("بالقطاعين (أ)و (ب)", "بالقطاعين (أ) و(ب)", "spacing around the enumerators"),
]

# --- reported, never rewritten ----------------------------------------------
# Substantive defects: the source states something legally wrong. Repairing
# these means writing law, so they are surfaced for a human instead.
SUBSTANTIVE_DEFECTS: list[tuple[str, str]] = [
    (
        "الهيئة : الوزارة المختصة بشئون",
        "Definition of الهيئة repeats the definition of الوزارة المختصة. "
        "The enacted text defines it as الهيئة العامة للاستثمار والمناطق الحرة.",
    ),
    (
        "المستثمر : استخدام كل شخص",
        "Definition of المستثمر opens with a stray 'استخدام' carried over "
        "from the preceding definition; the enacted text begins 'كل شخص'.",
    ),
    (
        "(٨٠)%)",
        "Malformed percentage bracket in article 11; the enacted text reads (٨٠٪).",
    ),
]


def apply_corrections(text: str) -> tuple[str, list[tuple[str, str, int]]]:
    """Apply the reviewed table, returning the text and a per-fix hit log."""
    log: list[tuple[str, str, int]] = []

    for find, repl, _why in ANCHORED_FIXES:
        pattern = re.compile(r"\s+".join(re.escape(w) for w in find.split()))
        text, n = pattern.subn(repl, text)
        if n:
            log.append((find, repl, n))

    for table in (MERGED_WORDS, WORD_FIXES):
        for bad, good in table.items():
            # Whole-token match so a correct longer word is never touched.
            pattern = re.compile(rf"(?<![ء-ي]){re.escape(bad)}(?![ء-ي])")
            text, n = pattern.subn(good, text)
            if n:
                log.append((bad, good, n))

    return text, log


def find_substantive_defects(text: str) -> list[tuple[str, str]]:
    """Return the substantive defects actually present, for the audit report."""
    return [(marker, note) for marker, note in SUBSTANTIVE_DEFECTS if marker in text]


# --- structural reconstruction ---------------------------------------------
"""Structural reconstruction of the article 32/33 boundary.

The 72/2017 source PDF loses three things at this seam, all confirmed against
the rasterized page 22:

  1. the tail of article 32's penultimate paragraph -- it stops mid-phrase at
     "العمل بهذا", dropping "النظام."
  2. article 32's final paragraph, about the executive regulation, entirely
  3. the "الفصل الرابع / نظام الاستثمار في المناطق الحرة" heading and the
     "مادة (٣٣)" header -- so article 33's text runs on inside article 32 and
     the law appears to jump from chapter 3 to chapter 5

The replacement text was supplied by the project owner. It was diffed
sentence-by-sentence against the extraction (see `diff_32_33.py`); apart from
punctuation and hamza styling, the only differences were exactly the three
gaps above, so the patch below is deliberately surgical -- everything already
extracted and verified stays untouched.
"""

SEAM_BEFORE = "العمل بهذا\n\nيكون إنشاء المنطقة الحرة"

SEAM_AFTER = """العمل بهذا النظام.

وتبين اللائحة التنفيذية لهذا القانون اشتراطات وضوابط العمل فيها وأسلوب إدارتها.

الفصل الرابع

نظام الاستثمار في المناطق الحرة

مادة "٣٣":

يكون إنشاء المنطقة الحرة"""

PATCHES: list[tuple[str, str, str]] = [
    (
        SEAM_BEFORE,
        SEAM_AFTER,
        "Restored article 32's dropped tail and final paragraph, the "
        "الفصل الرابع heading, and the missing مادة (٣٣) header.",
    ),
]


def apply_patches(text: str) -> tuple[str, list[str]]:
    """Apply every structural patch, returning the text and a description log."""
    log: list[str] = []
    for before, after, why in PATCHES:
        count = text.count(before)
        if count != 1:
            raise ValueError(
                f"patch anchor matched {count} times, expected exactly 1: {before[:40]!r}"
            )
        text = text.replace(before, after)
        log.append(why)
    return text, log
