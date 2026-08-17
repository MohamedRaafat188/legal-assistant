"""The concrete amendment map: what each amending law changes, and where.

New text is read out of the amending laws' own extractions rather than being
retyped here, so it stays byte-identical to the Gazette. Anchors -- the spans
of ORIGINAL wording that a partial replacement supersedes -- are given as
short quotations; `find_anchor` matches them ignoring wrapping and hamza
styling, and raises if a quotation is missing or ambiguous.

Every entry was read off the rasterized Gazette pages, not inferred:
  141/2019 المادة الأولى  -> page 1
  160/2023 المادة الأولى  -> pages 1-5, المادة الثانية -> pages 5-6
"""
from __future__ import annotations

import re

from arabic_text import format_clause_structure  # noqa: E402
from law72_amendments import (
    ADD_ARTICLE,
    APPEND_PART,
    REPLACE_ARTICLE,
    REPLACE_PART,
    Amendment,
)
from law72_extract import BUILD_DIR  # noqa: E402


def _load(name: str) -> str:
    text = (BUILD_DIR / name).read_text(encoding="utf-8")
    return re.sub(r"===== PAGE \d+ =====\n?", "", text)


def _between(text: str, start: str, end: str | None) -> str:
    """The body between two headers, formatted like a base-law article body.

    Flattened and re-broken on بند boundaries by the same
    `format_clause_structure` the base articles go through, so the new wording
    and the wording it replaces read identically in the merged text.
    """
    i = text.index(start) + len(start)
    j = text.index(end, i) if end else len(text)
    body = re.sub(r"\s+", " ", text[i:j]).strip()
    # 141/2019 marks its new clause with an en-dash ("– ١٤ طلب"), which
    # `format_clause_structure` does not treat as a بند marker. Canonicalise
    # it to the "١٤- " form the rest of the corpus uses.
    body = re.sub(r"^[–—-]\s*([٠-٩]+)\s+", r"\1- ", body)
    return format_clause_structure(body)


def build() -> list[Amendment]:
    a141 = _load("amend141_fixed.txt")
    a160 = _load("amend160_fixed.txt")

    # ---- Law 141 of 2019 -------------------------------------------------
    p12 = _between(a141, "مادة /١٢ فقرة أخيرة:", "مادة /٤٨ فقرة أخيرة:")
    p48 = _between(a141, "مادة /٤٨ فقرة أخيرة:", "مادة /٧٤ بند ١٤:")
    p74 = _between(a141, "مادة /٧٤ بند ١٤:", "مادة ٩١ مكررًا:")
    p91 = _between(a141, "مادة ٩١ مكررًا:", "المادة الثانية")

    # ---- Law 160 of 2023 -------------------------------------------------
    d1 = _between(a160, "مادة (١): (تعريف الحوافز الخاصة )", "مادة (٩):")
    m9 = _between(a160, "مادة (٩):", "مادة (/١١ فقرة أولى -بند )١ :")
    m11 = _between(a160, "مادة (/١١ فقرة أولى -بند )١ :", "مادة (/١٢ فقرة أولى -بند )٢ :")
    m12 = _between(a160, "مادة (/١٢ فقرة أولى -بند )٢ :", "مادة (١٣)")
    m13 = _between(a160, "مادة (١٣)", "مادة (١٤ /فقرة أولى):")
    m14 = _between(a160, "مادة (١٤ /فقرة أولى):", "مادة (١٧):")
    m17 = _between(a160, "مادة (١٧):", "مادة (٢٠):")
    m20 = _between(a160, "مادة (٢٠):", "مادة (٣٤):")
    m34 = _between(a160, "مادة (٣٤):", "مادة (/٤٠ فقرتان ثانية وثالثة):")
    m40 = _between(a160, "مادة (/٤٠ فقرتان ثانية وثالثة):", "(المادة الثانية)")
    m11bis = _between(a160, "مادة (١١ مكررا):", "( المادة الثالثة )")

    L141 = dict(law_number=141, law_year=2019)
    L160 = dict(law_number=160, law_year=2023)

    return [
        # --- 141/2019: four additions ---
        Amendment(**L141, kind=APPEND_PART, target=12,
                  part_label="فقرة أخيرة", new_text=p12),
        Amendment(**L141, kind=APPEND_PART, target=48,
                  part_label="فقرة أخيرة", new_text=p48),
        Amendment(**L141, kind=APPEND_PART, target=74,
                  part_label="البند (١٤)", new_text=p74),
        Amendment(**L141, kind=ADD_ARTICLE, target=91, target_suffix="مكرر",
                  new_text=p91),

        # --- 160/2023: one definition, five whole articles, four parts,
        #     and one new article ---
        Amendment(**L160, kind=REPLACE_PART, target=1,
                  part_label="تعريف الحوافز الخاصة",
                  anchor='الحوافز الخاصة: الحوافز المنصوص عليها في المادة "١١" من هذا القانون.',
                  new_text=d1),
        Amendment(**L160, kind=REPLACE_ARTICLE, target=9, new_text=m9),
        Amendment(**L160, kind=REPLACE_PART, target=11,
                  part_label="الفقرة الأولى - البند (١)",
                  anchor="١- نسبة (٥٠%) خصما من التكاليف الاستثمارية للقطاع (أ): ويشمل "
                         "المناطق الجغرافية الأكثر احتياجا للتنمية طبقا للخريطة الاستثمارية "
                         "وبناء على البيانات والإحصاءات الصادرة من الجهاز المركزى للتعبئة "
                         "العامة والاحصاء، ووفقا لتوزيع أنشطة الاستثمار بها على النحو الذى "
                         "تبينه اللائحة التنفيذية الاستثمارية للقطاع (ب) .",
                  new_text=m11),
        Amendment(**L160, kind=REPLACE_PART, target=12,
                  part_label="الفقرة الأولى - البند (٢)",
                  anchor="٢- أن تؤسس الشركة أو المنشأة خلال مدة اقصاها ثلاث سنوات من تاريخ "
                         "العمل باللائحة التنفيذية لهذا القانون ويجوز بقرار من مجلس الوزراء "
                         "وبناء على عرض الوزير المختص مد هذه المدة لمرة واحدة.",
                  new_text=m12),
        Amendment(**L160, kind=REPLACE_ARTICLE, target=13, new_text=m13),
        Amendment(**L160, kind=REPLACE_PART, target=14,
                  part_label="الفقرة الأولى",
                  anchor="يختص الرئيس التنفيذى للهيئة أو من يفوضه بإصدار الشهادة اللازمة "
                         "للتمتع بالحوافز المنصوص عليها فى المواد (١٠-١١-١٢) للشركات "
                         "والمنشآت الخاضعة لأحكام هذا القانون",
                  new_text=m14),
        Amendment(**L160, kind=REPLACE_ARTICLE, target=17, new_text=m17),
        Amendment(**L160, kind=REPLACE_ARTICLE, target=20, new_text=m20),
        Amendment(**L160, kind=REPLACE_ARTICLE, target=34, new_text=m34),
        Amendment(**L160, kind=REPLACE_PART, target=40,
                  part_label="الفقرتان الثانية والثالثة",  # noqa: E501 - matches the Gazette's own wording
                  anchor="واستثناء من ذلك يسمح بدخول المواد والنفايات والمخلفات الناتجة عن "
                         "أنشطة المشروعات العاملة بالمناطق الحرة إلى داخل البلاد متى كان "
                         "دخولها إلى البلاد بغرض التخلص منها أو إعادة تدويرها وذلك بالطرق "
                         "والوسائل الآمنة المقررة وفقاً لقانون البيئة الصادر بالقانون رقم ٤ "
                         "لسنة ١٩٩٤، على نفقة صاحب الشأن. وتطبق أحكام قانون البيئة المشار "
                         "اليه فى شان حظر استيراد النفايات الخطرة من الخارج.",
                  new_text=m40),
        Amendment(**L160, kind=ADD_ARTICLE, target=11, target_suffix="مكرر",
                  new_text=m11bis),
    ]
