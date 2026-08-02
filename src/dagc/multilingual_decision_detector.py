"""
multilingual_decision_detector.py

Lightweight, zero-model, zero-download extension for decision-verb detection
across non-English input.

Why this shape (vs. neural MT):
  - opus-mt-mul-en / NLLB etc. are 300MB-1.5GB, need torch, add real latency,
    and are a dependency people will just disable in prod.
  - This module is pure stdlib + re. Total lexicon data is a few dozen KB.
    No download, no GPU/CPU model inference, sub-millisecond per clause.

Two independent layers, both feeding into your EXISTING English decision logic:

  1. MULTILINGUAL_PATTERNS
     A merged bank of decision/request regexes across ~20 languages
     (want/would-like/please+verb/instead/cancel/change/add/could-you/etc,
     equivalents to what you already have for English). Run directly —
     no language ID needed to use this layer, since regex cost is trivial
     even run in full every time.

  2. lightweight_translate()
     A word/phrase-level dictionary swap (NOT sentence MT). Replaces known
     foreign decision-vocabulary tokens with their English gloss and leaves
     everything else untouched. The result is then re-checked with your
     REAL english_decision_fn callback, so all your existing English-side
     nuance (implicit patterns, commissive detection, scoring, etc.)
     applies unchanged to translated non-English input.

Language ID (detect_language) is included ONLY for logging/diagnostics.
It is never used to gate detection, so a wrong guess can't cause a miss.
"""

from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Optional


# --------------------------------------------------------------------------
# 1. Merged multilingual decision-phrase pattern bank
#    (extend freely — each entry is (lang_tag, compiled_regex))
# --------------------------------------------------------------------------

_RAW_PATTERNS: dict[str, list[str]] = {
    "en": [
        r"\bi'?d like\b", r"\bi would like\b", r"\bi want\b", r"\bi need\b",
        r"\bplease (change|cancel|add|book|exchange|return|update|remove)\b",
        r"\binstead\b", r"\bcan (i|you) (do|help|try)\b", r"\bcould you\b",
    ],
    "it": [  # Italian
        r"\bvorrei\b", r"\bmi servirebbe\b", r"\bho bisogno di\b", r"\bdevo\b",
        r"\bper favore\b", r"\bpotrebbe(sti)?\b", r"\binvece\b", r"\bcambiare\b",
        r"\bannullare\b", r"\baggiungere\b",
    ],
    "fr": [  # French
        r"\bje voudrais\b", r"\bj'aimerais\b", r"\bj'ai besoin de\b",
        r"\bs'il vous pla[iî]t\b", r"\bpourriez[- ]vous\b", r"\bpouvez[- ]vous\b",
        r"\bau lieu\b", r"\bannuler\b", r"\bchanger\b", r"\bajouter\b",
    ],
    "es": [  # Spanish
        r"\bquisiera\b", r"\bquiero\b", r"\bnecesito\b", r"\bpor favor\b",
        r"\bpodr[ií]a(s)?\b", r"\ben (su|vez) lugar\b", r"\bcancelar\b",
        r"\bcambiar\b", r"\ba[ñn]adir\b",
    ],
    "pt": [  # Portuguese
        r"\bgostaria\b", r"\bquero\b", r"\bpreciso\b", r"\bpor favor\b",
        r"\bpoderia\b", r"\bem vez disso\b", r"\bcancelar\b", r"\bmudar\b",
        r"\badicionar\b",
    ],
    "de": [  # German
        r"\bich möchte\b", r"\bich hätte gerne\b", r"\bich brauche\b",
        r"\bbitte\b", r"\bkönnten sie\b", r"\bstattdessen\b", r"\bstornieren\b",
        r"\bändern\b", r"\bhinzufügen\b",
    ],
    "nl": [  # Dutch
        r"\bik zou graag\b", r"\bik wil\b", r"\bik heb .* nodig\b",
        r"\balsjeblieft\b", r"\bkunt u\b", r"\bin plaats van\b", r"\bannuleren\b",
        r"\bwijzigen\b", r"\btoevoegen\b",
    ],
    "ru": [  # Russian
        r"\bя хочу\b", r"\bя хотел(а)? бы\b", r"\bмне нужно\b",
        r"\bпожалуйста\b", r"\bне могли бы вы\b", r"\bвместо\b",
        r"\bотменить\b", r"\bизменить\b", r"\bдобавить\b",
    ],
    "ar": [  # Arabic
        r"أريد", r"أرغب", r"من فضلك", r"بدلاً من ذلك", r"هل يمكنك",
        r"إلغاء", r"تغيير", r"إضافة",
    ],
    "hi": [  # Hindi
        r"मुझे चाहिए", r"मैं चाहता हूँ", r"मैं चाहती हूँ", r"कृपया",
        r"क्या आप", r"इसके बजाय", r"रद्द करें", r"बदलें", r"जोड़ें",
    ],
    "zh": [  # Chinese
        r"我想要", r"我需要", r"请帮我", r"请", r"能不能", r"可以帮我",
        r"取消", r"更改", r"改成", r"添加", r"代替",
    ],
    "ja": [  # Japanese
        r"お願いします", r"したいです", r"必要です", r"できますか",
        r"代わりに", r"キャンセル", r"変更", r"追加",
    ],
    "ko": [  # Korean
        r"하고 싶어요", r"필요해요", r"부탁드립니다", r"대신에", r"취소",
        r"변경", r"추가", r"해주실 수 있나요",
    ],
    "tr": [  # Turkish
        r"\bistiyorum\b", r"\bisterim\b", r"\blütfen\b", r"\byerine\b",
        r"\biptal\b", r"\bdeğiştir\b", r"\bekle(r|yebilir)? misiniz\b",
    ],
    "pl": [  # Polish
        r"\bchciałbym\b", r"\bchciałabym\b", r"\bpotrzebuję\b", r"\bproszę\b",
        r"\bzamiast\b", r"\banulować\b", r"\bzmienić\b", r"\bdodać\b",
    ],
    "vi": [  # Vietnamese
        r"\btôi muốn\b", r"\btôi cần\b", r"\blàm ơn\b", r"\bthay vào đó\b",
        r"\bhủy\b", r"\bthay đổi\b", r"\bthêm\b",
    ],
    "th": [  # Thai
        r"ฉันต้องการ", r"กรุณา", r"แทน", r"ยกเลิก", r"เปลี่ยน", r"เพิ่ม",
    ],
    "id": [  # Indonesian/Malay
        r"\bsaya ingin\b", r"\bsaya mau\b", r"\bsaya butuh\b", r"\btolong\b",
        r"\bsebagai gantinya\b", r"\bbatalkan\b", r"\bubah\b", r"\btambahkan\b",
    ],
    "sv": [  # Swedish
        r"\bjag skulle vilja\b", r"\bjag vill\b", r"\bjag behöver\b",
        r"\bsnälla\b", r"\bistället\b", r"\bavboka\b", r"\bändra\b", r"\blägga till\b",
    ],
    "el": [  # Greek
        r"θα ήθελα", r"θέλω", r"χρειάζομαι", r"παρακαλώ", r"αντ' αυτού",
        r"ακύρωση", r"αλλαγή", r"προσθήκη",
    ],
    "he": [  # Hebrew
        r"אני רוצה", r"אני צריך", r"בבקשה", r"במקום זאת", r"לבטל",
        r"לשנות", r"להוסיף",
    ],
}

MULTILINGUAL_PATTERNS: list[tuple[str, re.Pattern]] = [
    (lang, re.compile(pat, re.IGNORECASE))
    for lang, pats in _RAW_PATTERNS.items()
    for pat in pats
]


# --------------------------------------------------------------------------
# 2. Word/phrase-level dictionary "translator" (not sentence MT)
#    Longest-match-first substitution so multi-word phrases win over
#    single words they contain.
# --------------------------------------------------------------------------

_WORD_DICT: dict[str, str] = {
    # Italian
    "vorrei": "i would like", "mi servirebbe": "i need", "ho bisogno di": "i need",
    "per favore": "please", "invece": "instead", "cambiare": "change",
    "annullare": "cancel", "aggiungere": "add", "potrebbe": "could you",
    # French
    "je voudrais": "i would like", "j'aimerais": "i would like",
    "j'ai besoin de": "i need", "s'il vous plaît": "please",
    "au lieu": "instead", "annuler": "cancel", "changer": "change",
    "ajouter": "add", "pourriez-vous": "could you",
    # Spanish
    "quisiera": "i would like", "quiero": "i want", "necesito": "i need",
    "por favor": "please", "cancelar": "cancel", "cambiar": "change",
    "añadir": "add", "podría": "could you",
    # Portuguese
    "gostaria": "i would like", "preciso": "i need", "mudar": "change",
    "adicionar": "add",
    # German
    "ich möchte": "i would like", "ich brauche": "i need", "bitte": "please",
    "stattdessen": "instead", "stornieren": "cancel", "ändern": "change",
    "hinzufügen": "add",
    # Dutch
    "ik zou graag": "i would like", "ik wil": "i want", "alsjeblieft": "please",
    "annuleren": "cancel", "wijzigen": "change", "toevoegen": "add",
    # Russian
    "я хочу": "i want", "я хотел бы": "i would like", "мне нужно": "i need",
    "пожалуйста": "please", "вместо": "instead", "отменить": "cancel",
    "изменить": "change", "добавить": "add",
    # Turkish
    "istiyorum": "i want", "lütfen": "please", "iptal": "cancel",
    "değiştir": "change",
    # Polish
    "chciałbym": "i would like", "potrzebuję": "i need", "proszę": "please",
    "zmienić": "change", "dodać": "add",
    # Vietnamese / Indonesian / Swedish
    "tôi muốn": "i want", "làm ơn": "please", "hủy": "cancel",
    "saya ingin": "i want", "tolong": "please", "batalkan": "cancel",
    "jag skulle vilja": "i would like", "snälla": "please", "avboka": "cancel",
    # CJK / Arabic / Hindi / Thai / Greek / Hebrew (word-level, script-based, no spaces needed)
    "我想要": "i want", "我需要": "i need", "请帮我": "please help me",
    "取消": "cancel", "更改": "change", "添加": "add",
    "したいです": "i want", "必要です": "i need", "キャンセル": "cancel",
    "하고 싶어요": "i want", "필요해요": "i need", "취소": "cancel",
    "أريد": "i want", "من فضلك": "please", "إلغاء": "cancel",
    "मुझे चाहिए": "i need", "कृपया": "please", "रद्द करें": "cancel",
    "ฉันต้องการ": "i want", "กรุณา": "please", "ยกเลิก": "cancel",
    "θα ήθελα": "i would like", "παρακαλώ": "please", "ακύρωση": "cancel",
    "אני רוצה": "i want", "בבקשה": "please", "לבטל": "cancel",
}

# Longest phrases first, so "je voudrais" matches before a stray "je" would.
_SORTED_DICT_KEYS = sorted(_WORD_DICT.keys(), key=len, reverse=True)


def lightweight_translate(text: str) -> str:
    """
    Word/phrase-level gloss substitution — NOT full machine translation.
    Replaces recognized decision-vocabulary spans with their English
    equivalent; everything else in the sentence is left as-is.

    This is intentionally shallow: the goal isn't a fluent English sentence,
    it's to surface enough English decision-verb signal that your EXISTING
    english_decision_fn picks it up without any change to that function.
    """
    out = text
    for phrase in _SORTED_DICT_KEYS:
        if phrase in out.lower():
            # case-insensitive replace, phrase boundaries only for latin-script
            # (CJK/Arabic/etc. entries have no word-boundary concept, so plain substring is fine)
            if re.match(r"^[\w\s'-]+$", phrase):
                out = re.sub(re.escape(phrase), _WORD_DICT[phrase], out, flags=re.IGNORECASE)
            else:
                out = out.replace(phrase, _WORD_DICT[phrase])
    return out


# --------------------------------------------------------------------------
# 3. Optional language ID — diagnostics only, never gates detection
# --------------------------------------------------------------------------

_SCRIPT_RANGES = [
    ("zh", (0x4E00, 0x9FFF)),   # CJK ideographs (fallback bucket for zh/ja kanji)
    ("ja", (0x3040, 0x30FF)),   # Hiragana + Katakana -> Japanese overrides zh
    ("ko", (0xAC00, 0xD7A3)),   # Hangul syllables
    ("ar", (0x0600, 0x06FF)),   # Arabic
    ("hi", (0x0900, 0x097F)),   # Devanagari
    ("th", (0x0E00, 0x0E7F)),   # Thai
    ("el", (0x0370, 0x03FF)),   # Greek
    ("he", (0x0590, 0x05FF)),   # Hebrew
    ("ru", (0x0400, 0x04FF)),   # Cyrillic
]

_LATIN_STOPWORDS = {
    "it": {"vorrei", "grazie", "per", "favore", "invece", "servirebbe"},
    "fr": {"je", "voudrais", "merci", "s'il", "vous", "plaît", "pourriez"},
    "es": {"quisiera", "gracias", "por", "favor", "necesito"},
    "pt": {"gostaria", "obrigado", "por", "favor", "preciso"},
    "de": {"möchte", "bitte", "danke", "stattdessen"},
    "nl": {"graag", "alstublieft", "alsjeblieft"},
    "tr": {"istiyorum", "lütfen", "teşekkür"},
    "pl": {"chciałbym", "proszę", "dziękuję"},
    "vi": {"tôi", "muốn", "làm", "ơn"},
    "id": {"saya", "ingin", "tolong"},
    "sv": {"jag", "skulle", "vilja", "snälla"},
}


def detect_language(text: str) -> str:
    """Cheap heuristic for logging/analytics ONLY. Never used to gate detection."""
    for lang, (lo, hi) in _SCRIPT_RANGES:
        if any(lo <= ord(ch) <= hi for ch in text):
            return lang
    low = text.lower()
    tokens = set(re.findall(r"[a-zà-ÿ']+", low))
    best_lang, best_score = "en", 0
    for lang, stopwords in _LATIN_STOPWORDS.items():
        score = len(tokens & stopwords)
        if score > best_score:
            best_lang, best_score = lang, score
    return best_lang


# --------------------------------------------------------------------------
# 4. Integration wrapper — plug your real English detector in as a callback
# --------------------------------------------------------------------------

@dataclass
class DecisionResult:
    is_decision: bool
    reason: str
    language_guess: str
    matched_multilingual_pattern: Optional[str] = None
    translated_text: Optional[str] = None
    english_fn_result: Optional[bool] = None


def detect_decision(
    clause_text: str,
    role: str,
    english_decision_fn: Callable[[str, str], bool],
) -> DecisionResult:
    """
    Drop-in wrapper around your EXISTING English decision detector.

    english_decision_fn: your current function, e.g. `is_decision_clause(text, role)`.
    This wrapper does not change that function at all — it only widens what
    gets INTO it.

    Order of checks (cheapest / most confident first):
      1. Run your existing English logic directly (covers pure-English clauses,
         zero overhead added for the majority case).
      2. Run the merged multilingual pattern bank directly against raw text
         (covers common non-English decision phrasing with zero translation cost).
      3. If neither fires, apply the lightweight word-level translation and
         re-run your existing English logic against the translated text
         (covers phrasing outside the regex bank but with recognizable
         decision vocabulary).
    """
    lang = detect_language(clause_text)

    # 1. existing English logic, unmodified, tried first
    if english_decision_fn(clause_text, role):
        return DecisionResult(True, "english_fn_direct", lang, english_fn_result=True)

    # 2. multilingual regex bank, zero translation cost
    for pat_lang, pattern in MULTILINGUAL_PATTERNS:
        m = pattern.search(clause_text)
        if m:
            return DecisionResult(
                True, "multilingual_pattern_match", lang,
                matched_multilingual_pattern=f"{pat_lang}:{m.group(0)}",
            )

    # 3. word-level dictionary translate -> re-run existing English logic
    translated = lightweight_translate(clause_text)
    if translated != clause_text:
        fn_result = english_decision_fn(translated, role)
        if fn_result:
            return DecisionResult(
                True, "translated_then_english_fn", lang,
                translated_text=translated, english_fn_result=True,
            )
        return DecisionResult(
            False, "translated_but_no_match", lang,
            translated_text=translated, english_fn_result=False,
        )

    return DecisionResult(False, "no_signal", lang)


# --------------------------------------------------------------------------
# 5. Self-test against the actual missed rows from the eval CSV
# --------------------------------------------------------------------------

def _demo_english_fn(text: str, role: str) -> bool:
    """Stand-in for the caller's real English detector, for the self-test only."""
    low = text.lower()
    patterns = [
        r"\bi'?d like\b", r"\bi would like\b", r"\bi want\b", r"\bi need\b",
        r"\bplease (change|cancel|add|book|exchange|return)\b",
        r"\binstead\b", r"\bcan (i|you) (do|help|try)\b", r"\bcould you\b",
        r"\bi will\b", r"\blet me\b",
    ]
    return any(re.search(p, low) for p in patterns)


if __name__ == "__main__":
    test_rows = [
        ("it", 'Ciao. Mi servirebbe un codice in c# da usare con Unity. '
               'Praticamente devo fare un backup di un oggetto "Terrain".'),
        ("fr", "Merci! I would like to cancel any remaining reservations as well, "
               "even if there's no refund. Could you assist with that?"),
        ("zh", "请帮我把这个改成黑色，取消原来的订单。"),
        ("de", "Ich möchte den Flug stattdessen stornieren, bitte."),
        ("en_control", "This is just a plain statement with no decision content at all."),
    ]
    print(f"{'lang':<10}{'is_decision':<13}{'reason':<28}{'clause'}")
    print("-" * 100)
    for tag, text in test_rows:
        result = detect_decision(text, role="user", english_decision_fn=_demo_english_fn)
        print(f"{tag:<10}{str(result.is_decision):<13}{result.reason:<28}{text[:60]}")