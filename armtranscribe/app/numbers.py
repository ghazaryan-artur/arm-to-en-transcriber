"""Armenian number words -> digits before MT.

NLLB gets small numbers right (`երկու ժամ` "two hours") but garbles compound ones: `Քսաներեք։`
("Twenty-three.") -> "It's twenty", `Տասներկուսին` ("at twelve") -> "It's twelve". Written as
digits, with the case ending kept the way names are (`23`, `12-ին`, `2500 dram`), it translates
them reliably.

Only numbers above ten are converted (`քսաներեք`, `հարյուր հիսուն`, `երկու հազար տասնութ`): one-word
numbers up to ten stay words, NLLB knows them and they are often not quantities (`մի երկու բան`
"a couple of things", `երեքով ենք` "there are three of us").
"""
import re

UNITS = {"մեկ": 1, "երկու": 2, "երկուս": 2, "էրկու": 2, "էրկուս": 2, "երեք": 3, "էրեք": 3, "չորս": 4,
         "հինգ": 5, "վեց": 6, "յոթ": 7, "ութ": 8, "ինը": 9, "ինն": 9}
TENS = {"տաս": 10, "քսան": 20, "երեսուն": 30, "էրեսուն": 30, "քառասուն": 40, "հիսուն": 50, "վաթսուն": 60,
        "յոթանասուն": 70, "ութսուն": 80, "իննսուն": 90}
TEEN = "տասն"  # տասն + unit: տասնմեկ 11, տասներկու 12 … տասնինը 19
HUNDRED, THOUSAND = "հարյուր", "հազար"
ORDINALS = {"երկրորդ": 2, "երրորդ": 3, "չորրորդ": 4}  # irregular ordinals (also in compounds: քսաներկրորդ)

# endings kept after the digits: ordinal (երորդ / րդ), then a case ending
_ORD = r"երորդ|րդ"
_CASE = r"ական|ամյա|անոց|ից|ով|ում|ին|ուց|ու|ը|ն|ի"
_MARKS = "՚-՟"  # Armenian stress / question / exclamation marks sit inside the word (Տասներկուս՞ն)


def _alt(words) -> str:
    return "|".join(sorted(map(re.escape, words), key=len, reverse=True))


# One word of a number: [hundreds / thousands …] [tens (+ a stray ն: հիսուններկու) | տասն] [unit | irregular ordinal]
_MORPH = re.compile(rf"({_alt(ORDINALS)}|{_alt([HUNDRED, THOUSAND])}|{TEEN}(?=(?:{_alt(UNITS)}))|{_alt(TENS)}|{_alt(UNITS)})")
_WORD = re.compile(rf"(?:{_MORPH.pattern}n?)+".replace("n?", "ն?"))
# A bare "դրամ" after the digits is written in Latin: NLLB drops it after a number ("Գինը 2500 դրամ ա"
# -> "The price is 2,500") but keeps "2500 dram". With an ending it stays Armenian: "160 dram-ով"
# becomes "160-gram", "160 դրամով" is translated right.
_DRAM = re.compile(r" դրամ(?![\w-])", re.IGNORECASE)
_TOKEN = re.compile(rf"(?<![\w-])([\w{_MARKS}]+)(?![\w-])")


def _morphs(stem: str) -> list[str] | None:
    """`քսաներեք` -> ['քսան', 'երեք'], `տասներկուս` -> ['տասն', 'երկուս']; None if not a number."""
    out, i = [], 0
    while i < len(stem):
        m = _MORPH.match(stem, i)
        if not m:
            if out and stem[i] == "ն" and out[-1] in TENS:  # հիսուններկու: ASR doubles the ն
                i += 1
                continue
            return None
        out.append(m.group(1))
        i = m.end()
    return out or None


def _split(word: str):
    """(morphs, ordinal ending, case ending) of a number word, or None. The longest stem that is a
    number wins, so `երեքին` = երեք + ին and `ինը` = 9, not ին + ը."""
    low = re.sub(f"[{_MARKS}]", "", word.lower())
    for end in range(len(low), 0, -1):
        rest = low[end:]
        m = re.fullmatch(rf"({_ORD})?({_CASE})?", rest)
        if not m:
            continue
        morphs = _morphs(low[:end])
        if morphs:
            ordinal = m.group(1) or ""
            if morphs[-1] in ORDINALS:
                if ordinal:
                    continue
                ordinal = "րդ"
            return morphs, ordinal, m.group(2) or ""
    return None


def _value(morphs: list[str]) -> int | None:
    """Place-value sum: ['երկու', 'հազար', 'հինգ', 'հարյուր'] -> 2500. None if the words don't make
    one number (`երկու երեք` "two or three")."""
    total = current = 0
    prev = None
    for w in morphs:
        if w == TEEN:
            if current % 100:
                return None
            current += 10
            prev = TEEN
            continue
        v = UNITS.get(w) or ORDINALS.get(w)
        if v is not None:
            if current % 10 or (current % 100 >= 10 and prev not in (TEEN, *TENS)):
                return None
            current += v
        elif w in TENS:
            if current % 100:
                return None
            current += TENS[w]
        elif w == HUNDRED:
            if current >= 100 and current % 100 == 0 and prev != HUNDRED:  # "երկու հարյուր" ok, "հարյուր հարյուր" no
                return None
            current = (current or 1) * 100
        elif w == THOUSAND:
            if total:
                return None
            total, current = (current or 1) * 1000, 0
        prev = w
    return total + current


def digitize(hy: str) -> str:
    """Numbers above ten as digits, endings kept: `Քսաներեք։` -> `23։`, `Տասներկուսին` -> `12-ին`,
    `երկու հազար հինգ հարյուր դրամ` -> `2500 dram`, `հինգերորդ` stays."""
    out, last = [], 0
    tokens = list(_TOKEN.finditer(hy))
    i = 0
    while i < len(tokens):
        group, ordinal, case = [], "", ""
        j = i
        # consecutive number words separated by spaces; a word with an ending closes the number
        while j < len(tokens):
            if group and hy[tokens[j - 1].end():tokens[j].start()] != " ":
                break
            parts = _split(tokens[j].group(1))
            if not parts or _value(group + parts[0]) is None:
                break
            group += parts[0]
            j += 1
            ordinal, case = parts[1], parts[2]
            if ordinal or case:
                break
        value = _value(group) if group else None
        if value is None or value <= 10 or (j - i == 1 and len(group) == 1 and group[0] in (HUNDRED, THOUSAND)):
            i += 1
            continue
        start, end = tokens[i].start(), tokens[j - 1].end()
        ending = ("րդ" if ordinal else "") + case  # 30-րդ, the usual written form of երեսուներորդ
        out.append(hy[last:start] + str(value) + (f"-{ending}" if ending else ""))
        last = end
        if not ending and (m := _DRAM.match(hy, end)):
            out.append(" dram")
            last = m.end()
            while j < len(tokens) and tokens[j].start() < last:
                j += 1
        i = j
    out.append(hy[last:])
    return "".join(out)
