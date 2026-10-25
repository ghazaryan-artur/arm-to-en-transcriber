"""English words said in Armenian (IT / startup talk): `չատ ջիպիթի`, `վեյբ կոդինգ`, `սկիլերը`.

NLLB translates them as Armenian words ("chat jeep", "WebCat"). They are replaced by the English
word before MT, keeping the Armenian ending (`skill-երը`), the way names are. Spellings match
loosely: the ASR mixes up voiced / unvoiced / aspirated consonants (գ կ ք, դ տ թ, բ պ փ, ձ ծ ց,
ջ ճ չ), ր / ռ, ե / է and ո / օ, so each listed spelling stands for all of those variants.

Only words that are not also Armenian words are listed (no `փուշ` push = "thorn", `ափ` app =
"palm", `կոլ` call, `բազա`); words NLLB already gets right (`կոդ`, `սերվեր`, `մոդել`) are left out.
"""
import re

# English -> Armenian spellings (as heard in Yerevan; several where the ASR writes them differently)
TERMS = {
    "ChatGPT": ["չատ ջիպիթի", "չատ ջիպիտի", "չատ ջիպիցի", "չաթջիպիթի", "ջիպիթի", "ջիպիտի", "ջիպիցի", "ջիփիթի"],
    "Gemini": ["ջեմինի", "ջեյմինի", "ջեմինայ", "ջեմինայի"],
    "Claude": ["քլոդ"],
    "Grok": ["գրոկ"],
    "Copilot": ["կոպայլոտ", "կոպիլոտ"],
    "Cursor": ["կուրսոր", "քըրսոր"],
    "OpenAI": ["օպեն էյ այ", "օպենէյայ", "օպենայ"],
    "Anthropic": ["անթրոպիկ", "էնթրոպիկ"],
    "LLM": ["էլ էլ էմ", "էլէլէմ"],
    "vibe coding": ["վայբ կոդինգ", "վեյբ կոդինգ", "վայբկոդինգ", "վեյբկոդինգ"],
    "coding": ["կոդինգ"],
    "prompt": ["պրոմպտ", "պրոմտ", "պրոմփթ"],
    "agent": ["էյջենթ", "էյջենտ"],
    "token": ["տոկեն", "թոքեն"],
    "fine-tuning": ["ֆայնթյունինգ", "ֆայն թյունինգ"],
    "dataset": ["դատասեթ", "դատասետ"],
    "benchmark": ["բենչմարկ"],
    "latency": ["լեյթենսի"],
    "commit": ["կոմիտ", "քոմիթ"],
    "pull request": ["պուլ ռեքվեստ", "պուլռեքվեստ"],
    "merge": ["մերջ"],
    "deploy": ["դեպլոյ", "դիպլոյ"],
    "release": ["ռելիզ"],
    "bug": ["բագ"],
    "feature": ["ֆիչ", "ֆիչա", "ֆիչեր"],
    "task": ["տասկ"],
    "sprint": ["սպրինտ"],
    "deadline": ["դեդլայն"],
    "backend": ["բեքենդ", "բեկենդ", "բեքէնդ"],
    "frontend": ["ֆրոնտենդ", "ֆրոնտ էնդ"],
    "fullstack": ["ֆուլստեկ", "ֆուլ ստեկ"],
    "developer": ["դեվելոպեր", "դևելոպեր"],
    "senior": ["սինյոր", "սենյոր"],
    "junior": ["ջունիոր"],
    "team lead": ["տիմլիդ", "տիմ լիդ", "թիմլիդ", "թիմ լիդ"],
    "startup": ["ստարտապ", "ստարտափ"],
    "raised": ["ռեյզ արեցինք", "ռեյզ ենք արել", "ռեյզ արինք", "ռեյզ արին", "ռեյզ արեց"],
    "raise": ["ռեյզ"],
    "round": ["ռաունդ"],
    "Series A": ["սիրիս էյ", "սերիս էյ", "սիրիզ էյ"],
    "Series B": ["սիրիս բի", "սերիս բի"],
    "pitch": ["պիչ"],
    "MVP": ["էմ վի պի", "էմվիպի"],
    "framework": ["ֆրեյմվորկ"],
    "Laravel": ["լարավել"],
    "Python": ["պայթոն", "պայտոն"],
    "JavaScript": ["ջավասկրիպտ", "ջավա սկրիպտ"],
    "TypeScript": ["տայպսկրիպտ", "տայպ սկրիպտ"],
    "React": ["ռեակտ", "ռիեկտ"],
    "Docker": ["դոկեր"],
    "Kubernetes": ["կուբերնետես", "կուբեռնետիս"],
    "database": ["դատաբեյս", "դատաբեյզ"],
    "cloud": ["կլաուդ"],
    "API": ["էյ պի այ", "էյպիայ"],
    "skill": ["սկիլ"],
    "soft skills": ["սոֆտ սկիլ", "սոֆթ սկիլ"],
    "interview": ["ինտերվյու"],
    "meeting": ["միտինգ"],
    "feedback": ["ֆիդբեկ", "ֆիդբեք"],
    "remote": ["ռիմոուտ", "ռեմոուտ"],
    "offer": ["օֆեր"],
    "recruiter": ["ռեկրուտեր", "ռեքրութեր"],
    "HR": ["էյչ ար", "էյչար"],
    "CV": ["սի վի", "սիվի"],
    "LinkedIn": ["լինկդին", "լինկեդին"],
    "GitHub": ["գիտհաբ", "գիթհաբ"],
    "Jira": ["ջիրա"],
    "Slack": ["սլակ"],
    "Zoom": ["զում"],
    "update": ["ապդեյտ"],
    "dashboard": ["դաշբորդ"],
    "refactoring": ["ռեֆակտորինգ"],
    "unit test": ["յունիտ տեստ"],
    "production": ["պրոդակշն", "պրոդակշեն"],
}

_CLASSES = ["գկք", "դտթ", "բպփ", "ձծց", "ջճչ", "րռ", "եէ", "ոօ"]
_CLASS_OF = {c: f"[{cls}]" for cls in _CLASSES for c in cls}

# Armenian endings a borrowed noun takes: optional plural, then case / article / possessive
_ENDING = r"((?:ն?եր)?(?:ով|ում|ից|ին|ի|ը|ն|ս|դ|ու|յի|յին|յից|յով|յում)?)"


def _spelling_pattern(spelling: str) -> str:
    words = [
        "".join(_CLASS_OF.get(c, re.escape(c)) for c in word)
        for word in spelling.lower().replace("եւ", "և").split()
    ]
    return r"[\s-]*".join(words)


def _compile() -> list[tuple[re.Pattern, str]]:
    out = []
    # longest spellings first, so "vibe coding" wins over "coding" and "Series A" over a lone word
    pairs = sorted(((sp, en) for en, sps in TERMS.items() for sp in sps), key=lambda p: -len(p[0]))
    for spelling, english in pairs:
        out.append((re.compile(rf"(?<![\w-]){_spelling_pattern(spelling)}{_ENDING}(?![\w-])", re.IGNORECASE),
                    english))
    return out


_PATTERNS = _compile()


def englishize(hy: str) -> str:
    """`վեյբ կոտինգը` -> `vibe coding-ը`: English words said in Armenian, written in English for MT."""
    for pattern, english in _PATTERNS:
        hy = pattern.sub(lambda m, e=english: e + (f"-{m.group(1)}" if m.group(1) else ""), hy)
    return hy
