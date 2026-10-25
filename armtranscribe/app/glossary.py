"""Fixes around the models for words they get wrong in conversation: personal names, one-word
replies, a few terms. Wrappers keep the backends swappable:

    asr = NamedTranscriber(asr, names)    # capitalises known names in the Armenian text, drops a leading dash
    mt = GlossaryTranslator(mt, names)    # English words (terms.py), names -> Latin, numbers -> digits before MT, short replies
"""
import re

from app.numbers import digitize
from app.terms import englishize

# Armenian first names (and Russian ones common in Armenia), recognised anywhere, with case
# endings. Names that are much more often an ordinary word are in JAN_ONLY instead.
COMMON_NAMES = """
Աբգար Աբել Աբո Աբրահամ Ադամ Ազնիվ Ալբերտ Ալեն Ալեք Ալեքս Ալեքսան Ալեքսանդր Ալիկ Ակսել Ահարոն Աղասի Աղաս
Ամբարձում Անդրանիկ Անդրեյ Անտոն Անուշավան Աշոտ Առաքել Առնակ Ավետ Ավետիս Ավետիք Արամ Արամայիս Արամե Արգամ
Արգիշտի Արեգ Արեն Արթուր Արմեն Արմենակ Արման Արշակ Արշավիր Արսեն Արտակ Արտաշես Արտավազդ Արտեմ Արտյոմ Արտուր
Արփիար Բաբկեն Բագրատ Բարսեղ Բենիամին Բորիս Գաբրիել Գագիկ Գալուստ Գառնիկ Գարեգին Գարիկ Գեղամ Գևորգ Գոռ
Գուրգեն Գրիգոր Գրիշա Դանիել Դավիթ Դերենիկ Դմիտրի Եղիշե Եդգար Երվանդ Զավեն Զարեհ Զոհրաբ Զորի Էդգար Էդիկ
Էդմոնդ Էդուարդ Էմիլ Էմմանուել Էրիկ Էրնեստ Թաթուլ Թադևոս Թորոս Ժիրայր Ժորա Ժորժ Իգոր Իվան Լևոն Լյովա
Խաչատուր Խաչիկ Կամո Կառլեն Կարապետ Կարեն Կարո Կոլյա Կոմիտաս Կոստյա Կորյուն Հակոբ Համբարձում Համլետ Հայկ
Հայկազ Հայկո Հարություն Հենրիկ Հովակիմ Հովհաննես Հովիկ Հովսեփ Հրաչ Հրաչյա Հրանտ Հրայր Ղազար Ղուկաս Մաքսիմ
Մամիկոն Մանուկ Մանվել Մարատ Մարտին Մարտիրոս Մելիք Մերուժան Միխայիլ Միշա Միքայել Մկրտիչ Մհեր Մուշեղ Մովսես
Նարեկ Ներսես Նիկոլ Նիկոլայ Նորայր Շանթ Շավարշ Պավել Պարույր Պետրոս Պողոս Ռաֆայել Ռաֆիկ Ռոբերտ Ռոման Ռուբեն
Ռուդիկ Ռուստամ Սամվել Սաշա Սասուն Սարգիս Սեդրակ Սեյրան Սերգեյ Սերժ Սերոբ Սևակ Սիմոն Սլավիկ Սմբատ Սպարտակ
Սուրեն Ստեփան Վաղարշակ Վաղինակ Վազգեն Վալերի Վահագն Վահան Վահե Վաչագան Վաչե Վարդան Վարուժան Վիգեն Վիկտոր
Վլադիմիր Վոլոդյա Տարոն Տիգրան Տիրան Ֆելիքս Ֆրունզե Օնիկ Օհան
Ադելինա Աիդա Ալինա Ալիսա Ալլա Ալվարդ Ամալյա Անահիտ Անգելինա Անժելա Աննա Աննի Անուշ Աշխեն Ասյա Աստղիկ Աստղինե
Արաքսյա Արմինե Արուսյակ Արփի Արփինե Բելլա Գայանե Գոհար Գոհարիկ Գրետա Դիանա Եվա Եվգենյա Զառա Զարուհի Զինա
Էլեն Էլենա Էլինա Էլիզա Էլմիրա Էմմա Էվելինա Էրիկա Թամար Թամարա Թաթև Թաթևիկ Ժաննա Ինգա Ինեսսա Իննա Իրինա
Լալա Լաուրա Լենա Լիանա Լիդա Լիզա Լիլի Լիլիթ Լիլիա Լուիզա Լուսինե Լուսիկ Լյուդմիլա Ծովինար Կարինե Հասմիկ
Հերմինե Հռիփսիմե Ձյունիկ Մանե Մանյա Մարգարիտ Մարգարիտա Մարգո Մարի Մարիամ Մարիաննա Մարինա Մարինե Մելանյա
Մելինե Մերի Մոնիկա Նաիրա Նանե Նարե Նարինե Նաթալի Նատալյա Նազելի Նելլի Նինա Նոնա Նորա Նունե Նունուֆար Նվարդ
Շողիկ Շուշան Շուշանիկ Ջեմմա Ջուլիետա Ռիմա Ռիտա Ռոզա Ռուզան Ռուզաննա Սաթենիկ Սառա Սեդա Սեվադա Սիլվա
Սիրանուշ Սոնա Սոնյա Սոֆյա Սոֆի Սուսաննա Սվետա Սվետլանա Սյուզի Սյուզաննա Վարսենիկ Վերա Վիկտորյա Տաթև
Տիգրանուհի Քնարիկ Քրիստինա Քրիստինե Օլգա Օֆելյա
"""

# Names that are usually ordinary words in speech (Արա "hey / do!", Անի "would do", Սիրուն "pretty",
# Արև "sun", Ազատ "free" ...): a name only right before ջան ("Արա ջան"), which is how people say them.
JAN_ONLY = """
Արա Անի Ազատ Ավագ Արև Արշալույս Գարուն Թագուհի Իշխան Ծաղիկ Մանուշակ Նվեր Սիրուն Սոս Վարդ Վրեժ Փայլակ
"""

# A capitalised word before ջան is taken as a name too (rare names), unless it is one of these.
NOT_NAMES = set("""
մամ մամա մամիկ մայրիկ մայր պապ պապա պապիկ հայրիկ հայր տատ տատի տատիկ ախպեր ախպերս եղբայր եղբայրս քույր
քուրս քուրիկ բալես բալա բալիկ բալիկս ընկեր ընկերս ընկերուհի աղջիկ աղջիկս տղա տղաս որդի որդիս հոպար մորքուր
հորքուր քեռի ձյաձյա ապեր ախչի ջիգյար սիրտս հոգիս արևս կյանքս սիրելիս սիրելի պարոն տիկին դոկտոր ուստա
վարպետ բարեկամ հարևան կնիկ ամուսին մարդ երեխա թոռ թոռնիկ լավ շատ հա ու և էլ դե դու դուք
""".split())

# "Անուշ" is also "sweet": not a name in "անուշ եղի / լինի" (enjoy your meal), "անուշ երազներ".
NAME_EXCEPTIONS = {"Անուշ": r"\s+(?:եղի|եղեք|լինի|լինեք|ըլնի|ըլնեք|երազ|քնեք|քնի)"}

# Case endings a name can take (Գոռի, Գոռին, Գոռը, Գոռից, Աննայի, Լուսինեն ...)
SUFFIXES = ["", "ը", "ն", "ի", "ին", "ից", "ով", "ում", "ս", "դ", "ենք", "ենց", "անց", "ու", "ուն", "ուց",
            "յի", "յին", "յից", "յով", "յում", "յենց"]

_TRANSLIT = {
    "ա": "a", "բ": "b", "գ": "g", "դ": "d", "ե": "e", "զ": "z", "է": "e", "ը": "y", "թ": "t", "ժ": "zh",
    "ի": "i", "լ": "l", "խ": "kh", "ծ": "ts", "կ": "k", "հ": "h", "ձ": "dz", "ղ": "gh", "ճ": "ch", "մ": "m",
    "յ": "y", "ն": "n", "շ": "sh", "ո": "o", "չ": "ch", "պ": "p", "ջ": "j", "ռ": "r", "ս": "s", "վ": "v",
    "տ": "t", "ր": "r", "ց": "ts", "փ": "p", "ք": "k", "օ": "o", "ֆ": "f", "և": "ev",
}


def transliterate(name: str) -> str:
    """Armenian name -> common Latin spelling (Գոռ -> Gor, Երվանդ -> Yervand, Հովհաննես -> Hovhannes)."""
    s = name.lower().replace("եւ", "և")
    out, i = [], 0
    while i < len(s):
        if s.startswith("ու", i):
            out.append("u")
            i += 2
            continue
        c = s[i]
        if i == 0 and c == "ե":
            out.append("ye")
        elif i == 0 and c == "ո":
            out.append("vo")
        else:
            out.append(_TRANSLIT.get(c, c))
        i += 1
    return "".join(out).capitalize()


def _key(word: str) -> str:
    """Lookup key: lowercase, ռ = ր (the ASR mixes them up in names: Գոր for Գոռ)."""
    return word.lower().replace("ռ", "ր")


def _alt(name: str) -> str:
    return re.sub("[րռ]", "[րռ]", re.escape(name.lower()))


def _pattern(names) -> re.Pattern | None:
    """A whole word = one of `names` (any case) + a case ending; the longest name wins."""
    if not names:
        return None
    alts = "|".join(_alt(n) + (f"(?!{NAME_EXCEPTIONS[n]})" if n in NAME_EXCEPTIONS else "")
                    for n in sorted(names, key=len, reverse=True))
    sufs = "|".join(re.escape(s) for s in sorted(SUFFIXES, key=len, reverse=True))
    return re.compile(rf"(?<![\w-])({alts})({sufs})(?![\w-])", re.IGNORECASE)


# word + ջան ("Արա ջան", "Գոհար ջան", also written together: "Արաջան")
_JAN = re.compile(r"(?<![\w-])(\w+?)(\s+|-)?ջան(?![\w-])", re.IGNORECASE)


class Names:
    """Finds names in Armenian text: built-in names (any case, with case endings) anywhere, and
    before ջան also the names in JAN_ONLY and any other capitalised word."""

    def __init__(self, names=None, jan_only=None):
        names = names or COMMON_NAMES.split()
        self._jan_only = {_key(n): n for n in (jan_only if jan_only is not None else JAN_ONLY.split())}
        self.latin = {n: transliterate(n) for n in [*names, *self._jan_only.values()]}
        self._by_lower = {_key(n): n for n in names}
        self._pattern = _pattern(names)

    def _jan_name(self, word: str, sep: str | None) -> str | None:
        """The name in "<word> ջան", or None if <word> is not one."""
        low = _key(word)
        if low in self._by_lower:
            return self._by_lower[low]
        if low in self._jan_only:
            return self._jan_only[low]
        if sep and word[:1].isupper() and word.lower() not in NOT_NAMES:  # a rare name, written apart from ջան
            return word
        return None

    def _sub(self, text: str, repl, jan_repl) -> str:
        def jan(m):
            name = self._jan_name(m.group(1), m.group(2))
            return jan_repl(name) if name else m.group(0)

        text = _JAN.sub(jan, text)
        if self._pattern:
            text = self._pattern.sub(lambda m: repl(self._by_lower[_key(m.group(1))], m.group(2).lower()), text)
        return text

    def capitalize(self, hy: str) -> str:
        """գոռի -> Գոռի, արա ջան -> Արա ջան: ASR writes names in lowercase, MT then translates them as words."""
        return self._sub(hy, lambda name, suf: name + suf, lambda name: f"{name} ջան")

    def latinize(self, hy: str) -> str:
        """Գոռի -> Gor-ի, Արա ջան -> Ara: NLLB keeps Latin names as they are and still reads the case
        ending; "ջան" (dear) is dropped, NLLB would make it "Jan" or translate the name."""
        return self._sub(hy, lambda name, suf: self.latin.get(name, transliterate(name)) + (f"-{suf}" if suf else ""),
                         lambda name: self.latin.get(name, transliterate(name)))


# One-word / stock replies NLLB pads or misreads ("Այո" -> "Yes, you did", "Ապրես" -> "You live").
# Keys are normalised with _norm.
SHORT_REPLIES = {
    "այո": "Yes.", "հա": "Yeah.", "հաա": "Yeah.", "հա հա": "Yeah, yeah.", "չէ": "No.", "ոչ": "No.",
    "ոչ ոչ": "No, no.", "լավ": "OK.", "լավ լավ": "OK, OK.", "դե լավ": "Well, OK.", "օքեյ": "OK.", "օկ": "OK.",
    "ապրես": "Thanks, well done!", "ապրեք": "Thank you, well done!", "իհարկե": "Of course.",
    "շնորհակալություն": "Thank you.", "շնորհակալ եմ": "Thank you.", "լավ շնորհակալություն": "OK, thank you.",
    "մերսի": "Thanks.", "խնդրեմ": "You're welcome.", "բարև": "Hi.", "բարև ձեզ": "Hello.", "ողջույն": "Hello.",
    "ալո": "Hello?", "ցտեսություն": "Goodbye.", "հաջող": "Bye.", "հաջողություն": "Good luck.",
    "ճիշտ է": "Right.", "ճիշտ ա": "Right.", "համաձայն եմ": "I agree.", "չգիտեմ": "I don't know.",
    "գիտեմ": "I know.", "հասկացա": "Got it.", "հասկանալի է": "Understood.", "հասկանալի ա": "Understood.",
    "ներողություն": "Sorry.", "կներեք": "Excuse me.", "մի րոպե": "Just a minute.",
    "մի վայրկյան": "Just a second.", "վատ չի": "Not bad.", "լավ է": "Good.", "լավ ա": "Good.",
    "անուշ եղի": "Enjoy your meal!", "անուշ եղեք": "Enjoy your meal!", "անուշ լինի": "Enjoy!",
    "անուշ ըլնի": "Enjoy!",
}

# "Արա," (without ջան) is the interjection "hey", which NLLB reads as the verb "do".
HY_FIXES = [(re.compile(r"(?<![\w-])արա(?=\s*,)", re.IGNORECASE), "Hey")]

# Colloquial էս / էդ / էն ("this / that") -> literary այս / այդ / այն: NLLB reads a capitalised "Էս" as
# a name ("Ace Vegas code"). Also in their compounds (էսօր today, էդքան so much, էնտեղ there ...)
# and էսի / էդի / էնի (this one) -> սա / դա / նա.
_DEMONSTRATIVE = re.compile(r"(?<![\w-])([Էէ])([սդն])(?:(օր|քան|պես|տեղ)(\w*)|(ի)|)(?![\w-])")


def _literary(m: re.Match) -> str:
    cap = m.group(1) == "Է"
    if m.group(5):  # էսի -> սա
        word = {"ս": "սա", "դ": "դա", "ն": "նա"}[m.group(2)]
    else:
        word = "այ" + m.group(2) + (m.group(3) or "") + (m.group(4) or "")
    return word[0].upper() + word[1:] if cap else word

# Terms NLLB translates wrongly, fixed in its output.
EN_FIXES = [(re.compile(r"\bdrachma(s?)\b", re.IGNORECASE), r"dram\1")]


# What the ASR writes for (near-)silence: FastConformer turns 1 s of zeros into `Հովհ.`; it came out of
# the system stream in 4 of 5 Zoom / Meet sessions (quiet stretches at the start or end of the call),
# once as `«Մովհ է.` from the mic. Not words anyone says, so the line is dropped. Keys: _norm.
NOISE_OUTPUTS = {"հովհ", "մովհ", "մովհ է"}


def _norm(hy: str) -> str:
    """Lowercase, no punctuation. Armenian stress / question marks sit inside the word (Ապրե՛ս,
    Ո՞նց), so they are deleted rather than turned into spaces."""
    hy = re.sub("[\u055a-\u055f]", "", hy.lower().replace("եւ", "և"))
    return " ".join(re.sub(r"[^\w\s]", " ", hy).split())


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[։:?!.])\s+", text.strip()) if s]


def _lone_number(sentence: str) -> str | None:
    """A sentence that is only a number (`Քսաներեք։`, an answer) is printed as digits: NLLB, given
    `23։` alone, invents a sentence ("What is the meaning of the name?"). With a case ending
    (`Տասներկուսին` "at twelve" / "on the 12th") it is left to MT, which can tell them apart better."""
    d = _norm(digitize(sentence))
    return d + "." if d.isdigit() else None


class NamedTranscriber:
    """Capitalises known names, drops the dialogue dash the ASR sometimes starts a line with,
    outputs of a single letter and what it writes for non-speech sounds (NOISE_OUTPUTS)."""

    def __init__(self, asr, names: Names):
        self.asr, self.names = asr, names

    def transcribe(self, audio) -> str:
        text = re.sub(r"^[֊\-–—]\s*", "", self.asr.transcribe(audio))
        if len(re.findall(r"\w", text)) < 2:  # "Հ.", "Կ": noise read as a letter, MT turns it into "C. What is it?"
            return ""
        if _norm(text) in NOISE_OUTPUTS:  # "Հովհ." -> "How can we do this?"
            return ""
        return self.names.capitalize(text)


class GlossaryTranslator:
    def __init__(self, mt, names: Names | None = None):
        self.mt, self.names = mt, names

    def _pre(self, hy: str) -> str:
        for pattern, repl in HY_FIXES:
            hy = pattern.sub(repl, hy)
        hy = _DEMONSTRATIVE.sub(_literary, hy)
        hy = englishize(digitize(hy))
        return self.names.latinize(hy) if self.names else hy

    @staticmethod
    def _post(en: str) -> str:
        for pattern, repl in EN_FIXES:
            en = pattern.sub(repl, en)
        return en

    def translate(self, hy: str) -> str:
        """Short replies come from the table, also as a sentence of their own inside a segment
        ("Ապրես։ Աղաս դու ասա։": NLLB drops the "Ապրես" when given both sentences)."""
        if not hy.strip():
            return ""
        out, rest = [], []
        for sentence in _sentences(hy):
            if (reply := SHORT_REPLIES.get(_norm(sentence)) or _lone_number(sentence)) is not None:
                if rest:
                    out.append(self._mt(" ".join(rest)))
                    rest = []
                out.append(reply)
            else:
                rest.append(sentence)
        if rest:
            out.append(self._mt(" ".join(rest)))
        return " ".join(out)

    def _mt(self, hy: str) -> str:
        """Sentence by sentence: given two sentences NLLB often drops the second one entirely ("…,
        vibe coding. I like Grok." -> only the first; 4 of 8 joined segments on the calls). A
        "sentence" of one or two words is kept with its neighbour, so a stray full stop doesn't
        cut a phrase into fragments."""
        chunks: list[str] = []
        for sentence in _sentences(self._pre(hy)):
            if chunks and (len(sentence.split()) < 3 or len(chunks[-1].split()) < 3):
                chunks[-1] += " " + sentence
            else:
                chunks.append(sentence)
        return self._post(" ".join(self.mt.translate(c) for c in chunks))

    def translate_continuation(self, prev_hy: str, prev_en: str, hy: str) -> tuple[str, str]:
        if not hasattr(self.mt, "translate_continuation"):
            return prev_en, self.translate(hy)
        head, added = self.mt.translate_continuation(self._pre(prev_hy), prev_en, self._pre(hy))
        return self._post(head), self._post(added)
