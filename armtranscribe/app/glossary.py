"""Fixes around the models for words they get wrong in conversation: personal names, one-word
replies, a few terms. Wrappers keep the backends swappable:

    asr = NamedTranscriber(asr, names)    # capitalises known names in the Armenian text, drops a leading dash
    mt = GlossaryTranslator(mt, names)    # names -> Latin before MT, short replies, terms after MT
"""
import re

# Common Armenian first names, used without being asked for. Names that are also ordinary words
# (Արա "hey/do", Անի "would do", Անուշ "sweet", Գոհար "gem", Սիրուն "pretty", Վարդ "rose",
# Աստղիկ "little star", Շուշան "lily", Մարգարիտ "pearl", Արև "sun", Հասմիկ "jasmine", Վրեժ
# "revenge", Իշխան "prince", Թագուհի "queen", Կարեն, Կարո ...) are left out.
COMMON_NAMES = """
Աբրահամ Ադամ Ալբերտ Ալեքս Ալեքսանդր Ակսել Աղասի Աղաս Աշոտ Առաքել Արամ Արամայիս Արգիշտի Արեն Արթուր
Արման Արմեն Արմենակ Արսեն Արշակ Արշավիր Արտակ Արտաշես Արտավազդ Արտյոմ Արտուր Բաբկեն Բորիս Գագիկ Գարիկ Գարեգին
Գեղամ Գևորգ Գոռ Գուրգեն Գրիգոր Դավիթ Դերենիկ Դիանա Եդգար Եղիշե Երվանդ Զավեն Զորի Էդգար Էդուարդ Էդիկ Էմիլ
Էրիկ Էրիկա Թաթուլ Թադևոս Ժիրայր Ժորա Իգոր Լևոն Լյովա Խաչատուր Խաչիկ Կամո Կարապետ Կոմիտաս Համլետ Հայկ
Հայկազ Հակոբ Հարություն Հովհաննես Հովիկ Հովսեփ Հրաչ Հրաչյա Հրանտ Ղուկաս Մանվել Մանուկ Մարատ Մարտին Մերուժան
Մկրտիչ Մհեր Միքայել Մուշեղ Մովսես Նարեկ Նիկոլ Նորայր Շանթ Պավել Պետրոս Ռաֆայել Ռաֆիկ Ռոբերտ Ռուբեն Ռուստամ
Սամվել Սարգիս Սեդրակ Սերգեյ Սերոբ Սևակ Սիմոն Սուրեն Ստեփան Վազգեն Վահագն Վահե Վաչե Վարդան Վարուժան Վլադիմիր
Տիգրան Տիրան Ֆելիքս
Ալինա Ալլա Ալվարդ Անահիտ Աննա Անժելա Աննի Արմինե Արփինե Աստղինե Բելլա Գայանե Գոհարիկ Դիանա Եվա Զառա Զարուհի
Էլեն Էլինա Էլիզա Էմմա Էվելինա Թամարա Ինգա Իննա Իրինա Լաուրա Լենա Լիանա Լիլիթ Լիլիա Լուսինե Լուսիկ
Կարինե Հերմինե Հռիփսիմե Մանե Մանյա Մարգարիտա Մարիամ Մարիաննա Մարինե Մարի Մելանյա Մերի Մոնիկա Նաիրա
Նանե Նարինե Նազելի Նելլի Նինա Նոնա Նունե Շողիկ Ջուլիետա Ռիտա Ռոզա Ռուզաննա Սաթենիկ Սեդա Սիրանուշ Սոնա Սոֆյա
Սոֆի Սվետլանա Սյուզաննա Տաթև Քնարիկ Քրիստինե Օլգա Օֆելյա
"""

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


def _pattern(names) -> re.Pattern | None:
    """A whole word = one of `names` (any case) + a case ending; the longest name wins."""
    if not names:
        return None
    alts = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    sufs = "|".join(re.escape(s) for s in sorted(SUFFIXES, key=len, reverse=True))
    return re.compile(rf"(?<![\w-])({alts})({sufs})(?![\w-])", re.IGNORECASE)


class Names:
    """Finds the built-in names (any case, with case endings) in Armenian text."""

    def __init__(self, names=None):
        self.latin = {n: transliterate(n) for n in (names or COMMON_NAMES.split())}
        self._by_lower = {n.lower(): n for n in self.latin}
        self._pattern = _pattern(self.latin)

    def _sub(self, text: str, repl) -> str:
        if not self._pattern:
            return text
        return self._pattern.sub(lambda m: repl(self._by_lower[m.group(1).lower()], m.group(2).lower()), text)

    def capitalize(self, hy: str) -> str:
        """գոռի -> Գոռի: ASR writes names in lowercase, MT then translates them as words."""
        return self._sub(hy, lambda name, suf: name + suf)

    def latinize(self, hy: str) -> str:
        """Գոռի -> Gor-ի: NLLB keeps Latin names as they are and still reads the case ending."""
        return self._sub(hy, lambda name, suf: self.latin[name] + (f"-{suf}" if suf else ""))


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
}

# Terms NLLB translates wrongly, fixed in its output.
EN_FIXES = [(re.compile(r"\bdrachma(s?)\b", re.IGNORECASE), r"dram\1")]


def _norm(hy: str) -> str:
    """Lowercase, no punctuation. Armenian stress / question marks sit inside the word (Ապրե՛ս,
    Ո՞նց), so they are deleted rather than turned into spaces."""
    hy = re.sub("[\u055a-\u055f]", "", hy.lower().replace("եւ", "և"))
    return " ".join(re.sub(r"[^\w\s]", " ", hy).split())


class NamedTranscriber:
    """Capitalises known names and drops the dialogue dash the ASR sometimes starts a line with."""

    def __init__(self, asr, names: Names):
        self.asr, self.names = asr, names

    def transcribe(self, audio) -> str:
        return self.names.capitalize(re.sub(r"^[֊\-–—]\s*", "", self.asr.transcribe(audio)))


class GlossaryTranslator:
    def __init__(self, mt, names: Names | None = None):
        self.mt, self.names = mt, names

    def _pre(self, hy: str) -> str:
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
        for sentence in re.split(r"(?<=[։:?!.])\s+", hy.strip()):
            if (reply := SHORT_REPLIES.get(_norm(sentence))) is not None:
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
        return self._post(self.mt.translate(self._pre(hy)))

    def translate_continuation(self, prev_hy: str, prev_en: str, hy: str) -> str:
        if not hasattr(self.mt, "translate_continuation"):
            return self.translate(f"{prev_hy} {hy}")
        return self._post(self.mt.translate_continuation(self._pre(prev_hy), prev_en, self._pre(hy)))
