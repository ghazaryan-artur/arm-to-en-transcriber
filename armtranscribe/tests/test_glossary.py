from app.glossary import COMMON_NAMES, GlossaryTranslator, Names, NamedTranscriber, SHORT_REPLIES, transliterate


def test_transliterate():
    assert [transliterate(n) for n in ["Գոռ", "Աղաս", "Երվանդ", "Ոսկան", "Հովհաննես", "Գևորգ", "Ժիրայր", "Լուսինե"]] == \
        ["Gor", "Aghas", "Yervand", "Voskan", "Hovhannes", "Gevorg", "Zhirayr", "Lusine"]


def test_names_with_case_endings_are_capitalised_and_latinised():
    n = Names()
    hy = "Պատասխանեմ գոռի հարցին, աննայի հետ, լուսինեն ասաց"
    assert n.capitalize(hy) == "Պատասխանեմ Գոռի հարցին, Աննայի հետ, Լուսինեն ասաց"
    assert n.latinize(hy) == "Պատասխանեմ Gor-ի հարցին, Anna-յի հետ, Lusine-ն ասաց"


def test_words_that_only_start_like_a_name_are_left_alone():
    n = Names()
    assert n.latinize("հայկական պատմության արագ արա արի") == "հայկական պատմության արագ արա արի"
    assert n.latinize("Հայկը") == "Hayk-ը"


def test_longest_name_wins():
    assert Names().latinize("Աղասին") == "Aghasi-ն"   # Աղասի + ն, not Աղաս + ին
    assert Names().latinize("Աղասը") == "Aghas-ը"


def test_ambiguous_names_are_not_built_in():
    names = set(COMMON_NAMES.split())
    assert not names & {"Արա", "Անի", "Անուշ", "Գոհար", "Ապրես", "Վրեժ"}
    assert Names(["Աննա"]).latinize("Գոռ") == "Գոռ"


class EchoMT:
    def __init__(self):
        self.calls = []

    def translate(self, hy):
        self.calls.append(hy)
        return f"<{hy}> 150 drachmas"

    def translate_continuation(self, prev_hy, prev_en, hy):
        self.calls.append((prev_hy, prev_en, hy))
        return f"{prev_en} <{hy}> drachma"


def test_glossary_translator():
    mt = EchoMT()
    g = GlossaryTranslator(mt, Names())
    assert g.translate("Ապրե՛ս։") == SHORT_REPLIES["ապրես"] and mt.calls == []
    assert g.translate("Այո") == "Yes."
    assert g.translate("գոռի հարցը") == "<Gor-ի հարցը> 150 drams"
    assert g.translate_continuation("գոռը", "Gor", "եկավ") == "Gor <եկավ> dram"
    assert mt.calls[-1] == ("Gor-ը", "Gor", "եկավ")


def test_named_transcriber():
    class ASR:
        def transcribe(self, audio):
            return "֊ աղաս դու ասա"

    assert NamedTranscriber(ASR(), Names()).transcribe(None) == "Աղաս դու ասա"


def test_short_reply_inside_a_segment_is_kept():
    mt = EchoMT()
    g = GlossaryTranslator(mt, Names())
    assert g.translate("Ապրես։ Աղաս դու ասա։") == "Thanks, well done! <Aghas դու ասա։> 150 drams"
    assert g.translate("Լավ։ Գնացինք։ Այո։") == "OK. <Գնացինք։> 150 drams Yes."
    assert g.translate("") == ""
