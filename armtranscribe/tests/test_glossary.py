from app.glossary import (COMMON_NAMES, JAN_ONLY, GlossaryTranslator, Names, NamedTranscriber, SHORT_REPLIES,
                          transliterate)


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


def test_word_like_names_only_before_jan():
    n = Names()
    assert not set(COMMON_NAMES.split()) & set(JAN_ONLY.split())
    assert n.latinize("Արա, լսի, արա էդ գործը") == "Արա, լսի, արա էդ գործը"   # "hey", "do"
    assert n.capitalize("արա ջան, ինչ ես անում") == "Արա ջան, ինչ ես անում"
    assert n.latinize("Արա ջան, ինչ ես անում") == "Ara, ինչ ես անում"          # ջան dropped for MT
    assert n.latinize("Արաջան արի") == "Ara արի"


def test_jan_marks_rare_capitalised_names_but_not_kin_words():
    n = Names()
    assert n.latinize("Ծատուր ջան, բարև") == "Tsatur, բարև"
    assert n.latinize("Մամ ջան, արի") == "Մամ ջան, արի"
    assert n.latinize("ծատուր ջան") == "ծատուր ջան"                            # lowercase unknown: not sure


def test_gohar_and_anush_are_names_except_in_set_phrases():
    n = Names()
    assert n.latinize("Գոհարին ասա, Անուշը զանգեց") == "Gohar-ին ասա, Anush-ը զանգեց"
    assert n.latinize("անուշ եղի, անուշ երազներ") == "անուշ եղի, անուշ երազներ"
    assert SHORT_REPLIES["անուշ եղի"] == "Enjoy your meal!"


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


def test_ara_comma_is_hey():
    mt = EchoMT()
    GlossaryTranslator(mt, Names()).translate("Արա, էդ ինչ ես անում, արա")
    assert mt.calls[-1] == "Hey, էդ ինչ ես անում, արա"


def test_r_and_rr_are_interchangeable_in_names():
    n = Names()
    assert n.capitalize("գոր մի երկու բան ասա") == "Գոռ մի երկու բան ասա"
    assert n.latinize("Գորին ասա, Ռուբենը եկավ, Րուբեն") == "Gor-ին ասա, Ruben-ը եկավ, Ruben"
