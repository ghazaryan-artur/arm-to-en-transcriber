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
        return prev_en, f"<{hy}> drachma"


def test_glossary_translator():
    mt = EchoMT()
    g = GlossaryTranslator(mt, Names())
    assert g.translate("Ապրե՛ս։") == SHORT_REPLIES["ապրես"] and mt.calls == []
    assert g.translate("Այո") == "Yes."
    assert g.translate("գոռի հարցը") == "<Gor-ի հարցը> 150 drams"
    assert g.translate_continuation("գոռը", "Gor", "եկավ") == ("Gor", "<եկավ> dram")
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
    assert mt.calls[-1] == "Hey, այդ ինչ ես անում, արա"


def test_r_and_rr_are_interchangeable_in_names():
    n = Names()
    assert n.capitalize("գոր մի երկու բան ասա") == "Գոռ մի երկու բան ասա"
    assert n.latinize("Գորին ասա, Ռուբենը եկավ, Րուբեն") == "Gor-ին ասա, Ruben-ը եկավ, Ruben"


def test_single_letter_asr_output_is_dropped():
    class ASR:
        def __init__(self, text):
            self.text = text

        def transcribe(self, audio):
            return self.text

    assert NamedTranscriber(ASR("Հ."), Names()).transcribe(None) == ""
    assert NamedTranscriber(ASR("Ու։"), Names()).transcribe(None) == "Ու։"


def test_segment_is_translated_sentence_by_sentence():
    mt = EchoMT()
    g = GlossaryTranslator(mt, Names())
    g.translate("Դրա համար կա էդ ամենը։ Ես սիրում եմ գրոկ։ Լավ։ Դե արի։ Մենք էլ ենք գալիս շուտով։")
    # "Լավ" is a short reply on its own; "Դե արի" (2 words) stays with the next sentence
    assert mt.calls == ["Դրա համար կա այդ ամենը։", "Ես սիրում եմ Grok։", "Դե արի։ Մենք էլ ենք գալիս շուտով։"]


def test_terms_are_written_in_english_for_mt():
    from app.terms import englishize

    assert englishize("մեր սկիլերը, վեյբ կոտինգ, չատ ջիպիցի") == "մեր skill-երը, vibe coding, ChatGPT"
    assert englishize("Սիրիս էյն էր, քսան միլիոն ռեյզարեցինք") == "Series A-ն էր, քսան միլիոն raised"
    assert englishize("բագերը, ռելիզը, դեդլայնից, Լարավելով") == "bug-երը, release-ը, deadline-ից, Laravel-ով"
    # Armenian words that only look alike are left alone
    assert englishize("պատմության դասեր տասը գորգ փուշ ափ") == "պատմության դասեր տասը գորգ փուշ ափ"


# --- numbers -> digits (app/numbers.py) ---

from app.numbers import digitize  # noqa: E402


def test_digitize_compounds_with_endings():
    assert digitize("Քսաներեք։") == "23։"
    assert digitize("Տասներկուսին։") == "12-ին։"
    assert digitize("որ հիսուններկու երկիր") == "որ 52 երկիր"  # ASR doubles the ն
    assert digitize("երկու հազար տասնութ թվականին") == "2018 թվականին"
    assert digitize("հազար ինը հարյուր իննսունչորս") == "1994"
    assert digitize("երեք հարյուր հիսուն հազար") == "350000"
    assert digitize("մեր երեսուներորդ ռելիզը") == "մեր 30-րդ ռելիզը"
    assert digitize("քսաներկրորդ") == "22-րդ"


def test_digitize_dram_in_latin_only_without_ending():
    assert digitize("Գինը երկու հազար հինգ հարյուր դրամ ա") == "Գինը 2500 dram ա"
    assert digitize("հարյուր վաթսուն դրամով") == "160 դրամով"


def test_digitize_leaves_small_and_non_numbers():
    for text in ["մի երկու բան", "երեքով ենք", "ժամը երեքն անց կեսին", "հինգերորդ փորձ", "մի տասը րոպե",
                 "հարյուր տոկոսով", "հինգշաբթի", "ինչ-որ մեկը", "մեկ երկու երեք", "Թիվ մինիսուն երկու"]:
        assert digitize(text) == text


def test_lone_number_is_printed_as_digits():
    mt = EchoMT()
    g = GlossaryTranslator(mt)
    assert g.translate("Քսաներեք։") == "23." and mt.calls == []
    g.translate("Տասներկուսին։")  # with an ending it goes to MT, as digits
    assert mt.calls == ["12-ին։"]


def test_colloquial_demonstratives_become_literary():
    mt = EchoMT()
    GlossaryTranslator(mt).translate("Էս լեգասի կոդը էսօր էդքան էլ չի աշխատում, էնտեղից ա, Էսի նոր ա։")
    assert mt.calls == ["Այս լեգասի կոդը այսօր այդքան էլ չի աշխատում, այնտեղից ա, Սա նոր ա։"]
    mt.calls.clear()
    GlossaryTranslator(mt).translate("Էրեկ էսել էրկու ժամ սպասեցի։")  # էրեկ, էրկու, էսել are not demonstratives
    assert mt.calls == ["Էրեկ էսել էրկու ժամ սպասեցի։"]


def test_non_speech_outputs_are_dropped():
    for noise, kept in [("Հովհ.", "Հովհաննեսը եկավ։"), ("«Մովհ է.", "Մովսեսը եկավ։")]:
        class ASR:
            def __init__(self, text):
                self.text = text

            def transcribe(self, audio):
                return self.text
        assert NamedTranscriber(ASR(noise), Names()).transcribe(None) == ""
        assert NamedTranscriber(ASR(kept), Names()).transcribe(None) != ""
