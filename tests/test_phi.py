import pytest

from trogs26.lm import phi


def test_only_letters_visible_on_the_stone_enter_and_lacunae_and_vacats_split():
    # restoration, editorial addition, resolved abbreviation and erasure never enter; a letter written in error and
    # an underdotted one do; a lacuna and a vacat split, the interpunct stays
    text = ['ἔδοξεν τῆι [βουλῆι] καὶ', 'τῶι δή<μ>ωι· Ἀθη(ναῖος) ⟦ἐρασμένα⟧ {ε}ἰ̣ vac. ος']
    assert phi.visible_lines(text) == ['EDOXENTHI', 'KAI', 'TWIDHWI:AQHEI', 'OS']
    with pytest.raises(ValueError):
        phi.visible_lines('αβ]γ')


def test_references_of_editions_and_of_squeeze_names():
    assert phi.references('IG II² 333+334a, SEG 21.5') == {'IG II^2 333', 'IG II^2 334'}
    assert phi.references('IG II3 1 567; Agora XVI 12') == {'IG II^3,1 567', 'Agora XVI 12'}
    # a fascicle of a part is no inscription number
    assert phi.references('IG X,2 1 44 [Thessaloniki] — IG XII,4 1:303') == {'IG X,2 1:44', 'IG XII,4 1:303'}
    assert phi.references('IG IX,1² 2:123, IG IX,1 694, IG IV²,1 121') == {
        'IG IX,1^2 2:123', 'IG IX,1 694', 'IG IV^2,1 121'}
    assert phi.name_references('IG_II2_2061b+2093b') == {'IG II^2 2061', 'IG II^2 2093'}
    assert phi.name_references('IG_II2_2464_Copy2') == {'IG II^2 2464'}
    assert phi.name_references('Agora_I_4985') == set()
