import json

import pytest

from trogs26 import config


def write(path, content):
    path.write_text(json.dumps(content))
    return path


def test_override_is_merged_into_the_base_and_digested_as_merged(tmp_path):
    base = write(tmp_path / 'base.json', {'name': 'base', 'decode': {'a': 1, 'b': {'c': 2, 'd': 3}}, 'list': [1, 2]})
    experiment = write(tmp_path / 'exp.json', {'extends': str(base),
                                               'override': {'decode': {'b': {'c': 9}}, 'list': [7]}})
    cfg = config.load(experiment)
    assert cfg == {'name': 'base', 'decode': {'a': 1, 'b': {'c': 9, 'd': 3}}, 'list': [7]}
    assert config.digest(cfg) == config.digest({'list': [7], 'decode': {'b': {'d': 3, 'c': 9}, 'a': 1}, 'name': 'base'})
    assert cfg.source == {'path': str(experiment), 'extends': {'path': str(base)},
                          'override': {'decode': {'b': {'c': 9}}, 'list': [7]}}


def test_an_override_key_missing_from_the_base_is_an_error(tmp_path):
    base = write(tmp_path / 'base.json', {'decode': {'beam': 256}})
    with pytest.raises(KeyError, match=r'decode\.baem'):
        config.load(write(tmp_path / 'exp.json', {'extends': str(base), 'override': {'decode': {'baem': 128}}}))


def test_a_dict_is_only_overridden_by_a_dict(tmp_path):
    base = write(tmp_path / 'base.json', {'decode': {'cache': {'gamma': 0.2}}})
    with pytest.raises(TypeError):
        config.load(write(tmp_path / 'exp.json', {'extends': str(base), 'override': {'decode': {'cache': False}}}))


def test_an_extending_file_holds_nothing_but_the_override(tmp_path):
    base = write(tmp_path / 'base.json', {'name': 'base'})
    with pytest.raises(KeyError):
        config.load(write(tmp_path / 'exp.json', {'extends': str(base), 'name': 'x'}))


def test_experiments_can_extend_experiments(tmp_path):
    base = write(tmp_path / 'base.json', {'x': {'a': 1, 'b': 1}})
    first = write(tmp_path / 'first.json', {'extends': str(base), 'override': {'x': {'a': 2}}})
    second = write(tmp_path / 'second.json', {'extends': str(first), 'override': {'x': {'b': 3}}})
    assert config.load(second) == {'x': {'a': 2, 'b': 3}}


def test_added_keys_are_new_and_recorded_in_the_source(tmp_path):
    base = write(tmp_path / 'base.json', {'variants': {'a': {'k': 1}}})
    experiment = write(tmp_path / 'exp.json', {'extends': str(base), 'add': {'variants': {'b': {'k': 2}}}})
    cfg = config.load(experiment)
    assert cfg == {'variants': {'a': {'k': 1}, 'b': {'k': 2}}}
    assert cfg.source['add'] == {'variants': {'b': {'k': 2}}}


def test_an_added_key_that_exists_in_the_base_is_an_error(tmp_path):
    base = write(tmp_path / 'base.json', {'variants': {'a': {'k': 1}}})
    with pytest.raises(KeyError, match=r'variants\.a'):
        config.load(write(tmp_path / 'exp.json', {'extends': str(base), 'add': {'variants': {'a': {'k': 2}}}}))


def test_the_system_configuration_and_its_recipes_load_with_the_keys_they_extend():
    system = config.load('configs/pagenet_only_style_neural.json')
    five = config.load('configs/style_mild_5_epochs.json')
    fifteen = config.load('configs/style_mild_15_epochs.json')
    assert five['pagenet'] == system['pagenet']
    assert fifteen['pagenet'] == {**system['pagenet'], 'version': 'style_mild_15_epochs', 'epochs': 15,
                                  'drop_path': 0.1}
    assert fifteen.source['extends']['extends'] == {'path': 'configs/pagenet_only_style_neural.json'}
