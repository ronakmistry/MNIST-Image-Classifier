import numpy as np
import pytest
from classifier import preprocess, split_indices, fixture, build_model, run, load_data, SHAPE, CLASSES


def test_validation_isolation_and_preprocessing():
    train, validation = split_indices(100)
    assert not set(train).intersection(validation)
    assert sorted(np.r_[train, validation]) == list(range(100))
    assert np.array_equal(train, split_indices(100)[0])
    pixels = np.full((2, *SHAPE), 255, dtype='uint8')
    assert np.all(preprocess(pixels) == 1)
    with pytest.raises(ValueError, match='uint8'):
        preprocess(preprocess(pixels))
    with pytest.raises(ValueError, match='Expected nonempty'):
        preprocess(np.zeros((2, 1), dtype='uint8'))


def test_model_shape_and_inference_has_no_augmentation():
    x = preprocess(fixture()[0][:2])
    model = build_model()
    first = np.asarray(model(x, training=False))
    assert first.shape == (2, CLASSES)
    np.testing.assert_allclose(first, model(x, training=False))
    np.testing.assert_allclose(first.sum(axis=1), 1, atol=1e-6)


def test_smoke_round_trip_and_missing_data(tmp_path):
    report = run(output=tmp_path)
    assert np.isfinite(report['test']['loss'])
    assert report['split_sizes'] == {'train': 18, 'validation': 2, 'test': 8}
    assert sum(map(sum, report['confusion_matrix'])) == 8
    with pytest.raises(FileNotFoundError, match='No dataset is downloaded'):
        load_data(None)
