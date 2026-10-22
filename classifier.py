"""Original CNN experiment with explicit local data and isolated evaluation."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
from pathlib import Path
import argparse
import hashlib
import json
import platform
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

DATASET = 'mnist'
SHAPE = (28, 28, 1)
CLASSES = 10


def preprocess(images):
    images = np.asarray(images)
    if DATASET == 'mnist' and images.ndim == 3:
        images = images[..., None]
    if images.ndim != 4 or images.shape[1:] != SHAPE or len(images) == 0:
        raise ValueError(f'Expected nonempty (N, {SHAPE}) images, got {images.shape}')
    if images.dtype != np.uint8:
        raise ValueError('Expected raw uint8 pixels; normalization is applied exactly once')
    return images.astype('float32') / 255.0


def split_indices(size, seed=42):
    if size < 10:
        raise ValueError('Supply at least 10 training images for a validation split')
    indices = np.random.default_rng(seed).permutation(size)
    count = max(1, int(size * .1))
    return indices[count:], indices[:count]


def load_data(path):
    if not path or not Path(path).is_file():
        raise FileNotFoundError('Supply --data with a local NPZ containing x_train, y_train, x_test, y_test. No dataset is downloaded; see README.')
    with np.load(path, allow_pickle=False) as data:
        arrays = [data[key] for key in ('x_train', 'y_train', 'x_test', 'y_test')]
    return arrays


def fixture(seed=42):
    rng = np.random.default_rng(seed)
    # Random pixels and labels validate mechanics only; they are not a benchmark.
    return (rng.integers(0, 256, (20, *SHAPE), dtype='uint8'),
            np.arange(20) % CLASSES,
            rng.integers(0, 256, (8, *SHAPE), dtype='uint8'),
            np.arange(8) % CLASSES)


def build_model(seed=42):
    inputs = keras.Input(SHAPE)
    x = inputs
    if DATASET == 'cifar100':
        # Preserve the original augmentation, including vertical flips. The README
        # explains why this may be unsuitable for some natural-image categories.
        x = layers.RandomFlip('horizontal_and_vertical', seed=seed)(x)
        x = layers.RandomRotation(15/360, fill_mode='nearest', seed=seed)(x)
        x = layers.RandomTranslation(.1, .1, fill_mode='nearest', seed=seed)(x)
        for block, filters in enumerate((32, 64, 128)):
            x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(filters, 3, padding='same', activation='relu',
                              kernel_regularizer=keras.regularizers.l2(1e-4))(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D()(x)
            x = layers.Dropout(.2 + .1 * block)(x)
    else:
        for filters in (28, 56):
            x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
            x = layers.MaxPooling2D()(x)
    outputs = layers.Dense(CLASSES, activation='softmax')(layers.Flatten()(x))
    model = keras.Model(inputs, outputs)
    # The old `decay` argument implemented inverse-time learning-rate decay.
    schedule = keras.optimizers.schedules.InverseTimeDecay(.0005, decay_steps=1, decay_rate=1e-6)
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=schedule),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def labels(values, count):
    values = np.asarray(values).reshape(-1)
    if len(values) != count or not np.issubdtype(values.dtype, np.integer):
        raise ValueError('Supply one integer class label per image')
    if np.any(values < 0) or np.any(values >= CLASSES):
        raise ValueError(f'Labels must be between 0 and {CLASSES - 1}')
    return values


def epoch_pass(model, x, y, batch_size, training):
    model.reset_metrics()
    call = model.train_on_batch if training else model.test_on_batch
    totals, count = {}, 0
    for start in range(0, len(x), batch_size):
        labels_batch = y[start:start+batch_size]
        metrics = call(x[start:start+batch_size], labels_batch, return_dict=True)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * len(labels_batch)
        count += len(labels_batch)
    return {key: total / count for key, total in totals.items()}


def run(smoke=True, data=None, output='artifacts/smoke', epochs=10, batch_size=64, seed=42):
    if epochs < 1 or batch_size < 1:
        raise ValueError('epochs and batch_size must be positive')
    keras.utils.set_random_seed(seed)
    x_train, y_train, x_test, y_test = fixture(seed) if smoke else load_data(data)
    x_train, x_test = preprocess(x_train), preprocess(x_test)
    y_train, y_test = labels(y_train, len(x_train)), labels(y_test, len(x_test))
    train, validation = split_indices(len(x_train), seed)
    model = build_model(seed)
    epochs = 1 if smoke else epochs
    history = []
    for epoch in range(epochs):
        order = np.random.default_rng(seed + epoch).permutation(train)
        history.append({'epoch': epoch + 1,
                        'train': epoch_pass(model, x_train[order], y_train[order], batch_size, True),
                        'validation': epoch_pass(model, x_train[validation], y_train[validation], batch_size, False)})
    # The test partition is used only after fitting; augmentation is inactive here.
    test = epoch_pass(model, x_test, y_test, batch_size, False)
    predictions = np.asarray(model(x_test, training=False)).argmax(axis=1)
    confusion = np.zeros((CLASSES, CLASSES), dtype=int)
    np.add.at(confusion, (y_test, predictions), 1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / 'model.keras')
    restored = keras.models.load_model(output / 'model.keras')
    np.testing.assert_allclose(model(x_test, training=False), restored(x_test, training=False), atol=1e-6)
    np.savez_compressed(output / 'split_indices.npz', train=train, validation=validation)
    errors = np.flatnonzero(predictions != y_test)[:9]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(3, 3, figsize=(7, 7))
    for ax in axes.flat:
        ax.axis('off')
    for ax, index in zip(axes.flat, errors):
        ax.imshow(x_test[index].squeeze(), cmap='gray' if DATASET == 'mnist' else None)
        ax.set_title(f'true {y_test[index]}, predicted {predictions[index]}')
    figure.suptitle('Fixture errors: random data, not model quality' if smoke else 'Held-out test errors')
    figure.tight_layout()
    figure.savefig(output / 'errors.png')
    plt.close(figure)
    report = {'status': 'fixture-only; not research evidence' if smoke else 'real-data rerun',
              'configuration': {'dataset': DATASET, 'seed': seed, 'epochs': epochs, 'batch_size': batch_size,
                                'normalization': 'uint8 / 255', 'validation_fraction': .1},
              'dataset_sha256': None if smoke else hashlib.sha256(Path(data).read_bytes()).hexdigest(),
              'split_sizes': {'train': len(train), 'validation': len(validation), 'test': len(x_test)},
              'history': history, 'test': test, 'confusion_matrix': confusion.tolist(),
              'environment': {'python': platform.python_version(), 'tensorflow': tf.__version__,
                              'keras': keras.__version__, 'numpy': np.__version__}}
    (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['smoke', 'train'])
    parser.add_argument('--data')
    parser.add_argument('--output', default='artifacts/smoke')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.command == 'smoke', args.data, args.output, args.epochs, args.batch_size, args.seed), indent=2))
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.exit(2, f'{error}\n')
