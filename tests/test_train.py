import torch
from torch.utils.data import DataLoader, TensorDataset

import espdlx
import espdlx.layers as L


def _tiny_model():
    return espdlx.Model(
        [L.Conv2d(1, 4, 1), L.ReLU(), L.Flatten(), L.Linear(64, 2)],
        name="tiny_clf",
    )


def _toy_dataset(n=60):
    torch.manual_seed(0)
    x = torch.randn(n, 1, 4, 4)
    y = (x.mean(dim=(1, 2, 3)) > 0).long()
    return TensorDataset(x, y)


def test_fit_splits_80_20_when_no_val():
    model = _tiny_model()
    ds = _toy_dataset(50)
    history = model.fit(ds, epochs=1, batch_size=10, verbose=False, device="cpu")
    assert len(history["loss"]) == 1
    assert len(history["val_accuracy"]) == 1
    assert 0.0 <= history["val_accuracy"][0] <= 1.0


def test_fit_with_explicit_val_loader():
    model = _tiny_model()
    ds = _toy_dataset(40)
    train_ld = DataLoader(ds, batch_size=8, shuffle=True)
    val_ld = DataLoader(ds, batch_size=8)
    history = model.fit(train_ld, val_ld, epochs=2, verbose=False, device="cpu")
    assert len(history["loss"]) == 2
    assert len(history["val_accuracy"]) == 2


def test_fit_reuses_loader_batch_size_on_split():
    model = _tiny_model()
    ds = _toy_dataset(50)
    ld = DataLoader(ds, batch_size=8, shuffle=True)
    history = model.fit(ld, epochs=1, verbose=False, device="cpu")
    assert len(history["val_accuracy"]) == 1


def test_evaluate_returns_fraction():
    model = _tiny_model()
    ds = _toy_dataset(20)
    acc = model.evaluate(ds, batch_size=8, verbose=False, device="cpu")
    assert 0.0 <= acc <= 1.0
