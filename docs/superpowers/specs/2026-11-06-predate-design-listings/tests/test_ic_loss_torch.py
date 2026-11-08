"""Tests for the per-day Pearson IC training loss; skipped when torch is not installed."""
import pytest

torch = pytest.importorskip("torch")

from ic_loss_torch import pearson_ic_loss  # noqa: E402


def test_perfect_correlation_gives_minus_one() -> None:
    label = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert pearson_ic_loss(label * 2 + 1, label).item() == pytest.approx(-1.0, abs=1e-5)


def test_inverse_correlation_gives_plus_one() -> None:
    label = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert pearson_ic_loss(-label, label).item() == pytest.approx(1.0, abs=1e-5)


def test_matches_numpy_corrcoef_on_a_known_case() -> None:
    pred = torch.tensor([0.1, 0.4, 0.2, 0.9, 0.3], dtype=torch.float64)
    label = torch.tensor([0.0, 1.0, 0.5, 0.7, 0.2], dtype=torch.float64)
    expected = -torch.corrcoef(torch.stack([pred, label]))[0, 1]
    assert pearson_ic_loss(pred, label).item() == pytest.approx(expected.item(), abs=1e-6)


def test_zero_variance_prediction_is_finite_with_finite_gradient() -> None:
    pred = torch.ones(5, requires_grad=True)
    loss = pearson_ic_loss(pred, torch.arange(5.0))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()


def test_shape_mismatch_and_tiny_cross_section_raise() -> None:
    with pytest.raises(ValueError, match="shape"):
        pearson_ic_loss(torch.ones(4), torch.ones(5))
    with pytest.raises(ValueError, match="at least 3"):
        pearson_ic_loss(torch.ones(2), torch.ones(2))
