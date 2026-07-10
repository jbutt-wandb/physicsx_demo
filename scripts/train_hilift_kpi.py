from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def find_repo_root() -> Path:
    candidates = [Path.cwd(), Path.cwd() / "physicsx_demo", *Path.cwd().parents]
    return next(p for p in candidates if (p / "src" / "physicsx_demo").exists())


REPO_ROOT = find_repo_root()
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from physicsx_demo.utils.hilift_kpi import (  # noqa: E402
    HILIFT_TARGET_COLUMNS,
    HiLiftKpiDataset,
    denormalize_kpi_targets,
    kpi_frame_summary,
    prepare_hilift_kpi_bundle,
)
from physicsx_demo.utils.model import KpiTokenMLP, TinyTransolverKpi, count_params  # noqa: E402


def rmse_by_target(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    return {
        name: float(np.sqrt(np.mean((pred[:, idx] - true[:, idx]) ** 2)))
        for idx, name in enumerate(HILIFT_TARGET_COLUMNS)
    }


def evaluate(model: torch.nn.Module, loader: DataLoader, stats, device: torch.device) -> tuple[float, dict[str, float]]:
    model.eval()
    losses = []
    preds = []
    targets = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            losses.append(F.mse_loss(pred, yb).item())
            preds.append(pred.cpu())
            targets.append(yb.cpu())
    pred_norm = torch.cat(preds, dim=0)
    target_norm = torch.cat(targets, dim=0)
    pred_phys = denormalize_kpi_targets(pred_norm, stats)
    target_phys = denormalize_kpi_targets(target_norm, stats)
    return float(np.mean(losses)), rmse_by_target(pred_phys, target_phys)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small TinyTransolver KPI surrogate on HiLiftAeroML CSVs.")
    parser.add_argument("--data-dir", default="data/hiliftaeroml_subset")
    parser.add_argument("--n-geometries", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline", action="store_true", help="Train the pooled-token MLP baseline instead.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = prepare_hilift_kpi_bundle(args.data_dir, n_geometries=args.n_geometries, seed=args.seed)
    print(f"loaded HiLiftAeroML KPI data from {Path(args.data_dir).resolve()}")
    print(f"all:   {kpi_frame_summary(bundle.frame)}")
    print(f"train: {kpi_frame_summary(bundle.train)}")
    print(f"val:   {kpi_frame_summary(bundle.val)}")
    print(f"test:  {kpi_frame_summary(bundle.test)}")
    print()

    train_loader = DataLoader(
        HiLiftKpiDataset(bundle.train, bundle.stats),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(HiLiftKpiDataset(bundle.val, bundle.stats), batch_size=args.batch_size)
    test_loader = DataLoader(HiLiftKpiDataset(bundle.test, bundle.stats), batch_size=args.batch_size)

    model: torch.nn.Module
    model = KpiTokenMLP() if args.baseline else TinyTransolverKpi()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    print(f"model: {model.__class__.__name__}   params: {count_params(model):,}   device: {device}")

    best_state = None
    best_val = float("inf")
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        val_loss, val_rmse = evaluate(model, val_loader, bundle.stats, device)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            rmse_text = " ".join(f"{name}_rmse={value:.5f}" for name, value in val_rmse.items())
            print(f"epoch {epoch:03d} train={np.mean(train_losses):.5f} val={val_loss:.5f} {rmse_text}")

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_rmse = evaluate(model, test_loader, bundle.stats, device)
    print()
    print(f"best_epoch: {best_epoch}   best_val_mse: {best_val:.5f}")
    print(f"test_mse: {test_loss:.5f}")
    for name, value in test_rmse.items():
        print(f"test/{name}_rmse: {value:.6f}")


if __name__ == "__main__":
    main()
