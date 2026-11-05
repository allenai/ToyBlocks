"""--use_calibrated_cameras disables dataset intrinsics ingestion and sets env cfg."""

from __future__ import annotations

from types import SimpleNamespace


def _apply_use_calibrated_cameras_policy(args: SimpleNamespace) -> None:
    """Mirror blocksuite.eval.run_eval.evaluate() calibrated-camera handling."""
    if args.use_calibrated_cameras:
        args.ingest_camera_intrinsics = False
        args.env_cfg.use_calibrated_cameras = True


def test_use_calibrated_cameras_clears_ingest_and_sets_env_cfg() -> None:
    args = SimpleNamespace(
        use_calibrated_cameras=True,
        ingest_camera_intrinsics=True,
        env_cfg=SimpleNamespace(use_calibrated_cameras=False),
    )
    _apply_use_calibrated_cameras_policy(args)
    assert args.ingest_camera_intrinsics is False
    assert args.env_cfg.use_calibrated_cameras is True


def test_use_calibrated_cameras_off_leaves_ingest_and_env_cfg() -> None:
    args = SimpleNamespace(
        use_calibrated_cameras=False,
        ingest_camera_intrinsics=True,
        env_cfg=SimpleNamespace(use_calibrated_cameras=False),
    )
    _apply_use_calibrated_cameras_policy(args)
    assert args.ingest_camera_intrinsics is True
    assert args.env_cfg.use_calibrated_cameras is False
