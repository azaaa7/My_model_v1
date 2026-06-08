from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from .transforms import (
    add_gaussian_noise_snr,
    build_appearance_replay_augmenter,
    build_replay_augmenter,
    build_spatial_replay_augmenter,
    is_image_file,
    simulate_jpeg_compression_cv2,
    threshold_mask,
)


SampleList = Union[str, os.PathLike[str], Sequence[str], Sequence[Sequence[str]], np.ndarray]


class DistributedEvalSampler(Sampler):
    """Shard eval datasets across ranks without padding duplicate samples."""

    def __init__(self, dataset: Dataset, num_replicas: int, rank: int) -> None:
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        return (len(self.dataset) + self.num_replicas - 1 - self.rank) // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        return None


class VideoWindowSampler(Sampler):
    """Yield whole videos as ordered window sequences.

    Video order can be shuffled per epoch, but windows inside the same video
    remain chronological so a stateful FGM bank can be carried safely.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle_videos: bool = True,
        pad_to_equal_length: bool = False,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle_videos = bool(shuffle_videos)
        self.pad_to_equal_length = bool(pad_to_equal_length)
        self.seed = int(seed)
        self.epoch = 0
        self.video_ids = list(getattr(dataset, "window_video_order", []))
        self.video_to_indices = dict(getattr(dataset, "video_to_window_indices", {}))
        if self.num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}")
        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError(f"rank must be in [0, {self.num_replicas}), got {rank}")
        if not self.video_ids or not self.video_to_indices:
            raise ValueError("VideoWindowSampler requires a dataset with precomputed video windows")

    def _rank_video_ids(self) -> list[str]:
        video_ids = list(self.video_ids)
        if self.shuffle_videos:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(video_ids)
        return [video_id for i, video_id in enumerate(video_ids) if i % self.num_replicas == self.rank]

    def _rank_indices(self) -> list[int]:
        rank_videos: list[list[str]] = [[] for _ in range(self.num_replicas)]
        video_ids = list(self.video_ids)
        if self.shuffle_videos:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(video_ids)
        for i, video_id in enumerate(video_ids):
            rank_videos[i % self.num_replicas].append(video_id)

        rank_indices: list[list[int]] = []
        for videos in rank_videos:
            indices: list[int] = []
            for video_id in videos:
                indices.extend(self.video_to_indices[video_id])
            rank_indices.append(indices)

        if self.pad_to_equal_length and self.num_replicas > 1:
            max_len = max((len(indices) for indices in rank_indices), default=0)
            for indices in rank_indices:
                if not indices and max_len > 0:
                    raise ValueError("VideoWindowSampler cannot pad an empty rank; reduce nproc_per_node or add videos.")
                base = list(indices)
                while len(indices) < max_len:
                    indices.extend(base[: max_len - len(indices)])

        return rank_indices[self.rank]

    def __iter__(self):
        return iter(self._rank_indices())

    def __len__(self) -> int:
        return len(self._rank_indices())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def _numeric_key(path_or_name: str | os.PathLike[str]) -> int | str:
    """Extract trailing integer from filename for correct numeric sort.

    Ensures "10.png" follows "2.png" rather than string ordering.
    """
    name = Path(str(path_or_name)).stem
    nums = re.findall(r"\d+", name)
    if nums:
        return int(nums[-1])
    return name


def _is_path_like(value: object) -> bool:
    return isinstance(value, (str, os.PathLike))


def _rows_to_samples(data: np.ndarray | Sequence[Sequence[str]]) -> List[Tuple[str, str]]:
    result: List[Tuple[str, str]] = []
    for item in data:
        if len(item) < 2:
            raise ValueError(f"Each sample row must contain video_dir and mask_dir, got: {item}")
        video_dir, mask_dir = item[:2]
        result.append((str(video_dir), str(mask_dir)))
    return result


def _load_npy_samples(path: str | os.PathLike[str]) -> List[Tuple[str, str]]:
    data = np.load(path, allow_pickle=True)
    return _rows_to_samples(data)


def _load_samples(samples: SampleList) -> List[Tuple[str, str]]:
    if _is_path_like(samples):
        return _load_npy_samples(samples)

    if isinstance(samples, np.ndarray):
        if samples.ndim == 1 and all(_is_path_like(item) for item in samples.tolist()):
            result: List[Tuple[str, str]] = []
            for path in samples.tolist():
                result.extend(_load_npy_samples(path))
            return result
        return _rows_to_samples(samples)

    if all(_is_path_like(item) for item in samples):
        result: List[Tuple[str, str]] = []
        for path in samples:
            result.extend(_load_npy_samples(path))
        return result

    return _rows_to_samples(samples)


def _sample_indices(
    video_length: int,
    mode: str,
    num_frames: int = 5,
    val_num_frames: int = 0,
) -> List[int]:
    half = num_frames // 2

    if mode == "train":
        center = random.randint(0, video_length - 1)
        return [min(max(center + offset, 0), video_length - 1) for offset in range(-half, half + 1)]

    if mode == "val":
        if val_num_frames > 0:
            # 等间隔采样 val_num_frames 帧，保证每个视频帧数一致
            if video_length <= val_num_frames:
                return list(range(video_length))
            step = (video_length - 1) / (val_num_frames - 1) if val_num_frames > 1 else 0.0
            return [min(round(i * step), video_length - 1) for i in range(val_num_frames)]
        # fallback: 中心连续帧 (与 train 相同逻辑)
        center = video_length // 2
        return [min(max(center + offset, 0), video_length - 1) for offset in range(-half, half + 1)]

    if mode == "test":
        return list(range(video_length))

    raise ValueError(f"Unknown mode: {mode}")


def _sample_multi_clip_indices(
    video_length: int,
    mode: str,
    num_clips: int = 4,
    num_frames: int = 4,
    stride: int = 1,
) -> list[list[int]]:
    """Sample ``num_clips`` clips from a video, each of ``num_frames`` frames.

    Train: randomly sample non-overlapping starting positions, sorted chronologically.
           If the video is too short, pad by repeating the last frame.
    Val/Test: sequential sliding windows.

    Returns:
        List of clip index lists, e.g. [[0,1,2,3], [5,6,7,8], ...].
    """
    clip_len = num_frames * stride
    max_start = video_length - clip_len

    if mode == "train":
        if max_start < 0:
            # Video too short — pad with last frame repeats
            clips: list[list[int]] = []
            for _ in range(num_clips):
                clip: list[int] = []
                for i in range(num_frames):
                    clip.append(min(i * stride, video_length - 1))
                clips.append(clip)
            return clips

        starts = random.sample(
            range(0, max_start + 1),
            k=min(num_clips, max_start + 1),
        )
        starts = sorted(starts)

        # If fewer than num_clips possible starts, pad by repeating
        while len(starts) < num_clips:
            starts.append(random.randint(0, max_start))
        starts = sorted(starts)

        clips = []
        for s in starts:
            clip = [min(s + i * stride, video_length - 1) for i in range(num_frames)]
            clips.append(clip)
        return clips

    # Val / Test: sequential windows
    step = num_frames  # non-overlapping by default
    clips = []
    for start in range(0, video_length, step):
        clip = [min(start + i * stride, video_length - 1) for i in range(num_frames)]
        clips.append(clip)

    # Ensure exactly num_clips clips (pad with last clip if needed)
    if len(clips) > num_clips:
        # Evenly subsample
        indices = [round(i * (len(clips) - 1) / (num_clips - 1)) for i in range(num_clips)] if num_clips > 1 else [0]
        clips = [clips[i] for i in indices]
    elif len(clips) < num_clips:
        while len(clips) < num_clips:
            clips.append(clips[-1])  # repeat last clip
    return clips


def _sample_eval_windows(
    video_length: int,
    num_clips: int,
    num_frames: int,
    stride: int = 1,
) -> list[dict[str, list[list[int]] | list[list[bool]]]]:
    """Build sequential fixed-size windows for eval-time inference."""
    assert video_length > 0
    assert num_clips > 0
    assert num_frames > 0
    assert stride > 0

    all_clips: list[list[int]] = []
    all_valid: list[list[bool]] = []
    step = num_frames * stride

    for start in range(0, video_length, step):
        clip: list[int] = []
        valid: list[bool] = []

        for i in range(num_frames):
            idx = start + i * stride
            if idx < video_length:
                clip.append(idx)
                valid.append(True)
            else:
                clip.append(video_length - 1)
                valid.append(False)

        all_clips.append(clip)
        all_valid.append(valid)

    windows: list[dict[str, list[list[int]] | list[list[bool]]]] = []
    for i in range(0, len(all_clips), num_clips):
        window = all_clips[i:i + num_clips]
        valid_window = all_valid[i:i + num_clips]

        while len(window) < num_clips:
            window.append(window[-1])
            valid_window.append([False] * num_frames)

        windows.append({
            "frame_indices": window,
            "valid_mask": valid_window,
        })

    return windows


_sample_test_windows = _sample_eval_windows


def _validate_num_frames(num_frames: int, *, allow_even: bool = False) -> None:
    if isinstance(num_frames, bool) or not isinstance(num_frames, int):
        raise TypeError(f"num_frames must be an int, got {type(num_frames).__name__}")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be a positive integer, got {num_frames}")
    if not allow_even and num_frames % 2 == 0:
        raise ValueError(
            f"num_frames must be a positive odd integer in baseline mode, got {num_frames}. "
            f"Set use_tfcu_adapter=true or num_clips>1 to allow even frames."
        )


def _read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def _align_mask_to_frame(mask: np.ndarray, frame: np.ndarray) -> np.ndarray:
    frame_h, frame_w = frame.shape[:2]
    mask_h, mask_w = mask.shape[:2]
    if (mask_h, mask_w) == (frame_h, frame_w):
        return mask
    return cv2.resize(mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)


def _derive_sample_name(video_dir: str) -> str:
    p = Path(video_dir)
    if len(p.parents) >= 2:
        prefix = p.parents[1].name
    else:
        prefix = p.parent.name
    return f"{prefix}_{p.name}.jpg"


def _apply_replay_to_sequence(
    augmenter: A.ReplayCompose,
    frames: list[np.ndarray],
    masks: list[np.ndarray],
    replay=None,
):
    """Apply one replay transform consistently to a frame/mask sequence."""
    if not frames:
        return frames, masks, replay

    start = 0
    if replay is None:
        aug = augmenter(image=frames[0], mask=masks[0])
        replay = aug["replay"]
        frames[0], masks[0] = aug["image"], aug["mask"]
        start = 1

    for i in range(start, len(frames)):
        aug = A.ReplayCompose.replay(replay, image=frames[i], mask=masks[i])
        frames[i], masks[i] = aug["image"], aug["mask"]

    return frames, masks, replay


def _flatten_clip_indices(clip_indices) -> list[int]:
    return [int(idx) for clip in clip_indices for idx in clip]


def _reshape_clip_indices(flat: list[int], num_clips: int, num_frames: int) -> list[list[int]]:
    return [flat[i * num_frames:(i + 1) * num_frames] for i in range(num_clips)]


def _apply_temporal_index_augment(
    clip_indices,
    valid_mask,
    *,
    video_length: int,
    num_clips: int,
    num_frames: int,
    stride: int,
    cfg: dict,
) -> tuple[list[list[int]], list[list[bool]], dict[str, int]]:
    """Apply temporal robustness perturbations to frame indices.

    Frame drop removes positions from the temporal sequence, shifts later frames
    left, and pads from later real frames when available. Frame swap exchanges
    nearby temporal positions. Image/mask loading then follows these indices, so
    masks remain aligned with augmented frames.
    """
    flat = _flatten_clip_indices(clip_indices)
    flat_valid = [bool(v) for clip in valid_mask for v in clip]
    target_len = num_clips * num_frames
    stats = {"frame_drop_count": 0, "frame_swap_count": 0}
    if target_len <= 1 or video_length <= 0:
        return _reshape_clip_indices(flat, num_clips, num_frames), _reshape_clip_indices(flat_valid, num_clips, num_frames), stats

    drop_cfg = (cfg.get("frame_drop", {}) or {}) if cfg else {}
    if bool(drop_cfg.get("enabled", False)) and random.random() < float(drop_cfg.get("prob", 0.0)):
        valid_positions = [pos for pos, is_valid in enumerate(flat_valid) if is_valid]
        max_drops = max(1, int(drop_cfg.get("max_drops", 1)))
        max_drops = min(max_drops, max(0, len(valid_positions) - 1))
        if max_drops > 0:
            drop_count = random.randint(1, max_drops)
            drop_positions = set(random.sample(valid_positions, k=drop_count))
        else:
            drop_count = 0
            drop_positions = set()

        flat = [idx for pos, idx in enumerate(flat) if pos not in drop_positions]
        flat_valid = [is_valid for pos, is_valid in enumerate(flat_valid) if pos not in drop_positions]
        next_idx = flat[-1] if flat else 0
        while len(flat) < target_len:
            candidate = next_idx + max(1, int(stride))
            is_valid = candidate < video_length
            next_idx = min(candidate, video_length - 1)
            flat.append(next_idx)
            flat_valid.append(is_valid)
        stats["frame_drop_count"] = drop_count

    swap_cfg = (cfg.get("frame_swap", {}) or {}) if cfg else {}
    if bool(swap_cfg.get("enabled", False)) and random.random() < float(swap_cfg.get("prob", 0.0)):
        max_swaps = max(1, int(swap_cfg.get("max_swaps", 1)))
        radius = max(1, int(swap_cfg.get("local_radius", 2)))
        swap_count = random.randint(1, max_swaps)
        for _ in range(swap_count):
            i = random.randrange(target_len)
            j_min = max(0, i - radius)
            j_max = min(target_len - 1, i + radius)
            candidates = [j for j in range(j_min, j_max + 1) if j != i]
            if not candidates:
                continue
            j = random.choice(candidates)
            flat[i], flat[j] = flat[j], flat[i]
            flat_valid[i], flat_valid[j] = flat_valid[j], flat_valid[i]
            stats["frame_swap_count"] += 1

    return (
        _reshape_clip_indices(flat[:target_len], num_clips, num_frames),
        _reshape_clip_indices(flat_valid[:target_len], num_clips, num_frames),
        stats,
    )


class VideoInpaintingDataset(Dataset):
    """
    可配置的视频 inpainting 数据集。

    模式 1 — Baseline (num_clips=1):
        训练: return frames [T,3,H,W], center_mask [1,H,W], H, W, name
        验证: return frames [T,3,H,W], all_masks [T,1,H,W], H, W, name

    模式 2 — TFCU (num_clips>1):
        训练/验证/测试: return frames [N,T,3,H,W], masks [N,T,1,H,W], H, W, name
    """

    def __init__(
        self,
        samples: SampleList,
        mode: str = "train",
        input_size: int = 512,
        gt_ratio: int = 1,
        num_frames: int = 5,
        val_num_frames: int = 0,
        dataset_repeat: int = 1,
        augment_prob: float = 0.75,
        spatial_augment_prob: float | None = None,
        appearance_augment_prob: float | None = None,
        robust_noise_snr: int = 0,
        robust_jpeg_quality: int = 0,
        num_clips: int = 1,
        clip_stride: int = 1,
        use_tfcu_adapter: bool = False,
        test_max_clips: int = 4,
        train_full_video_windows: bool = False,
        val_full_video: bool = False,
        test_full_video: bool = True,
        temporal_augment: dict | None = None,
    ):
        self.use_tfcu_adapter = bool(use_tfcu_adapter)
        allow_even = self.use_tfcu_adapter or num_clips > 1
        _validate_num_frames(num_frames, allow_even=allow_even)

        self.samples = _load_samples(samples)
        self.mode = mode
        self.input_size = input_size
        self.gt_ratio = gt_ratio
        self.num_frames = num_frames
        self.val_num_frames = val_num_frames
        self.dataset_repeat = dataset_repeat
        self.augment_prob = augment_prob
        self.spatial_augment_prob = augment_prob if spatial_augment_prob is None else spatial_augment_prob
        self.appearance_augment_prob = augment_prob if appearance_augment_prob is None else appearance_augment_prob
        self.robust_noise_snr = robust_noise_snr
        self.robust_jpeg_quality = robust_jpeg_quality
        self.num_clips = num_clips
        self.clip_stride = clip_stride
        self.test_max_clips = test_max_clips
        self.train_full_video_windows = bool(train_full_video_windows)
        self.val_full_video = bool(val_full_video)
        self.test_full_video = bool(test_full_video)
        self.temporal_augment = temporal_augment or {}
        self.use_train_windows = self.mode == "train" and self.train_full_video_windows and self.num_clips > 0
        self.use_eval_windows = (
            self.num_clips > 1
            and (
                (self.mode == "val" and self.val_full_video)
                or (self.mode == "test" and self.test_full_video)
            )
        )

        self.to_tensor = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
        ])
        self.replay_aug = build_replay_augmenter()
        self.spatial_replay_aug = build_spatial_replay_augmenter()
        self.appearance_replay_aug = build_appearance_replay_augmenter()
        self.eval_items: list[dict[str, object]] | None = None
        self.window_video_order: list[str] = []
        self.video_to_window_indices: dict[str, list[int]] = {}

        if self.use_train_windows or self.use_eval_windows:
            self.eval_items = []
            repeats = self.dataset_repeat if self.use_train_windows else 1
            for repeat_idx in range(repeats):
                for sample_idx, (video_dir, _mask_dir) in enumerate(self.samples):
                    frame_list = sorted(
                        [p for p in os.listdir(video_dir) if is_image_file(p)],
                        key=_numeric_key,
                    )
                    video_length = len(frame_list)
                    if video_length <= 0:
                        continue

                    windows = _sample_eval_windows(
                        video_length=video_length,
                        num_clips=self.num_clips,
                        num_frames=self.num_frames,
                        stride=self.clip_stride,
                    )
                    name = _derive_sample_name(video_dir)
                    state_video_id = f"{name}#rep{repeat_idx}" if repeats > 1 else name
                    self.window_video_order.append(state_video_id)
                    self.video_to_window_indices[state_video_id] = []

                    for window_id, item in enumerate(windows):
                        global_idx = len(self.eval_items)
                        self.video_to_window_indices[state_video_id].append(global_idx)
                        self.eval_items.append({
                            "sample_idx": sample_idx,
                            "video_id": state_video_id,
                            "name": name,
                            "window_id": window_id,
                            "frame_indices": item["frame_indices"],
                            "valid_mask": item["valid_mask"],
                            "is_first_window": window_id == 0,
                            "is_last_window": window_id == len(windows) - 1,
                        })

    def __len__(self) -> int:
        if (self.use_train_windows or self.use_eval_windows) and self.eval_items is not None:
            return len(self.eval_items)
        return len(self.samples) * self.dataset_repeat

    def __getitem__(self, idx: int):
        if (self.use_train_windows or self.use_eval_windows) and self.eval_items is not None:
            item = self.eval_items[idx]
            sample_idx = int(item["sample_idx"])
            video_dir, mask_dir = self.samples[sample_idx]
            frame_list = sorted(
                [p for p in os.listdir(video_dir) if is_image_file(p)], key=_numeric_key,
            )
            mask_list = sorted(
                [p for p in os.listdir(mask_dir) if is_image_file(p)], key=_numeric_key,
            )

            if len(frame_list) != len(mask_list):
                raise ValueError(
                    f"Frame count and mask count mismatch in {video_dir}: "
                    f"{len(frame_list)} vs {len(mask_list)}"
                )

            return self._get_multi_clip_by_indices(
                video_dir=video_dir,
                mask_dir=mask_dir,
                frame_list=frame_list,
                mask_list=mask_list,
                clip_indices=item["frame_indices"],
                video_id=str(item["video_id"]),
                display_name=str(item["name"]),
                window_id=int(item["window_id"]),
                valid_mask=item["valid_mask"],
                is_first_window=bool(item["is_first_window"]),
                is_last_window=bool(item["is_last_window"]),
            )

        idx = idx % len(self.samples)  # 支持 dataset_repeat：取模映射到原始样本
        video_dir, mask_dir = self.samples[idx]
        frame_list = sorted(
            [p for p in os.listdir(video_dir) if is_image_file(p)], key=_numeric_key,
        )
        mask_list = sorted(
            [p for p in os.listdir(mask_dir) if is_image_file(p)], key=_numeric_key,
        )

        if len(frame_list) != len(mask_list):
            raise ValueError(
                f"Frame count and mask count mismatch in {video_dir}: "
                f"{len(frame_list)} vs {len(mask_list)}"
            )

        video_length = len(frame_list)
        name = _derive_sample_name(video_dir)

        # ── TFCU / multi-clip mode ────────────────────────────────────
        # Even num_clips=1 should stay in the TFCU path so num_frames=4
        # returns exactly one 4-frame clip instead of the legacy odd
        # center-window sampler returning 5 frames.
        if self.use_tfcu_adapter or self.num_clips > 1:
            return self._get_multi_clip(
                video_dir, mask_dir, frame_list, mask_list, video_length, name,
            )

        # ── Single-clip mode (original behaviour) ─────────────────────
        indices = _sample_indices(video_length, self.mode, self.num_frames, self.val_num_frames)

        frames: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        original_h, original_w = None, None

        for frame_idx in indices:
            frame_path = str(Path(video_dir) / frame_list[frame_idx])
            mask_path = str(Path(mask_dir) / mask_list[frame_idx])

            frame = _read_image(frame_path)
            mask = _read_image(mask_path)
            mask = _align_mask_to_frame(mask, frame)
            mask = threshold_mask(mask)

            if original_h is None:
                original_h, original_w = frame.shape[:2]

            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            masks.append(mask)

        if self.mode == "train" and random.random() < self.spatial_augment_prob:
            frames, masks, _ = _apply_replay_to_sequence(self.spatial_replay_aug, frames, masks)

        if self.mode == "train" and random.random() < self.appearance_augment_prob:
            frames, masks, _ = _apply_replay_to_sequence(self.appearance_replay_aug, frames, masks)

        if self.mode == "test" and self.robust_noise_snr > 0:
            frames = [add_gaussian_noise_snr(img, self.robust_noise_snr) for img in frames]

        if self.mode == "test" and 1 <= self.robust_jpeg_quality <= 100:
            frames = [simulate_jpeg_compression_cv2(img, self.robust_jpeg_quality) for img in frames]

        frame_tensors = []
        mask_tensors = []
        for img, mask in zip(frames, masks):
            img = cv2.resize(img, (self.input_size, self.input_size))

            if self.mode == "train":
                # Train: resize mask to model output resolution for efficient loss computation.
                mask = cv2.resize(mask, (self.input_size // self.gt_ratio, self.input_size // self.gt_ratio))
                mask = threshold_mask(mask)
            # val / test: keep mask at original (aligned) resolution.
            # align_logits_and_masks will upsample logits to match.

            img = img.astype(np.float32) / 255.0
            mask = mask.astype(np.float32) / 255.0

            frame_tensors.append(self.to_tensor(img).unsqueeze(0))
            mask_tensors.append(torch.from_numpy(mask[:, :, :1]).float().permute(2, 0, 1).unsqueeze(0))

        frames_out = torch.cat(frame_tensors, dim=0)
        masks_out = torch.cat(mask_tensors, dim=0)

        if self.mode == "train":
            return frames_out, masks_out[self.num_frames // 2], original_h, original_w, name
        return frames_out, masks_out, original_h, original_w, name

    # ------------------------------------------------------------------
    # Multi-clip sampling
    # ------------------------------------------------------------------

    def _get_multi_clip_by_indices(
        self,
        video_dir: str,
        mask_dir: str,
        frame_list: list[str],
        mask_list: list[str],
        clip_indices,
        video_id: str,
        display_name: str,
        window_id: int,
        valid_mask,
        is_first_window: bool,
        is_last_window: bool,
    ):
        """Load a fixed eval window without re-sampling clip indices."""
        temporal_stats = {"frame_drop_count": 0, "frame_swap_count": 0}
        if self.mode == "train":
            clip_indices, valid_mask, temporal_stats = _apply_temporal_index_augment(
                clip_indices,
                valid_mask,
                video_length=len(frame_list),
                num_clips=self.num_clips,
                num_frames=self.num_frames,
                stride=self.clip_stride,
                cfg=self.temporal_augment,
            )

        all_frames_np: list[list[np.ndarray]] = []
        all_masks_np: list[list[np.ndarray]] = []
        original_h, original_w = None, None

        for clip in clip_indices:
            frames: list[np.ndarray] = []
            masks: list[np.ndarray] = []

            for frame_idx in clip:
                frame_path = str(Path(video_dir) / frame_list[frame_idx])
                mask_path = str(Path(mask_dir) / mask_list[frame_idx])

                frame = _read_image(frame_path)
                mask = _read_image(mask_path)
                mask = _align_mask_to_frame(mask, frame)
                mask = threshold_mask(mask)

                if original_h is None:
                    original_h, original_w = frame.shape[:2]

                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                masks.append(mask)

            all_frames_np.append(frames)
            all_masks_np.append(masks)

        if self.mode == "train" and random.random() < self.spatial_augment_prob:
            spatial_replay = None
            for clip_idx in range(len(all_frames_np)):
                all_frames_np[clip_idx], all_masks_np[clip_idx], spatial_replay = _apply_replay_to_sequence(
                    self.spatial_replay_aug,
                    all_frames_np[clip_idx],
                    all_masks_np[clip_idx],
                    replay=spatial_replay,
                )

        if self.mode == "train":
            for clip_idx in range(len(all_frames_np)):
                if random.random() < self.appearance_augment_prob:
                    all_frames_np[clip_idx], all_masks_np[clip_idx], _ = _apply_replay_to_sequence(
                        self.appearance_replay_aug,
                        all_frames_np[clip_idx],
                        all_masks_np[clip_idx],
                    )

        all_frames: list[torch.Tensor] = []
        all_masks: list[torch.Tensor] = []

        for frames, masks in zip(all_frames_np, all_masks_np):
            if self.mode == "test" and self.robust_noise_snr > 0:
                frames = [add_gaussian_noise_snr(img, self.robust_noise_snr) for img in frames]
            if self.mode == "test" and 1 <= self.robust_jpeg_quality <= 100:
                frames = [simulate_jpeg_compression_cv2(img, self.robust_jpeg_quality) for img in frames]

            frame_tensors = []
            mask_tensors = []
            for img, mask in zip(frames, masks):
                img = cv2.resize(img, (self.input_size, self.input_size))
                if self.mode == "train":
                    mask = cv2.resize(mask, (self.input_size // self.gt_ratio, self.input_size // self.gt_ratio))
                    mask = threshold_mask(mask)
                img = img.astype(np.float32) / 255.0
                mask = mask.astype(np.float32) / 255.0

                frame_tensors.append(self.to_tensor(img).unsqueeze(0))
                mask_tensors.append(torch.from_numpy(mask[:, :, :1]).float().permute(2, 0, 1).unsqueeze(0))

            all_frames.append(torch.cat(frame_tensors, dim=0))
            all_masks.append(torch.cat(mask_tensors, dim=0))

        frames_out = torch.stack(all_frames, dim=0)
        masks_out = torch.stack(all_masks, dim=0)

        return {
            "images": frames_out,
            "masks": masks_out,
            "video_id": video_id,
            "window_id": window_id,
            "frame_indices": torch.tensor(clip_indices, dtype=torch.long),
            "valid_mask": torch.tensor(valid_mask, dtype=torch.bool),
            "is_first_window": is_first_window,
            "is_last_window": is_last_window,
            "original_h": original_h,
            "original_w": original_w,
            "name": display_name,
            **temporal_stats,
        }

    def _get_multi_clip(
        self,
        video_dir: str,
        mask_dir: str,
        frame_list: list[str],
        mask_list: list[str],
        video_length: int,
        name: str,
    ):
        """Sample ``num_clips`` clips from the same video, chronologically ordered.

        Test mode: num_clips auto-expands to cover all frames, but capped at
        ``test_max_clips`` to avoid OOM (default 4 = 16 frames at T=4).
        """
        actual_num_clips = self.num_clips
        if self.mode in ("test", "val"):
            max_clips = max(1, (video_length + self.num_frames - 1) // self.num_frames)
            cap = getattr(self, "test_max_clips", self.num_clips)
            actual_num_clips = min(max(self.num_clips, max_clips), cap)

        all_clip_indices = _sample_multi_clip_indices(
            video_length,
            self.mode,
            num_clips=actual_num_clips,
            num_frames=self.num_frames,
            stride=self.clip_stride,
        )

        all_frames_np: list[list[np.ndarray]] = []
        all_masks_np: list[list[np.ndarray]] = []
        original_h, original_w = None, None

        for clip_indices in all_clip_indices:
            frames: list[np.ndarray] = []
            masks: list[np.ndarray] = []

            for frame_idx in clip_indices:
                frame_path = str(Path(video_dir) / frame_list[frame_idx])
                mask_path = str(Path(mask_dir) / mask_list[frame_idx])

                frame = _read_image(frame_path)
                mask = _read_image(mask_path)
                mask = _align_mask_to_frame(mask, frame)
                mask = threshold_mask(mask)

                if original_h is None:
                    original_h, original_w = frame.shape[:2]

                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                masks.append(mask)

            all_frames_np.append(frames)
            all_masks_np.append(masks)

        if self.mode == "train" and random.random() < self.spatial_augment_prob:
            spatial_replay = None
            for clip_idx in range(len(all_frames_np)):
                all_frames_np[clip_idx], all_masks_np[clip_idx], spatial_replay = _apply_replay_to_sequence(
                    self.spatial_replay_aug,
                    all_frames_np[clip_idx],
                    all_masks_np[clip_idx],
                    replay=spatial_replay,
                )

        if self.mode == "train":
            for clip_idx in range(len(all_frames_np)):
                if random.random() < self.appearance_augment_prob:
                    all_frames_np[clip_idx], all_masks_np[clip_idx], _ = _apply_replay_to_sequence(
                        self.appearance_replay_aug,
                        all_frames_np[clip_idx],
                        all_masks_np[clip_idx],
                    )

        all_frames: list[torch.Tensor] = []   # each: [T, 3, H, W]
        all_masks: list[torch.Tensor] = []    # each: [T, 1, H, W]

        for frames, masks in zip(all_frames_np, all_masks_np):

            # Robustness perturbations (test only)
            if self.mode == "test" and self.robust_noise_snr > 0:
                frames = [add_gaussian_noise_snr(img, self.robust_noise_snr) for img in frames]
            if self.mode == "test" and 1 <= self.robust_jpeg_quality <= 100:
                frames = [simulate_jpeg_compression_cv2(img, self.robust_jpeg_quality) for img in frames]

            # Resize & convert to tensor
            frame_tensors = []
            mask_tensors = []
            for img, mask in zip(frames, masks):
                img = cv2.resize(img, (self.input_size, self.input_size))
                if self.mode == "train":
                    mask = cv2.resize(mask, (self.input_size // self.gt_ratio, self.input_size // self.gt_ratio))
                    mask = threshold_mask(mask)
                img = img.astype(np.float32) / 255.0
                mask = mask.astype(np.float32) / 255.0
                frame_tensors.append(self.to_tensor(img).unsqueeze(0))
                mask_tensors.append(torch.from_numpy(mask[:, :, :1]).float().permute(2, 0, 1).unsqueeze(0))

            all_frames.append(torch.cat(frame_tensors, dim=0))     # [T, 3, H, W]
            all_masks.append(torch.cat(mask_tensors, dim=0))       # [T, 1, H, W]

        frames_out = torch.stack(all_frames, dim=0)   # [N, T, 3, H, W]
        masks_out = torch.stack(all_masks, dim=0)     # [N, T, 1, H, W]

        # TFCU mode: always return full [N,T,*,*,*] masks for temporal loss
        # Baseline mode (num_clips=1): return center-frame mask for training
        if self.use_tfcu_adapter or self.num_clips > 1:
            return frames_out, masks_out, original_h, original_w, name
        if self.mode == "train":
            return frames_out, masks_out[self.num_frames // 2], original_h, original_w, name
        return frames_out, masks_out, original_h, original_w, name


def build_dataloader(
    samples: SampleList,
    mode: str = "train",
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool | None = None,
    pin_memory: bool = True,
    drop_last: bool | None = None,
    num_clips: int = 1,
    clip_stride: int = 1,
    use_tfcu_adapter: bool = False,
    test_max_clips: int = 4,
    train_full_video_windows: bool = False,
    val_full_video: bool = False,
    test_full_video: bool = True,
    temporal_augment: dict | None = None,
    seed: int = 0,
    spatial_augment_prob: float | None = None,
    appearance_augment_prob: float | None = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    **dataset_kwargs,
):
    dataset = VideoInpaintingDataset(
        samples=samples,
        mode=mode,
        num_clips=num_clips,
        clip_stride=clip_stride,
        use_tfcu_adapter=use_tfcu_adapter,
        test_max_clips=test_max_clips,
        train_full_video_windows=train_full_video_windows,
        val_full_video=val_full_video,
        test_full_video=test_full_video,
        temporal_augment=temporal_augment,
        spatial_augment_prob=spatial_augment_prob,
        appearance_augment_prob=appearance_augment_prob,
        **dataset_kwargs,
    )
    use_window_sampler = bool(getattr(dataset, "use_train_windows", False) or getattr(dataset, "use_eval_windows", False))
    if use_window_sampler and batch_size != 1:
        raise ValueError("Video-window FGM bank training/eval requires batch_size=1 so each batch owns one video state.")
    if shuffle is None:
        shuffle = mode == "train"
    if drop_last is None:
        drop_last = mode == "train" and not use_window_sampler
    sampler = None
    if use_window_sampler:
        sampler = VideoWindowSampler(
            dataset,
            num_replicas=world_size if distributed else 1,
            rank=rank if distributed else 0,
            shuffle_videos=bool(shuffle) if mode == "train" else False,
            pad_to_equal_length=bool(distributed and mode == "train"),
            seed=seed,
        )
        shuffle = False
    elif distributed:
        if mode == "train":
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                drop_last=drop_last,
            )
        else:
            sampler = DistributedEvalSampler(dataset, num_replicas=world_size, rank=rank)
        shuffle = False
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
