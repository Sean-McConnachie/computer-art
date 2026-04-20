from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from perlin_numpy import (
    generate_fractal_noise_2d, generate_fractal_noise_3d,
    generate_perlin_noise_2d, generate_perlin_noise_3d
)
from tqdm import tqdm

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

resolutions = {
    "480p": (854, 480),
    "1080p": (1920, 1080),
    "4k": (3840, 2160)
}


def hsb_to_rgb(hsb: np.ndarray) -> np.ndarray:
    hue = hsb[..., 0]
    saturation = hsb[..., 1]
    brightness = hsb[..., 2]
    C = brightness * saturation
    H = hue / 60.0
    X = C * (1 - np.abs(H % 2 - 1))
    sector_idx = (H % 6).astype(int)
    # match sector_idx:
    #     case 0:
    #         i0, i1, i2 = C, X, 0
    #     case 1:
    #         i0, i1, i2 = X, C, 0
    #     case 2:
    #         i0, i1, i2 = 0, C, X
    #     case 3:
    #         i0, i1, i2 = 0, X, C
    #     case 4:
    #         i0, i1, i2 = X, 0, C
    #     case 5:
    #         i0, i1, i2 = C, 0, X
    interm = np.zeros_like(hsb)

    # 0th element
    interm[:, :, 0] = np.where((sector_idx == 0) | (sector_idx == 5), C, interm[:, :, 0])
    interm[:, :, 0] = np.where((sector_idx == 1) | (sector_idx == 4), X, interm[:, :, 0])
    interm[:, :, 0] = np.where((sector_idx == 2) | (sector_idx == 3), 0, interm[:, :, 0])
    # 1st element
    interm[:, :, 1] = np.where((sector_idx == 1) | (sector_idx == 2), C, interm[:, :, 1])
    interm[:, :, 1] = np.where((sector_idx == 0) | (sector_idx == 3), X, interm[:, :, 1])
    interm[:, :, 1] = np.where((sector_idx == 4) | (sector_idx == 5), 0, interm[:, :, 1])
    # 2nd element
    interm[:, :, 2] = np.where((sector_idx == 3) | (sector_idx == 4), C, interm[:, :, 2])
    interm[:, :, 2] = np.where((sector_idx == 2) | (sector_idx == 5), X, interm[:, :, 2])
    interm[:, :, 2] = np.where((sector_idx == 0) | (sector_idx == 1), 0, interm[:, :, 2])

    m = brightness - C
    rgb = interm + np.stack([m, m, m], axis=2)
    return rgb


def calculate_closest_octave(get: int, res: int, octave: int):
    multiplier = res * 2**(octave - 1)
    return ((get + multiplier - 1) // multiplier) * multiplier


def fractal_noise_2d(
    shape: tuple[int, int],
    res: int, octaves: int,
    lacunarity: float, persistence: float,
    rng: np.random.Generator
):
    height, width = shape
    noise_height = calculate_closest_octave(height, res[0], octaves)
    noise_width = calculate_closest_octave(width, res[1], octaves)
    noise = generate_fractal_noise_2d(
        shape=(noise_height, noise_width),
        res=res,
        octaves=octaves,
        lacunarity=lacunarity,
        persistence=persistence,
        rng=rng
    )
    noise = noise[:height, :width]
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return noise


def fractal_noise_3d(
    shape: tuple[int, int, int],
    res: int, octaves: int,
    lacunarity: float, persistence: float,
    rng: np.random.Generator
):
    depth, height, width = shape
    noise_depth = calculate_closest_octave(depth, res[0], octaves)
    noise_height = calculate_closest_octave(height, res[1], octaves)
    noise_width = calculate_closest_octave(width, res[2], octaves)
    noise = generate_fractal_noise_3d(
        shape=(noise_depth, noise_height, noise_width),
        res=res,
        octaves=octaves,
        lacunarity=lacunarity,
        persistence=persistence,
        rng=rng
    )
    noise = noise[:depth, :height, :width]
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return noise


class FlowyGradientVideo:
    def __init__(self, width: int, height: int, num_frames: int):
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.frames = np.ones((self.num_frames, self.height, self.width, 3)).astype(np.float32)

    def fractal_noise_vid_1(self):
        # create fractal noise
        print("Generating fractal noise for video frames...")
        noise = fractal_noise_3d(
            shape=(self.num_frames, self.height, self.width),
            res=(1, 8, 8),
            octaves=3,
            lacunarity=1,
            persistence=0.5,
            rng=rng
        )

        # threshold noise
        n_cols = 32
        thresholds = np.linspace(0, 1.0, n_cols + 1)
        for f in tqdm(range(self.num_frames), desc="Thresholding noise for video frames"):
            for i in range(n_cols):
                thres_lb, thres_ub = thresholds[i], thresholds[i+1]
                hue = thresholds[i]
                self.frames[f, :, :, 0] = np.where(
                    (noise[f] >= thres_lb) & (noise[f] <= thres_ub),
                    hue * 360.0,
                    self.frames[f, :, :, 0]
                )
        
        for f in tqdm(range(self.num_frames), desc="Converting frames to RGB"):
            self.frames[f] = hsb_to_rgb(self.frames[f])

    def fractal_noise_vid_2(self):
        # create fractal noise
        print("Generating fractal noise for video frames...")
        noise = fractal_noise_3d(
            shape=(self.num_frames, self.height, self.width),
            res=(1, 8, 8),
            octaves=3,
            lacunarity=1,
            persistence=0.5,
            rng=rng
        )
  
        # threshold noise
        n_cols = 32

        col_thresholds = np.linspace(0.0, 1.0, n_cols + 1)
        col_thresholds_1 = col_thresholds[:n_cols//2]
        col_thresholds_2 = col_thresholds[n_cols//2:]
        col_thresholds = [val for pair in zip(col_thresholds_1, col_thresholds_2) for val in pair]

        noise_thresholds = np.linspace(0, 1.0, n_cols + 1)
        for f in tqdm(range(self.num_frames), desc="Thresholding noise for video frames"):
            for i in range(n_cols):
                thres_lb, thres_ub = noise_thresholds[i], noise_thresholds[i+1]
                hue = col_thresholds[i]
                self.frames[f, :, :, 0] = np.where(
                    (noise[f] >= thres_lb) & (noise[f] <= thres_ub),
                    hue * 360.0,
                    self.frames[f, :, :, 0]
                )

        for f in tqdm(range(self.num_frames), desc="Converting frames to RGB"):
            self.frames[f] = hsb_to_rgb(self.frames[f])
        
    def save_mp4(self, path: str) -> None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, 30.0, (self.width, self.height))
        for frame in tqdm(self.frames, desc="Saving video frames"):
            rgb_255 = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
            bgr_255 = cv2.cvtColor(rgb_255, cv2.COLOR_RGB2BGR)
            out.write(bgr_255)
        out.release()
        print(f"Wrote {path} ({self.width}×{self.height}, {self.num_frames} frames)")        

    def save_png(self, path: str) -> None:
        """
        Saves the first frame as a PNG for quick preview.
        Assumes self.frames are in ranges [0, 1]
        """
        rgb_255 = (np.clip(self.frames[0], 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(rgb_255).save(path)
        print(f"Wrote {path} ({self.width}×{self.height})")



if __name__ == "__main__":
    # res = "480p"
    res = "1080p"
    res = "4k"
    img = True
    num_frames = 120 if not img else 1

    # flowy_vid = FlowyGradientVideo(*resolutions[res], num_frames=num_frames)
    # flowy_vid.fractal_noise_vid_1()
    # flowy_vid.save_mp4("fractal_noise_vid_1.mp4")
    # flowy_vid.save_png("fractal_noise_vid_1.png")

    flowy_vid = FlowyGradientVideo(*resolutions[res], num_frames=num_frames)
    flowy_vid.fractal_noise_vid_2()
    # flowy_vid.save_mp4("fractal_noise_vid_2.mp4")
    flowy_vid.save_png("fractal_noise_vid_2.png")