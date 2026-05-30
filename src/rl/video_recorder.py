# src/rl/video_recorder.py
import os
import numpy as np
import imageio

class EpisodeVideoRecorder:
    def __init__(self, save_dir, fps=5):
        self.save_dir = save_dir
        self.fps = fps
        self.frames = []
        self.episode_count = 0
        os.makedirs(save_dir, exist_ok=True)

    def reset(self):
        self.frames = []

    def capture(self, frame):
        self.frames.append(frame)

    @staticmethod
    def _normalize_frame(frame):
        frame = np.array(frame)

        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]

        return frame

    def save(self, tag="episode"):
        if len(self.frames) == 0:
            return

        frames = [self._normalize_frame(f) for f in self.frames]

        self.episode_count += 1

        filename = f"{tag}_{self.episode_count:06d}.mp4"
        path = os.path.join(self.save_dir, filename)

        imageio.mimsave(path, frames, fps=1, codec="libx264")

        self.frames = []