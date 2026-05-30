# src/rl/video_wrapper.py
import gym
import numpy as np

class VideoWrapper(gym.Wrapper):
    """Wraps a Gym environment to record frames for training videos."""
    
    def __init__(self, env, recorder=None):
        super().__init__(env)  # important: gym.Wrapper will do type checks
        self.recorder = recorder

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        if self.recorder:
            self.recorder.reset()
            frame = self.render(mode="rgb_array")
            self.recorder.capture(frame)
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if self.recorder:
            frame = self.render(mode="rgb_array")

            frame = np.array(frame)
            if frame.ndim == 2:
                frame = np.stack([frame]*3, axis=-1)
            elif frame.shape[2] == 4:
                frame = frame[:, :, :3]
            self.recorder.capture(frame)
            if done:
                self.recorder.save(tag="train_episode")
        return obs, reward, done, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()
    
    def seed(self, seed=None):
        return self.env.seed(seed)

    def __getattr__(self, name):
        return getattr(self.env, name)