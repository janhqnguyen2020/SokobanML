# src/utils/show_ui.py
import matplotlib.pyplot as plt
import imageio
import os
import numpy as np

class GifRecorder:
    def __init__(self, path, fps=5):
        self.path = path
        self.fps = fps
        self.frames = []

    def add_frame(self, fig):
        """Capture current matplotlib figure as RGB array."""
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frame = buf[:, :, :3].copy()   # force copy so buffer doesn't overwrite
        self.frames.append(frame)

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        imageio.mimsave(self.path, self.frames, fps=self.fps)
        
def create_plot(obs, title):
    """Create one matplotlib window to display Sokoban frames."""
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 3))
    image = ax.imshow(obs)
    ax.axis("off")
    ax.set_title(title)
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig, ax, image

def update_plot(fig, ax, image, obs, title, delay):
    """Update the existing matplotlib window with a new frame."""
    image.set_data(obs)
    ax.set_title(title)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(delay)


def finish_plot():
    """Keep the final frame visible and close interactive mode."""
    plt.ioff()
    plt.show()


def build_title(algo, map_type, map_name, step, reward, status=""):
    """Build a consistent display title for all solver visualizations."""
    parts = [algo, map_type, map_name, f"Step {step}", f"Reward: {reward:.2f}"]
    if status:
        parts.append(status)
    return " | ".join(parts)


def print_step(step, action, action_name, reward, done):
    """Print one executed planner action in a readable format."""
    print(
        f"Step {step + 1}: "
        f"action={action}, "
        f"name={action_name}, "
        f"reward={reward}, "
        f"done={done}"
    )

