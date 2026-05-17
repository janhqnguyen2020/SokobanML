import matplotlib.pyplot as plt


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


def print_step(step, action, action_name, reward, done):
    """Print one executed planner action in a readable format."""
    print(
        f"Step {step + 1}: "
        f"action={action}, "
        f"name={action_name}, "
        f"reward={reward}, "
        f"done={done}"
    )