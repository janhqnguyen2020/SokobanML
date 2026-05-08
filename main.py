import gym
import gym_sokoban
import time
import warnings

import pygame
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def draw_frame(env, screen, scale):
    """
    Render gym-sokoban as an rgb_array and display it using pygame.
    """
    frame = env.render(mode="rgb_array")

    # gym-sokoban returns image as: height x width x RGB
    # pygame expects: width x height x RGB
    frame_for_pygame = np.transpose(frame, (1, 0, 2))

    surface = pygame.surfarray.make_surface(frame_for_pygame)

    height, width, _ = frame.shape
    surface = pygame.transform.scale(surface, (width * scale, height * scale))

    screen.blit(surface, (0, 0))
    pygame.display.flip()


def main():
    env_name = "Sokoban-v2"
    env = gym.make(env_name)

    action_lookup = env.unwrapped.get_action_lookup()
    print(f"Created environment: {env_name}")

    observation = env.reset()

    pygame.init()

    scale = 4

    # First render only to get image size
    first_frame = env.render(mode="rgb_array")
    height, width, _ = first_frame.shape

    screen = pygame.display.set_mode((width * scale, height * scale))
    pygame.display.set_caption("gym-sokoban visual test")

    draw_frame(env, screen, scale)

    running = True

    for t in range(100):
        if not running:
            break

        # This keeps the pygame window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = env.action_space.sample()

        # Slow down so you can observe it
        time.sleep(1)

        observation, reward, done, info = env.step(action)

        print(action_lookup[action], reward, done, info)

        # Actually display the updated frame
        draw_frame(env, screen, scale)

        if done:
            print(f"Episode finished after {t + 1} timesteps")
            break

    print("Finished. Window will stay open for 10 seconds.")
    time.sleep(10)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()