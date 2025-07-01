import pygame
import random
import numpy as np

class SnakeGame:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        self.snake = [(self.width // 2, self.height // 2)]
        self.direction = (0, -1)  # Start: nach oben
        self.spawn_apple()
        self.score = 0
        self.frame = 0
        return self.get_state()

    def spawn_apple(self):
        while True:
            self.apple = (random.randint(0, self.width - 1),
                          random.randint(0, self.height - 1))
            if self.apple not in self.snake:
                break

    def step(self, action):
        self.change_direction(action)

        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        self.frame += 1
        reward = 0
        done = False

        # Kollision?
        if (
            new_head in self.snake
            or new_head[0] < 0 or new_head[0] >= self.width
            or new_head[1] < 0 or new_head[1] >= self.height
        ):
            reward = -1
            done = True
            return self.get_state(), reward, done, {}

        self.snake.insert(0, new_head)

        if new_head == self.apple:
            reward = 1
            self.score += 1
            self.spawn_apple()
        else:
            self.snake.pop()

        return self.get_state(), reward, done, {}

    def change_direction(self, action):
        # 0 = geradeaus, 1 = links, 2 = rechts (relativ)
        dx, dy = self.direction
        if action == 1:  # links
            self.direction = (-dy, dx)
        elif action == 2:  # rechts
            self.direction = (dy, -dx)
        # 0 = keine Änderung

    def get_state(self):
        head_x, head_y = self.snake[0]

        point_l = (head_x - self.direction[1], head_y + self.direction[0])
        point_r = (head_x + self.direction[1], head_y - self.direction[0])
        point_s = (head_x + self.direction[0], head_y + self.direction[1])

        def is_collision(point):
            return (
                point in self.snake
                or point[0] < 0 or point[0] >= self.width
                or point[1] < 0 or point[1] >= self.height
            )

        danger_straight = int(is_collision(point_s))
        danger_left = int(is_collision(point_l))
        danger_right = int(is_collision(point_r))

        dir_l = self.direction == (-1, 0)
        dir_r = self.direction == (1, 0)
        dir_u = self.direction == (0, -1)
        dir_d = self.direction == (0, 1)

        apple_x, apple_y = self.apple

        state = [
            danger_straight,
            danger_left,
            danger_right,
            int(dir_l),
            int(dir_r),
            int(dir_u),
            int(dir_d),
            int(apple_x < head_x),  # Apfel links
            int(apple_x > head_x),  # Apfel rechts
            int(apple_y < head_y),  # Apfel oben
            int(apple_y > head_y)   # Apfel unten
        ]

        return np.array(state, dtype=int)

    def render(self, block_size=30):
        pygame.init()
        screen = pygame.display.set_mode((self.width * block_size, self.height * block_size))
        pygame.display.set_caption("Snake DQN")

        screen.fill((0, 0, 0))
        for part in self.snake:
            pygame.draw.rect(screen, (0, 255, 0),
                             pygame.Rect(part[0] * block_size, part[1] * block_size, block_size, block_size))

        pygame.draw.rect(screen, (255, 0, 0),
                         pygame.Rect(self.apple[0] * block_size, self.apple[1] * block_size, block_size, block_size))

        pygame.display.flip()
        pygame.time.wait(100)
