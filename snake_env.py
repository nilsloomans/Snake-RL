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
        self.direction = (0, -1)  # nach oben
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
        new_head = (self.snake[0][0] + self.direction[0],
                    self.snake[0][1] + self.direction[1])

        self.frame += 1
        reward = 0
        done = False

        # Check collision
        if (new_head in self.snake) or not (0 <= new_head[0] < self.width) or not (0 <= new_head[1] < self.height):
            done = True
            reward = -1
            return self.get_state(), reward, done, {}

        self.snake.insert(0, new_head)

        # Check apple
        if new_head == self.apple:
            reward = 1
            self.score += 1
            self.spawn_apple()
        else:
            self.snake.pop()

        return self.get_state(), reward, done, {}

    def change_direction(self, action):
        # Action: 0 = straight, 1 = left, 2 = right
        dx, dy = self.direction
        if action == 1:  # left turn
            self.direction = (-dy, dx)
        elif action == 2:  # right turn
            self.direction = (dy, -dx)

    def get_state(self):
        head_x, head_y = self.snake[0]
        point_l = (head_x - self.direction[1], head_y + self.direction[0])  # links = -90°
        point_r = (head_x + self.direction[1], head_y - self.direction[0])  # rechts = +90°
        point_s = (head_x + self.direction[0], head_y + self.direction[1])  # geradeaus

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
            int(apple_x < head_x),  # apple left
            int(apple_x > head_x),  # apple right
            int(apple_y < head_y),  # apple up
            int(apple_y > head_y)   # apple down
        ]

        return np.array(state, dtype=int)


    def render(self, block_size=30):
        pygame.init()
        screen = pygame.display.set_mode((self.width * block_size, self.height * block_size))
        pygame.display.set_caption("Snake RL")

        screen.fill((0, 0, 0))
        for part in self.snake:
            pygame.draw.rect(screen, (0, 255, 0),
                             pygame.Rect(part[0] * block_size, part[1] * block_size, block_size, block_size))

        pygame.draw.rect(screen, (255, 0, 0),
                         pygame.Rect(self.apple[0] * block_size, self.apple[1] * block_size, block_size, block_size))

        pygame.display.flip()
        pygame.time.wait(100)