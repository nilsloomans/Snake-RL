import pygame
import random
import numpy as np

BLOCK_SIZE = 20
GRID_SIZE = 20
WIDTH = HEIGHT = BLOCK_SIZE * GRID_SIZE

# Farben
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

class SnakeGame:
    def __init__(self):
        pygame.init()
        self.display = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = (1, 0)
        self.head = [GRID_SIZE // 2, GRID_SIZE // 2]
        self.snake = [self.head[:], [self.head[0] - 1, self.head[1]], [self.head[0] - 2, self.head[1]]]
        self.score = 0
        self.food = self.place_food()
        self.frame_iteration = 0
        return self.get_state()

    def place_food(self):
        while True:
            x = random.randint(0, GRID_SIZE - 1)
            y = random.randint(0, GRID_SIZE - 1)
            if [x, y] not in self.snake:
                return [x, y]

    def step(self, action):
        self.frame_iteration += 1

        # Richtungsänderung
        clock_wise = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        idx = clock_wise.index(self.direction)

        if action == 0:  # geradeaus
            new_dir = self.direction
        elif action == 1:  # rechts
            new_dir = clock_wise[(idx + 1) % 4]
        else:  # links
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir
        x, y = self.head
        dx, dy = self.direction
        self.head = [x + dx, y + dy]
        self.snake.insert(0, self.head[:])

        reward = -0.05
        done = False

        ate_food = self.head == self.food

        if ate_food:
            self.score += 1
            reward = 1.0
            self.food = self.place_food()
        else:
            self.snake.pop()

        if (self.head in self.snake[1:] or
            self.head[0] < 0 or self.head[0] >= GRID_SIZE or
            self.head[1] < 0 or self.head[1] >= GRID_SIZE):
            done = True
            reward = -1.0
            return self.get_state(), reward, done, {}

        old_dist = np.linalg.norm(np.array(self.snake[1]) - np.array(self.food))
        new_dist = np.linalg.norm(np.array(self.head) - np.array(self.food))

        if new_dist < old_dist:
            reward += 0.1
        else:
            reward -= 0.1

        def free_space(direction, max_depth=4):
            x, y = self.head
            dx, dy = direction
            count = 0
            for i in range(1, max_depth + 1):
                nx, ny = x + dx * i, y + dy * i
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and [nx, ny] not in self.snake:
                    count += 1
                else:
                    break
            return count / max_depth

        forward_dir = self.direction
        free_ahead = free_space(forward_dir)
        if free_ahead < 0.25:
            reward -= 0.3

        return self.get_state(), reward, done, {}

    def render(self):
        self.display.fill(BLACK)
        for part in self.snake:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(part[0] * BLOCK_SIZE, part[1] * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food[0] * BLOCK_SIZE, self.food[1] * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        pygame.display.flip()
        self.clock.tick(10)

    def get_state(self):
        head_x, head_y = self.head
        dir_x, dir_y = self.direction

        def danger_at(pos):
            return (
                pos in self.snake or
                pos[0] < 0 or pos[0] >= GRID_SIZE or
                pos[1] < 0 or pos[1] >= GRID_SIZE
            )

        left = (-dir_y, dir_x)
        right = (dir_y, -dir_x)
        front = (dir_x, dir_y)

        front_block = [head_x + front[0], head_y + front[1]]
        right_block = [head_x + right[0], head_y + right[1]]
        left_block = [head_x + left[0], head_y + left[1]]

        danger_straight = danger_at(front_block)
        danger_right = danger_at(right_block)
        danger_left = danger_at(left_block)

        dir_up = dir_y == -1
        dir_down = dir_y == 1
        dir_left = dir_x == -1
        dir_right = dir_x == 1

        food_left = self.food[0] < head_x
        food_right = self.food[0] > head_x
        food_up = self.food[1] < head_y
        food_down = self.food[1] > head_y

        snake_length = len(self.snake) / (GRID_SIZE * GRID_SIZE)
        tail = self.snake[-1]
        dist_to_tail = np.linalg.norm(np.array(self.head) - np.array(tail)) / np.sqrt(GRID_SIZE**2 * 2)

        def free_space(direction, max_depth=5):
            x, y = head_x, head_y
            dx, dy = direction
            count = 0
            for i in range(1, max_depth + 1):
                nx, ny = x + dx * i, y + dy * i
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and [nx, ny] not in self.snake:
                    count += 1
                else:
                    break
            return count / max_depth

        free_straight = free_space(front)
        free_left = free_space(left)
        free_right = free_space(right)

        reachable_area = self.flood_fill_area(self.head)

        state = [
            int(danger_straight),
            int(danger_right),
            int(danger_left),

            int(dir_left),
            int(dir_right),
            int(dir_up),
            int(dir_down),

            int(food_left),
            int(food_right),
            int(food_up),
            int(food_down),

            snake_length,
            dist_to_tail,
            free_straight,
            free_left,
            free_right,
            reachable_area  # 🆕 hinzugefügt
        ]

        return np.array(state, dtype=float)

    def flood_fill_area(self, start_pos, max_depth=100):
        visited = set()
        queue = [tuple(start_pos)]
        area = 0

        while queue and area < max_depth:
            x, y = queue.pop(0)
            if (x, y) in visited:
                continue
            if x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE:
                continue
            if [x, y] in self.snake:
                continue

            visited.add((x, y))
            area += 1

            queue.append((x + 1, y))
            queue.append((x - 1, y))
            queue.append((x, y + 1))
            queue.append((x, y - 1))

        return area / (GRID_SIZE * GRID_SIZE)