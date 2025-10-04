import pygame
import sys
import random

pygame.init()
WIDTH, HEIGHT = 400, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Player settings
player_size = 50
player_x = WIDTH // 2 - player_size // 2
player_y = HEIGHT - player_size - 30
player_speed = 10

# Obstacle settings
obstacle_size = 50
obstacle_speed = 7
obstacles = []

score = 0
font = pygame.font.SysFont(None, 36)

def spawn_obstacle():
    x = random.choice([WIDTH//4 - obstacle_size//2, WIDTH//2 - obstacle_size//2, 3*WIDTH//4 - obstacle_size//2])
    y = -obstacle_size
    obstacles.append([x, y])

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and player_x > 0:
        player_x -= player_speed
    if keys[pygame.K_RIGHT] and player_x < WIDTH - player_size:
        player_x += player_speed

    # Spawn obstacles
    if random.randint(0, 20) == 0:  # Change spawn frequency for tuning
        spawn_obstacle()

    # Move obstacles
    for obs in obstacles:
        obs[1] += obstacle_speed

    # Detect collision
    for obs in obstacles:
        if (player_x < obs[0] + obstacle_size and
            player_x + player_size > obs[0] and
            player_y < obs[1] + obstacle_size and
            player_y + player_size > obs[1]):
            print("Game Over! Score:", score)
            pygame.quit()
            sys.exit()

    obstacles = [obs for obs in obstacles if obs[1] < HEIGHT]

    score += 1

    screen.fill((40, 40, 40))
    pygame.draw.rect(screen, (0, 200, 255), (player_x, player_y, player_size, player_size))
    for obs in obstacles:
        pygame.draw.rect(screen, (255, 0, 0), (obs[0], obs[1], obstacle_size, obstacle_size))
    score_text = font.render(f"Score: {score}", True, (255, 255, 255))
    screen.blit(score_text, (10, 10))

    pygame.display.flip()
    clock.tick(60)

