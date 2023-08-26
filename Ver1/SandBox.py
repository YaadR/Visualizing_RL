import pygame
import time

pygame.init()
# Define screen dimensions
width = 400
height = 500

# Create the screen
screen = pygame.display.set_mode((width, height))

# Define colors
green = (0, 255, 0)
red = (255, 0, 0)
yellow = (255, 255, 0)

# Set font
font = pygame.font.Font(None, 22)

# Initialize clock
clock = pygame.time.Clock()

# Initial values
vol = 0
volb = 20
volp = 0
length = 0
pTime = time.time()


running = True
while running:
    screen.fill((0, 0, 0))  # Clear the screen

    # Update values
    vol = vol + 1 if vol <250 else vol   # Replace with your logic to update the volume
    length += 1  # Replace with your logic to update the length

    t = vol / 250
    color = (int((1-t)*255),int(t*255),0)


    # Draw percentage bar
    # pygame.draw.rect(screen, color, (50, int(volb - vol), 35, vol), 0)  # Percentage bar
    pygame.draw.rect(screen, color, (10, 10, int(t*120), 40), 0)  # Percentage bar
    pygame.draw.rect(screen, color, (10, 10, 120, 40), 3)  # Outline of bar


    volp = round((vol/250)*100,2)

    # Draw text
    text = font.render(f'{int(volp)}%', True, yellow)
    screen.blit(text, (60, 25))
    title = font.render('Certainty Level', True, yellow)
    screen.blit(title, (140, 25))


    pygame.display.flip()  # Update the display

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    clock.tick(60)  # Control the frame rate

pygame.quit()
