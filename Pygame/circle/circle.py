
import pygame
import math
import numpy as np

def Init(size):
    global _size
    global _radius
    _size = np.array(size)
    _radius = _size[0]
    if _size[1] < _radius:
        _radius = _size[1]
    _radius -= 1
    _radius //= 2

def Update(deltaTime):
    pass

def DrawLine(screen, x1, y1, x2, y2):
    pygame.draw.line(screen, (255,255,255), (x1,y1), (x2,y2))

def Render(screen):
    global _size
    global _radius

    # Assignment #1
    # In this space write code to draw a perfect circle on the screen using only the DrawLine call above
    # The circle must be perfectly centered in the window even if the dimensions of the window found in _size are changed
    # The circle must have the radius given by the global variable _radius
    # The circle must not be filled in. I.e. it must be a white circular line with black inside and out
    # Hint: use the known equation for a circle r^2 = x^2 * y^2
    # Hint: try every x value in a loop to find y at that x then plot them all
    # You may NOT use any drawing code other than DrawLine
    # You may not import any module except the ones imported above
    # You may not use any code copied from the internet (write your own code so you become a better programmer)
    # You may not modify any code outside of this Render function
    
    # Your code here
def Render(screen):
    #centre
    o1 = _size[0] // 2
    o2 = _size[1] // 2

    # imlementing the loop
    for x in range(-_radius, _radius + 1):
        y = int(math.sqrt(_radius ** 2 - x ** 2))  # finding y using circle equation r^2 = x^2 + y^2

        # Drawline top and bottom half
        DrawLine(screen, o1 + x, o2 + y, o1 + x, o2 + y)
        DrawLine(screen, o1 + x, o2 - y, o1 + x, o2 - y)

        # Drawline left and right
        DrawLine(screen, o1 + y, o2 + x, o1 + y, o2 + x)
        DrawLine(screen, o1 - y, o2 + x, o1 - y, o2 + x)

def Close():
    pass