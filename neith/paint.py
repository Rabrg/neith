import pygame
import numpy as np
from pygame.locals import QUIT
from pygame.surfarray import array2d
from neith import preprocess
from neith import dataset
from neith import network

resolution = (1280, 720)
background_color = (255, 255, 255)
draw_color = (0, 0, 0)
draw_thickness = 7
left_mouse_button = (1, 0, 0)
start_position = (0, 0)

pygame.init()
screen = pygame.display.set_mode(resolution)
background = pygame.Surface(screen.get_size())
background.fill(background_color)

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            exit()
        elif event.type == pygame.MOUSEMOTION:
            end_position = pygame.mouse.get_pos()
            if pygame.mouse.get_pressed() == left_mouse_button:
                pygame.draw.line(background, draw_color, start_position, end_position, draw_thickness)
            start_position = end_position
        elif event.type == pygame.MOUSEBUTTONUP:
            pixels = np.asarray(array2d(background))
            pixels = np.divide(pixels, 16777215)
            pixels = pixels.transpose()
            chars = preprocess.equation_char_list(pixels)
            chars = chars.reshape(chars.shape[0], 32, 32, 1)
            pred = network.model.predict_classes(chars, verbose=0)
            pred_str = str()
            for p in pred:
                pred_str += dataset.CLASS_INDEX[p]
            pred_str = pred_str.replace('star', '*').replace('slash', '/')
            print(pred_str)
            if (pred_str[-1].isdigit() or pred_str[-1] is ')') and pred_str.count('(') is pred_str.count(')'):
                print(pred_str + '=' + str(eval(pred_str)))
            # for i, char in enumerate(chars):
            #     scipy.misc.imsave(')_' + str(i) + '.png', char)
    screen.blit(background, (0, 0))
    pygame.display.flip()
