'''
This is a sample class that you can use to control the mouse pointer.
It uses the pyautogui library. You can set the precision for mouse movement
(how much the mouse moves) and the speed (how fast it moves) by changing 
precision_dict and speed_dict.
Calling the move function with the x and y output of the gaze estimation model
will move the pointer.
This class is provided to help get you started; you can choose whether you want to use it or create your own from scratch.
'''
import pyautogui
import math
import time

pyautogui.FAILSAFE = False
class MouseController:
    def __init__(self, args):
        precision_dict={'high':100, 'low':1000, 'medium':500}
        speed_dict={'fast':0.5, 'slow':1, 'medium':0.75}
        self.precision=precision_dict[args.precision]
        self.speed=speed_dict[args.speed]
        self.prev_x = None
        self.prev_y = None

    def move(self, x, y):
        # get current mouse position
        # get screen size
        mouse_pos_x = pyautogui.position().x
        mouse_pos_y = pyautogui.position().y
        screen_size_x = pyautogui.size().width
        screen_size_y = pyautogui.size().height

        # reset the mouse position if it goes above the screen size
        if  mouse_pos_x > screen_size_x and mouse_pos_y > screen_size_y:
            pyautogui.moveTo(screen_size_x, screen_size_y)
        elif mouse_pos_x < 0 and mouse_pos_y > screen_size_y:
            pyautogui.moveTo(0, screen_size_y)
        elif mouse_pos_x > screen_size_x and mouse_pos_y < 0:
            pyautogui.moveTo(screen_size_x, 0)
        elif mouse_pos_x < 0 and mouse_pos_y < 0:
            pyautogui.moveTo(0, 0)
        elif mouse_pos_y > screen_size_y:
            pyautogui.moveTo(mouse_pos_x, screen_size_y)
        elif mouse_pos_y < 0:
            pyautogui.moveTo(mouse_pos_x, 0)
        elif mouse_pos_x > screen_size_x:
            pyautogui.moveTo(screen_size_x, mouse_pos_y)
        elif mouse_pos_x < 0:
            pyautogui.moveTo(0, mouse_pos_y)

        diff_x = 0
        diff_y = 0
        move_x = 0
        move_y = 0
        if self.prev_x:
            diff_x = self.move_factor_x - x*screen_size_x
            diff_y = self.move_factor_y - y*screen_size_y
            move_x = math.ceil(self.prev_x*self.precision) - math.ceil(x*self.precision)
            move_y = math.ceil(1*self.prev_y*self.precision) - math.ceil(1*y*self.precision)

        self.prev_x = x
        self.prev_y = y
        self.move_factor_x = x*screen_size_x
        self.move_factor_y = y*screen_size_y

        if not abs(diff_x) >= 350:
            move_x = 0
        
        if not abs(diff_y) >= 100:
            move_y = 0
        pyautogui.moveRel(move_x, move_y, duration=self.speed)
