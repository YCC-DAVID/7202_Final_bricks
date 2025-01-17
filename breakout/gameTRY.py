import pygame
import math
import random

# Constant for determining the screen size. 
SIZE_OF_THE_SCREEN = 424, 430
# Dimensions of the brikcs
HEIGHT_OF_BRICK  = 26
WIDTH_OF_BRICK   = 64
# Dimensions of the paddle
HEIGH_OF_PADDLE = 8
PADDLE_WIDTH  = 100
# Y coordinate for Paddle
PADDLE_Y = SIZE_OF_THE_SCREEN[1] - HEIGH_OF_PADDLE - 10
# Dimensions of the ball
BALL_DIAMETER = 12
BALL_RADIUS   = BALL_DIAMETER // 2
# X coordinate for Paddles
MAX_PADDLE_X = SIZE_OF_THE_SCREEN[0] - PADDLE_WIDTH
MAX_BALL_X   = SIZE_OF_THE_SCREEN[0] - BALL_DIAMETER
MAX_BALL_Y   = SIZE_OF_THE_SCREEN[1] - BALL_DIAMETER

# Color constants
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE  = (0, 0, 255)
COLOR_OF_BRICK = (153, 76, 0)
PADDLE_COLOR = (204,0,0)

FPS = 60
FPSCLOCK = pygame.time.Clock()

pygame.init() # Calling pygame module
screen = pygame.display.set_mode(SIZE_OF_THE_SCREEN)
pygame.display.set_caption(" BREAKOUT")
clock = pygame.time.Clock()


class Breakout:

    def __init__(self):
        self.capture = 0
        # Used 12-velocity to train
        self.ball_vel = [6,-6]
        self.paddle   = pygame.Rect(215, PADDLE_Y,PADDLE_WIDTH, HEIGH_OF_PADDLE)
        self.ball     = pygame.Rect(225,PADDLE_Y - BALL_DIAMETER,BALL_DIAMETER,BALL_DIAMETER)
        self.Hit_point = 0
        self.create_bricks()

    def create_bricks(self):
        y_ofs = 20
        self.bricks = []
        for i in range(5):
            x_ofs = 15
            for j in range(6):
                self.bricks.append(pygame.Rect(x_ofs,y_ofs,WIDTH_OF_BRICK,HEIGHT_OF_BRICK))
                x_ofs += WIDTH_OF_BRICK + 1
            y_ofs += HEIGHT_OF_BRICK + 1
        self.initial_bricks_count = len(self.bricks)

    def draw_bricks(self):
        for brick in self.bricks:
            pygame.draw.rect(screen, COLOR_OF_BRICK, brick)
    def draw_paddle(self):
        pygame.draw.rect(screen, PADDLE_COLOR, self.paddle)
    def draw_ball(self):
        pygame.draw.circle(screen, WHITE, (self.ball.left + BALL_RADIUS, self.ball.top + BALL_RADIUS), BALL_RADIUS)
        
    def check_input(self,input_action):
        # 0-LEFT, 1-Right
        if input_action[0] == 1:
        	# Used 12-velocity to train --> self.paddle.left -= 12
            self.paddle.left -= 12
            if self.paddle.left < 0:
                self.paddle.left = 0
        if input_action[1] == 1:
        	# Likewise 12
            self.paddle.left += 12
            if self.paddle.left > MAX_PADDLE_X:
                self.paddle.left = MAX_PADDLE_X
        if input_action[2] == 1:
        	# stay
            self.paddle.left += 0
            # if self.paddle.left > MAX_PADDLE_X:
            #     self.paddle.left = MAX_PADDLE_X

    def move_ball(self):
        self.ball.left += self.ball_vel[0]
        self.ball.top  += self.ball_vel[1]
        if self.ball.left <= 0:
            self.ball.left = 0
            self.ball_vel[0] = -self.ball_vel[0]
        elif self.ball.left >= MAX_BALL_X:
            self.ball.left = MAX_BALL_X
            self.ball_vel[0] = -self.ball_vel[0]
        if self.ball.top < 0:
            self.ball.top = 0
            self.ball_vel[1] = -self.ball_vel[1]
        elif self.ball.top >= MAX_BALL_Y:            
            self.ball.top = MAX_BALL_Y
            self.ball_vel[1] = -self.ball_vel[1]
              
    def take_action(self,input_action):

        pygame.event.pump()

        reward = 0.1
        terminal = False
        randNum = random.randint(0,1)

        # Get every event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        screen.fill(BLACK)
        self.check_input(input_action)
        self.move_ball()
        
        # Handle Collisions
        for brick in self.bricks:
            if self.ball.colliderect(brick):
                # reward = 2
                self.Hit_point += 1
                ball_center_x = self.ball.centerx
                ball_center_y = self.ball.centery

                # 砖块边缘坐标
                brick_left = brick.left
                brick_right = brick.right
                brick_top = brick.top
                brick_bottom = brick.bottom

                # 检测垂直方向碰撞
                if brick_left <= ball_center_x <= brick_right:
                    self.ball_vel[1] = -self.ball_vel[1]
                elif brick_top <= ball_center_y <= brick_bottom:
                    self.ball_vel[0] = -self.ball_vel[0]
                else:
                    self.ball_vel[1] = -self.ball_vel[1]

                self.bricks.remove(brick)
                break
        # bricks_destroyed = self.initial_bricks_count - len(self.bricks)-1
        # reward += bricks_destroyed * 0.1

        if len(self.bricks) == 0:
            self.terminal = True
            with open('simple_data2.txt', 'a') as file:
                file.write(str(self.Hit_point) + '\n')  # 将数据转换为字符串（如果需要的话），并添加换行符

            self.__init__()  
        if self.ball.colliderect(self.paddle):
            self.ball.top = PADDLE_Y - BALL_DIAMETER
            self.ball_vel[1] = -self.ball_vel[1]
            reward = 2 
        elif self.ball.top > self.paddle.top:
            terminal = True
            with open('simple_data2.txt', 'a') as file:
                file.write(str(self.Hit_point) + '\n')  # 将数据转换为字符串（如果需要的话），并添加换行符

            self.__init__()
            reward = -2

        self.draw_bricks()
        self.draw_ball()
        self.draw_paddle()
        # Capture screen and assign to image_data
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        # Update the screen
        pygame.display.update()
        FPSCLOCK.tick(FPS)
        return image_data, reward, terminal
