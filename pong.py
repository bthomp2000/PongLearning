import random

PADDLE_HEIGHT=0.2

class State:
	def __init__(self,ball_x_pos=0.5,ball_y_pos=0.5,ball_x_velo=0.3,ball_y_velo=0.1,paddle_y_pos=0.5-PADDLE_HEIGHT/2,reward=0):
		self.ball_x_pos=ball_x_pos
		self.ball_y_pos=ball_y_pos
		self.ball_x_velo=ball_x_velo
		self.ball_y_velo=ball_y_velo
		self.paddle_y_pos=paddle_y_pos
		self.reward=reward

	def copy(self):
		copy = State()
		copy.ball_x_pos=self.ball_x_pos
		copy.ball_y_pos=self.ball_y_pos
		copy.ball_x_velo=self.ball_x_velo
		copy.ball_y_velo=self.ball_y_velo
		copy.paddle_y_pos=self.paddle_y_pos
		copy.reward=self.reward
		return copy

class Action:
	def __init__(self,UP= -0.04, DOWN=0.04, STAY=0.00):
		self.UP = UP
		self.DOWN = DOWN
		self.STAY = STAY

def descretize(state):
	discreteState = state.copy()
	discreteState.ball_x_pos=int(state.ball_x_pos*12)
	if discreteState.ball_x_pos==12:
		discreteState.ball_x_pos=11
	discreteState.ball_y_pos=int(state.ball_y_pos*12)
	if discreteState.ball_y_pos==12:
		discreteState.ball_y_pos=11
	discreteState.ball_x_velo=int(state.ball_x_velo/abs(state.ball_x_velo))
	if(abs(state.ball_y_velo)<0.015):
		discreteState.ball_y_velo=0
	else:
		discreteState.ball_y_velo=int(state.ball_y_velo/abs(state.ball_y_velo))
	discreteState.paddle_y_pos=int(12*state.paddle_y_pos/(1-PADDLE_HEIGHT))
	if state.paddle_y_pos==1-PADDLE_HEIGHT:
		discreteState.paddle_y_pos=11
	return discreteState


def transition(state,action):
	result = state.copy()
	if result.reward==-1:
		return result
	result.ball_x_pos+=state.ball_x_velo
	result.ball_y_pos+=state.ball_y_velo

	#update paddle
	result.paddle_y_pos+=action
	if result.paddle_y_pos<0:
		result.paddle_y_pos=0
	elif result.paddle_y_pos>1-PADDLE_HEIGHT:
		result.paddle_y_pos=1-PADDLE_HEIGHT

	#bounces
	if result.ball_y_pos < 0:
		result.ball_y_pos*=-1
		result.ball_y_velo=-state.ball_y_velo
	elif result.ball_y_pos > 1:
		result.ball_y_pos=2-result.ball_y_pos
		result.ball_y_velo=-state.ball_y_velo
	elif result.ball_x_pos < 0:
		result.ball_x_pos*=-1
		result.ball_x_velo=-state.ball_x_velo
	elif result.ball_x_pos > 1:
		#We have hit the paddle
		if (result.ball_y_pos>=state.paddle_y_pos and result.ball_y_pos<=state.paddle_y_pos+PADDLE_HEIGHT):
			result.ball_x_pos=2-result.ball_x_pos
			result.ball_x_velo=-state.ball_x_velo
			result.ball_x_velo+=random.uniform(-0.015,0.015)
			result.ball_y_velo+=random.uniform(-0.03,0.03)
			result.reward=1
		else:
			result.reward=-1
	return result




state = State()
action = Action()
result = transition(state,action.UP)
descretizedState = descretize(state)
print state.paddle_y_pos
print result.paddle_y_pos
print state.ball_y_pos
print result.ball_y_pos
print descretizedState.paddle_y_pos