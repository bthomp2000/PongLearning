import random
import time

PADDLE_HEIGHT=0.2
EXPLORE_THRESHOLD=15
DECAY_RATE=0.9
LEARNING_CONSTANT = 15


class State:
	def __init__(self,ball_x_pos=0.5,ball_y_pos=0.5,ball_x_velo=0.03,ball_y_velo=0.01,paddle_y_pos=0.5-PADDLE_HEIGHT/2,reward=0):
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

	def toTuple(self):
		return (self.ball_x_pos,self.ball_y_pos,self.ball_x_velo,self.ball_y_velo,self.paddle_y_pos)

class Action:
	def __init__(self,UP= -0.04, DOWN=0.04, STAY=0.00):
		self.UP = UP
		self.DOWN = DOWN
		self.STAY = STAY

action = Action()
Q = {} # (ball_x_pos,ball_y_pos,ball_x_velo,ball_y_velo,paddle_y_pos,action):q_value
N = {} # (ball_x_pos,ball_y_pos,ball_x_velo,ball_y_velo,paddle_y_pos,action): count

def descretize(state):
	# print state.toTuple(),": ",
	discreteState = state.copy()
	discreteState.ball_x_pos=int(state.ball_x_pos*12)
	if state.ball_x_pos==1:
		discreteState.ball_x_pos=11
	discreteState.ball_y_pos=int(state.ball_y_pos*12)
	if discreteState.ball_y_pos==12:
		discreteState.ball_y_pos=11
	discreteState.ball_x_velo=int(state.ball_x_velo/abs(state.ball_x_velo))
	if(abs(state.ball_y_velo)<0.015):
		discreteState.ball_y_velo=0
	else:
		discreteState.ball_y_velo=int(state.ball_y_velo/abs(state.ball_y_velo))
	discreteState.paddle_y_pos=int(12.0*state.paddle_y_pos/(1.0-PADDLE_HEIGHT))
	if state.paddle_y_pos==1-PADDLE_HEIGHT:
		discreteState.paddle_y_pos=11
	# print discreteState.toTuple()
	return discreteState

def getKey(state,action):
	key = list(state.toTuple())
	key.append(action)
	key = tuple(key)
	return key

def transition(state,action):
	key = getKey(descretize(state),action)
	# print state.toTuple()
	# print "action: ",action
	N[key]+=1
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
	if  result.ball_x_pos < 0:
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

def initDicts():
	a = Action()
	for ball_x in range(12):
		for ball_y in range(12):
			for x_velo in range(-1,2,2):
				for y_velo in range(-1,2):
					for paddle_pos in range(12):
						actions = [a.UP,a.STAY,a.DOWN]
						for action in actions:
							key = (ball_x,ball_y,x_velo,y_velo,paddle_pos,action)
							N[key]=0.0
							Q[key]=0.0
							if (ball_x==11):
								if (ball_y==paddle_pos):
									Q[key] = 1.0
							# 	else:
							# 		Q[key] = -1.0
	Q[(-1,-1,-1,-1,-1)]=-1.0


def exploration(state,action):
	key = getKey(state,action)
	q_value = Q[key]
	count = N[key]
	# print key,": ",q_value
	if count < EXPLORE_THRESHOLD:
		return 10
	return q_value

def printState(state):
	for i in range(12):
		print '|',
		for j in range(12):
			if state.ball_y_pos==i and state.ball_x_pos==j:
				print '*',
			if state.paddle_y_pos==i and j==11:
				print '|',
			else:
				print ' ',
		print ''
	print '------------------------\n\n'

def tdUpdate(curr_state,next_action,next_state):
	# curr_key = tuple(list(curr_state.toTuple()).append(next_action))
	action = Action()
	curr_key = getKey(curr_state,next_action)
	# print next_state.toTuple()
	if getKey(next_state,action.UP) not in Q:
		# print "failed"
		alpha = LEARNING_CONSTANT/(LEARNING_CONSTANT+N[curr_key])
		# print "before: ",Q[curr_key]
		Q[curr_key]=Q[curr_key] + alpha * (curr_state.reward + DECAY_RATE * -1 - Q[curr_key])
		# print "after: ",Q[curr_key]
		return
	actions = [action.UP,action.STAY,action.DOWN]
	max_next_q = -10
	for a in actions:
		key = getKey(next_state,a)
		# q_val = -1
		q_val = Q[key]
		if q_val > max_next_q:
			max_next_q = q_val

	# print "max next q: ",max_next_q
	# if next_state.reward==1:
	# 	print "reward 1"	
	alpha = LEARNING_CONSTANT/(LEARNING_CONSTANT+N[curr_key])
	# print Q[curr_key]
	Q[curr_key]=Q[curr_key] + alpha * (curr_state.reward + DECAY_RATE * max_next_q - Q[curr_key])
	# print Q[curr_key]
def Q_Learning_Training():
	#Running 20,000 games
	average = []
	for i in range(50000):
		first = True
		numPaddleHits = 0
		curr_state = State()
		prev_discrete_state = descretize(curr_state)
		while curr_state.reward!=-1:
			# 
			next_possible_actions = [exploration(descretize(curr_state), action.UP), exploration(descretize(curr_state), action.STAY), exploration(descretize(curr_state), action.DOWN)]
			# print descretize(curr_state).toTuple(),": ",next_possible_actions
			next_action = action.UP
			max_value = next_possible_actions[0]
			if next_possible_actions[1] > max_value:
				next_action = action.STAY
				max_value = next_possible_actions[1]
			if next_possible_actions[2] > max_value:
				# print "down"
				next_action = action.DOWN
				max_value = next_possible_actions[2]

			next_state = transition(curr_state,next_action)
			# print next_state.reward
			# if next_state.ball_x_pos > 1:
				# print "greater than one"
			if descretize(curr_state).toTuple() != prev_discrete_state.toTuple():
				# print "reward of 1"
				
				if(prev_discrete_state.ball_x_pos==11 and descretize(curr_state).ball_x_pos<11):
					numPaddleHits+=1
				prev_discrete_state=descretize(curr_state)

			# if(not first):
			tdUpdate(descretize(curr_state),next_action,descretize(next_state))
			curr_state=next_state

				# time.sleep(0.1)
				# printState(descretize(curr_state))
		# print i
		# print numPaddleHits
		average.append(numPaddleHits)
		if i%1000==0:
			print "Training iteration ",i,": average of ",sum(average)/float(len(average))," bounces"
	print sum(average)/len(average)
		# print Q

def Q_Learning_Testing():
	average = []
	print "Testing Data"
	for i in range(10000):
		numPaddleHits = 0
		curr_state = State()
		prev_discrete_state = descretize(curr_state)
		while curr_state.reward != -1:
			next_state = transition(curr_state,next_action)
			if descretize(curr_state).toTuple() != prev_discrete_state.toTuple():
				if(prev_discrete_state.ball_x_pos==11 and descretize(curr_state).ball_x_pos<11):
					numPaddleHits+=1
				prev_discrete_state=descretize(curr_state)
			curr_state=next_state
		average.append(numPaddleHits)
		if i%1000==0:
			print "Testing iteration ",i,": average of ",sum(average)/float(len(average))," bounces"
	print sum(average)/len(average)

initDicts()
Q_Learning_Training()
print Q.values()