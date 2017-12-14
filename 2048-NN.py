from game2048 import Game
import sys
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import DenseLayer
from copy import deepcopy
floatX = theano.config.floatX
input_var = T.tensor4()
target_var = T.ivector()
_ = lasagne.layers.InputLayer(shape=(None, 4, 4, 20), input_var=input_var)
_ = DenseLayer(_, num_units=900, nonlinearity=lasagne.nonlinearities.rectify)
_ = DenseLayer(_, num_units=300, nonlinearity=lasagne.nonlinearities.rectify)
_ = DenseLayer(_, num_units=200, nonlinearity=lasagne.nonlinearities.rectify)
l_out = DenseLayer(_, num_units=4, nonlinearity=lasagne.nonlinearities.softmax)
prediction = lasagne.layers.get_output(l_out)
P = theano.function([input_var], prediction)

loss = lasagne.objectives.squared_error(prediction, target_var).mean()/2
train_fn = theano.function([input_var, target_var], loss, updates=updates)
loss_fn = theano.function([input_var, target_var], loss)
accuracy_fn =theano.function([input_var, target_var], accuracy)

table ={2**(i+1):i for i in range(20)}
def make_input(grid, d=0):
    g0 = np.rot90(grid, -d)
    r = np.zeros(shape=(4,4,20), dtype=floatX)
    for i in range(4):
        for j in range(4):
            v = g0[i, j]
            if v:
                r[i, j][table[v]]=1
    return r
#with np.load('model-0.npz') as f:
#     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
#lasagne.layers.set_all_param_values(l_out, param_values)
def Vchange(grid, v):
    g0 = grid
    g1 = g0[::-1,:,:]
    g2 = g0[:,::-1,:]
    g3 = g2[::-1,:,:]
    r0 = grid.swapaxes(0,1)
    r1 = r0[::-1,:,:]
    r2 = r0[:,::-1,:]
    r3 = r2[::-1,:,:]
    xtrain = np.array([g0,g1,g2,g3,r0,r1,r2,r3], dtype=floatX)
    ytrain = np.array([v]*8, dtype=floatX)
    train_fn(xtrain, ytrain)

N =100
s=0
slen=0
stat = {}
for i in range(N):
    game = Game()
    game_len = 0
    while not game.end and game.max()<=2048:
        board = np.array([make_input(game.grid)], dtype=floatX)
        p = P(board)
        #print 'p.shape: ', p.shape
        #moves = np.argsort(p)[::-1]
        moves = np.argsort(P(board)[0])[::-1]
        p_sort = np.sort(P(board)[0])[::-1]
        prev_score = game.score
        for m in moves:
            if game.move(m):
                v = 2*(game.score-prev_score)+p_sort[m]
                Vchange(deepcopy(game.grid), v)
                break
        game_len+=1
    print (i, game_len, game.max(), game.score)
   # s+= game.score
   # slen+=game_len
   # m = game.max()
   # stat[m]=stat.get(m,0)+1
#print(s/N, slen/N, stat)
N =1000
s=0
slen=0
stat = {}
#for i in range(N):
#    game = Game()
#    game_len = 0
#    while not game.end:
#        board = np.array([make_input(game.grid)], dtype=floatX)
#        moves = np.argsort(P(board)[0])[::-1]
#        for m in moves:
#            if game.move(m):
#                break
#        game_len+=1
#    s+= game.score
#    slen+=game_len
#    m = game.max()
#    stat[m]=stat.get(m,0)+1
#print(s/N, slen/N, stat)
# from IPython.display import clear_output
#game = Game()
#game_len = 0
#game.display()
#while not game.end:
#    board = np.array([make_input(game.grid)], dtype=floatX)
#    moves = np.argsort(P(board)[0])[::-1]
#    for m in moves:
#        if game.move(m):
#            break
#    game_len+=1
#    #clear_output()
#    print("\n%d"%game_len, u"lurd"[m])
#    game.display()
#    sys.stdout.flush()
