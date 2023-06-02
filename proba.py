import random

import chess as c
import numpy

import search
from constants import dataset_path
from learning import evaluate_on_dataset,loss_fn,optimizer,load_dataset,get_y_true,predict_on_dataset
def get_board():
    b = c.Board()
    b.clear()
    b.set_piece_at(piece=c.Piece(piece_type=c.KING,color=c.WHITE),square=c.F5)
    b.set_piece_at(piece=c.Piece(piece_type=c.KING,color=c.BLACK),square=c.B8)
    b.set_piece_at(piece=c.Piece(piece_type=c.PAWN,color=c.BLACK),square=c.D2)
    b.set_piece_at(piece=c.Piece(piece_type=c.BISHOP,color=c.BLACK),square=c.E8)
    b.set_piece_at(piece=c.Piece(piece_type=c.QUEEN,color=c.BLACK),square=c.E1)
    b.turn = True

    return b

def get_board():
    b = c.Board()
    b.clear()
    b.set_piece_at(piece=c.Piece(piece_type=c.KING,color=c.WHITE),square=c.B1)
    b.set_piece_at(piece=c.Piece(piece_type=c.KING,color=c.BLACK),square=c.D7)
    b.set_piece_at(piece=c.Piece(piece_type=c.ROOK,color=c.BLACK),square=c.G8)
    b.turn = True

    return b

if __name__ == '__main__':
    #from learning import learn_init_from_zero,custom_learning
    #learn_init_from_zero()
    #custom_learning()
    import os.path
    from model_chooser import model
    from constants import model_path,model_weights_path
    if os.path.exists(model_path):
        print('loading model weights...')
        model.load_weights(model_weights_path)
        print('model weights loaded')
    if os.path.exists(dataset_path):
        print('loading dataset...')
        load_dataset(dataset_path)
        print('dataset loaded')
    model.compile(loss=loss_fn, metrics=['mean_squared_error'],optimizer=optimizer)
    #print(model.get_weights())
    #for i in model.trainable_weights:
    #    print(type(i))
    #    print(i.shape)
        #print(i)
    #print(evaluate_on_dataset())
    #y_pred = predict_on_dataset()
    #y_true = get_y_true()
    #for i in range(len(y_true)):
    #    print(y_true[i,0],y_pred[i,0])
    #print('total',len(y_pred),'one sign',len([i for i in range(len(y_pred)) if y_true[i]!=0.0 and y_true[i]*y_pred[i]>0]))
    #print(numpy.unique(y_true))
    #print(numpy.unique(y_pred))
    #root = None
    #b = get_board()
    #for i in range(100):
    #    print('iteration',i)
    #    print('position:')
    #    position_value, best_move, root, dataset_adding = search.get_move(b, printing=False, root=root, learn_mode=False)
    #    print("best move", best_move)
    #    b.push_uci(best_move)
    #for i in range(100):
    #    b,value = get_position_from_dataset(random.randint(1,20000))
    #    print(b)
    #    if b.turn:
    #        print('turn WHITE')
    #    else:
    #        print('turn BLACK')
    #    print('value',value)
    #    print()
    #b = get_board()
    #print(b)
    #print(list(b.legal_moves))
    #from search import pre_evaluation
    #from Node import Node
    #pre_evaluation(Node(),b)
    #print('is valid',b.is_valid())

