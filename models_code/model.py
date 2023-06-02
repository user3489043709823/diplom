# тут нету пакетной нормализации

import keras.utils
import numpy as np
from keras import Model, Input
from keras.layers import Dense, AveragePooling2D, Conv1D, Conv2D, Concatenate, DepthwiseConv2D, GlobalAveragePooling2D
from keras.layers import LocallyConnected2D, Flatten, Layer
from model_utilites import get_all_shapes, relu_with_max


#activation = relu_with_max
#activation = 'softsign'
activation = 'sigmoid'
bitboards_num = 2 * 6 + 2 + 1
bitboards_input_names = []
colors_names = ['_WHITE_', '_BLACK_']
pieces_names = ['PAWN', 'KNIGHT', 'BISHOP', 'ROOK', 'QUEEN', 'KING']
for color in colors_names:
    for piece in pieces_names:
        bitboards_input_names.append(color + ' ' + piece)
for color in colors_names:
    bitboards_input_names.append(color + ' ' + 'PIECES')
bitboards_input_names.append('EMPTY SQUARES')
print(bitboards_input_names)
# создаем входы
bitboards_input = {bitboards_input_names[i]: Input(shape=(8, 8, 1), name=bitboards_input_names[i]) for i in
                   range(bitboards_num)}
# такая форма, чтобы была ось каналов, которая нужна для использования этого слоя в average pooling 2d
en_passant_3_line_input = Input(shape=(8,), name='en_passant_3_line')
en_passant_6_line_input = Input(shape=(8,), name='en_passant_6_line')
castling_rights_input = Input(shape=(4,), name='castling_rights')
turn_input = Input(shape=(1,), name='turn')
is_check_input = Input(shape=(1,), name='is_check')
input_list = list(bitboards_input.values()) + [en_passant_3_line_input, en_passant_6_line_input, castling_rights_input,
                                               turn_input, is_check_input]
# создаем слои - линии длины 8
line8layers = dict()

conv_features_coef = 32
group_features_coef = 8 * conv_features_coef
inter_group_coef = 2
lines_coef = 32 # число нравится
oblast_coef = inter_group_coef * group_features_coef

average_to_max_weights = [np.array([[[8000.0]]]), np.array([-500.0])]
average_to_max_oblast_weights = \
    [np.array([[[[8000.0] for i in range(oblast_coef)]]]), np.array([-500.0 for i in range(oblast_coef)])]
average_to_max_diagonal_weights = [np.array([[[[8000.0] for i in range(15)]]]),
                                   np.array([-500.0 for i in range(15)])]
dense_weights = [np.array([[8000.0]]), np.array([-500.0])]
diagonal_weights1_kernel = np.zeros(shape=(8, 8, 1, 15))
diagonal_weights2_kernel = np.zeros(shape=(8, 8, 1, 15))
diagonal_weights_bias = np.zeros(15)
for i in range(8):
    for j in range(8):
        for x in range(15):
            if (i + j == x):
                diagonal_weights1_kernel[i, j, 0, x] = 1.0
            if (abs(i - j) == x):
                diagonal_weights2_kernel[i, j, 0, x] = 1.0

for color in colors_names:
    for piece in ['ROOK', 'QUEEN', 'PAWN']:
        line8layers[('VERTICAL_AVERAGE', color, piece)] = AveragePooling2D(pool_size=(8, 1),
                                                                           name='VERTICAL_AVERAGE' + color + piece) \
            (bitboards_input[color + ' ' + piece])
        tmp = Conv1D(filters=1, kernel_size=1, activation=activation, trainable=False,
                     name='VERTICAL_MAX' + color + piece)
        line8layers[('VERTICAL_MAX', color, piece)] = tmp(line8layers[('VERTICAL_AVERAGE', color, piece)])
        print('VERTICAL_AVERAGE', line8layers[('VERTICAL_AVERAGE', color, piece)].shape)
        print('VERTICAL_MAX', line8layers[('VERTICAL_MAX', color, piece)].shape)
        # print('VERTICAL_MAX weights')
        # print(tmp.get_weights())
        tmp.set_weights(average_to_max_weights)
        # print('VERTICAL_MAX weights 2')
        # print(tmp.get_weights())

        if piece != 'PAWN':
            line8layers[('HORIZONTAL_AVERAGE', color, piece)] \
                = AveragePooling2D(pool_size=(1, 8), name='HORIZONTAL_AVERAGE' + color + piece)(
                bitboards_input[color + ' ' + piece])
            tmp = Conv1D(filters=1, kernel_size=1, activation=activation, trainable=False,
                         name='HORIZONTAL_MAX' + color + piece)
            line8layers[('HORIZONTAL_MAX', color, piece)] = tmp(line8layers[('HORIZONTAL_AVERAGE', color, piece)])
            tmp.set_weights(average_to_max_weights)
        # установить веса и заморозить или заменить на другой слой
    for piece in ['BISHOP', 'QUEEN']:
        tmp1 = Conv2D(filters=15, kernel_size=(8, 8), trainable=False, name='DIAGONAL1_AVERAGE' + color + piece)
        line8layers[('DIAGONAL1_AVERAGE', color, piece)] = tmp1(bitboards_input[color + ' ' + piece])
        print('diagonal weights')
        print(tmp1.get_weights()[0].shape)
        # print('bitboards',bitboards_input[color+' '+piece].shape)
        # tmp_np = np.zeros(shape=(8,8,1))
        # tmp_np = np.zeros(shape=(8,8,1))
        # tmp_np.shape = (1,8,8)
        # tmp_np.shape = (8,1,8)
        # tmp_np.shape = (8,8,1)
        # tmp_model = Model(inputs = bitboards_input[color+' '+piece], outputs = bitboards_input[color+' '+piece])
        # print('proba')
        # print(tmp_model(tmp_np).shape)
        # tmp_model = Model(inputs = bitboards_input[color+' '+piece], outputs = line8layers[('DIAGONAL1_AVERAGE',color,piece)])
        # print('proba 2')
        # print(tmp_model(tmp_np).shape)
        # print('proba 2')
        # print(bitboards_input[color+' '+piece](tmp_np).shape)
        tmp1.set_weights([diagonal_weights1_kernel, diagonal_weights_bias])
        print('DIAGONAL1_AVERAGE', line8layers[('DIAGONAL1_AVERAGE', color, piece)].shape)
        tmp2 = Conv2D(filters=15, kernel_size=(8, 8), trainable=False, name='DIAGONAL2_AVERAGE' + color + piece)
        line8layers[('DIAGONAL2_AVERAGE', color, piece)] = tmp2(bitboards_input[color + ' ' + piece])
        tmp2.set_weights([diagonal_weights2_kernel, diagonal_weights_bias])
        tmp1 = DepthwiseConv2D(kernel_size=(1, 1), activation=activation, trainable=False,
                               name='DIAGONAL1_MAX' + color + piece)
        line8layers[('DIAGONAL1_MAX', color, piece)] = tmp1(line8layers[('DIAGONAL1_AVERAGE', color, piece)])
        tmp1.set_weights(average_to_max_diagonal_weights)
        print('DIAGONAL1_MAX', line8layers[('DIAGONAL1_MAX', color, piece)].shape)
        tmp2 = DepthwiseConv2D(kernel_size=(1, 1), activation=activation, trainable=False,
                               name='DIAGONAL2_MAX' + color + piece)
        line8layers[('DIAGONAL2_MAX', color, piece)] = tmp2(line8layers[('DIAGONAL2_AVERAGE', color, piece)])
        tmp2.set_weights(average_to_max_diagonal_weights)
        # установить веса и заморозить или заменить на другой слой

vertical_lines_layers = \
    Concatenate(name='vertical_lines_layers')([i[1] for i in line8layers.items() if i[0][0].startswith('VERTICAL')])
horizontal_lines_layers = \
    Concatenate(name='horizontal_lines_layers')([i[1] for i in line8layers.items() if i[0][0].startswith('HORIZONTAL')])
print('vertical_lines_layers', vertical_lines_layers.shape)
print('horizontal_lines_layers', horizontal_lines_layers.shape)
diagonal1_lines_layers = \
    Concatenate(axis=-2, name='diagonal1_lines_layers')(
        [i[1] for i in line8layers.items() if i[0][0].startswith('DIAGONAL1')])
diagonal2_lines_layers = \
    Concatenate(axis=-2, name='diagonal2_lines_layers')(
        [i[1] for i in line8layers.items() if i[0][0].startswith('DIAGONAL2')])
print('diagonal1_lines_layers', diagonal1_lines_layers.shape)
print('diagonal2_lines_layers', diagonal2_lines_layers.shape)
from custom_constraints import MaxNormL1
def get_constraint(axis):
    return None
    #return MaxNormL1(max_value=16.0,axis=axis)
tmp = Conv2D(filters=2 * lines_coef, kernel_size=(1, 1), activation=activation, name='vertical_lines_layer',
             kernel_constraint=get_constraint(axis=-2),bias_constraint=get_constraint(axis=[]))
vertical_lines_layer = tmp(vertical_lines_layers)
print('vertical_lines_layer', vertical_lines_layer.shape,'weights shape',get_all_shapes(tmp))
tmp = Conv2D(filters=2 * lines_coef, kernel_size=(1, 1), activation=activation, name='horizontal_lines_layer',
             kernel_constraint=get_constraint(axis=-2),bias_constraint=get_constraint(axis=[]))
horizontal_lines_layer = tmp(horizontal_lines_layers)
print('horizontal_lines_layer', horizontal_lines_layer.shape,'weights shape',get_all_shapes(tmp))
# для этих слоев веса нейрона расположены вдоль предпоследнего измерения
tmp = DepthwiseConv2D(depth_multiplier=2 * lines_coef, kernel_size=(1, 8), activation=activation,name='diagonal1_lines_layer',
                      kernel_constraint=get_constraint(axis=[0,1]),bias_constraint=get_constraint(axis=[]))
diagonal1_lines_layer = tmp(diagonal1_lines_layers)  # depth_multiplier - количество фильтров
print('diagonal1_lines_layer', diagonal1_lines_layer.shape,'weights shape',get_all_shapes(tmp))
tmp = DepthwiseConv2D(depth_multiplier=2 * lines_coef, kernel_size=(1, 8), activation=activation,name='diagonal2_lines_layer',
                      kernel_constraint=get_constraint(axis=[0,1]),bias_constraint=get_constraint(axis=[]))
diagonal2_lines_layer = tmp(diagonal2_lines_layers)
print('diagonal2_lines_layer', diagonal2_lines_layer.shape,'weights shape',get_all_shapes(tmp))
# веса нейрона вдоль первых 2 измерений
# proba_av = AveragePooling2D(pool_size=(8,1))
# proba_data = np.array([0.0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
#              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,])
# proba_data.reshape((8,8,1))
# proba_data.shape = (1,8,8,1)
# print(proba_data)
# print(proba_av(proba_data))
pawn_vertical_lines_layers = Concatenate(name='pawn_vertical_lines_layers') \
    ([i[1] for i in line8layers.items() if i[0][0].startswith('VERTICAL') and i[0][2] == 'PAWN'])
# не добавлено про пустые пространства использовать их вместе с пешками
print('pawn_vertical_lines_layers', pawn_vertical_lines_layers.shape)
tmp = Conv2D(filters=2 * lines_coef, kernel_size=(1, 3), activation=activation,name='pawn_vertical_lines_layer',
             kernel_constraint=get_constraint(axis=[1,2]),bias_constraint=get_constraint(axis=[]))
pawn_vertical_lines_layer = tmp(pawn_vertical_lines_layers)
print('pawn_vertical_lines_layer', pawn_vertical_lines_layer.shape,'weights shape',get_all_shapes(tmp))
# веса нейрона вдоль предпоследнего измерения и второго
# делаем pooling по всей доске
global_pooling_layers = dict()
for color in colors_names:
    for piece in pieces_names:
        if piece != 'KING':  # для короля нету смысла, результат всегда будет одинаков так как король всегда есть на доске
            global_pooling_layers[('AVERAGE', color, piece)] = GlobalAveragePooling2D(name='AVERAGE' + color + piece) \
                (bitboards_input[color + ' ' + piece])
            tmp = Dense(1, activation=activation, trainable=False, name='MAX' + color + piece)
            global_pooling_layers[('MAX', color, piece)] = tmp(global_pooling_layers[('AVERAGE', color, piece)])
            tmp.set_weights(dense_weights)
            print('Dense weights')
            print(tmp.get_weights())
            print('global_pooling_layers average', global_pooling_layers[('AVERAGE', color, piece)].shape)
            print('global_pooling_layers max', global_pooling_layers[('MAX', color, piece)].shape)
# делаем нейроны наличия фигуры в определенной области
oblast_layers = dict()
for color in colors_names:
    for piece in pieces_names:
        # без адаптации к пакетной нормализации
        #tmp = Conv2D(filters=oblast_coef, kernel_size=(8, 8),name='OBLAST_AVERAGE' + color + piece,
        #             kernel_constraint=get_constraint(axis=[0,1]),bias_constraint=get_constraint(axis=[]))
        #oblast_layers[('AVERAGE', color, piece)] = tmp(bitboards_input[color + ' ' + piece])
        #print('oblast_layers average', oblast_layers[('AVERAGE', color, piece)].shape,
        #      'weights shape',get_all_shapes(tmp))
        ## веса вдоль первых 2 измерений
        #tmp = DepthwiseConv2D(kernel_size=(1, 1), activation=activation, trainable=False,
        #                      name='OBLAST_MAX' + color + piece)
        #oblast_layers[('MAX', color, piece)] = tmp(oblast_layers[('AVERAGE', color, piece)])
        #tmp.set_weights(average_to_max_oblast_weights)
        #print('oblast_layers max', oblast_layers[('MAX', color, piece)].shape)
        # с адаптацией
        tmp = Conv2D(filters=oblast_coef, kernel_size=(8, 8),name='OBLAST_MAX' + color + piece,activation=activation,
                     kernel_constraint=get_constraint(axis=[0,1]),bias_constraint=get_constraint(axis=[]))
        oblast_layers[('MAX', color, piece)] = tmp(bitboards_input[color + ' ' + piece])
        print('oblast_layers max', oblast_layers[('MAX', color, piece)].shape,
              'weights shape',get_all_shapes(tmp))
# делаем свертки
bitboards_for_conv_layers = Concatenate(name='bitboards_for_conv_layers') \
    ([i[1] for i in bitboards_input.items() if not i[0].endswith('PIECES')])
conv_layers = dict()
print('bitboards_layers', bitboards_for_conv_layers.shape)
for i in range(1, 5):
    for j in range(1, 5):
        tmp = Conv2D(filters=2 * conv_features_coef, kernel_size=(i, j), activation=activation,
                     name='conv_layer_' + str(i) + '_' + str(j),
                     kernel_constraint=get_constraint(axis=[0,1,2]),bias_constraint=get_constraint(axis=[]))
        conv_layers[(i, j)] = tmp(bitboards_for_conv_layers)
        print('conv_layers', i, j, conv_layers[(i, j)].shape,'weights shape',get_all_shapes(tmp))
        # веса вдоль первых 3 измерений
# делаем локально связанные слои
bitboards_pieces_layers = Concatenate(name='bitboards_pieces_layers')([i[1] for i in bitboards_input.items()
                                                                       if not i[0].startswith('EMPTY') and not i[
        0].endswith('PIECES')])
for i in bitboards_input.items():
    print(i[0])
print('bitboards_pieces_layers', bitboards_pieces_layers.shape)
tmp = LocallyConnected2D(filters=2, kernel_size=(1, 1), activation=activation,name='locally_connected_layer',
                         kernel_constraint=get_constraint(axis=1),bias_constraint=get_constraint(axis=[]))
locally_connected_layer = tmp(bitboards_pieces_layers)
# тут могут быть для разных фаз игры
print('locally_connected_layer', locally_connected_layer.shape,'weights shape',get_all_shapes(tmp))
# веса вдоль второго измерения
# добавляем нейроны, которые соединены со всеми входами
bitboards_layers = Concatenate(name='bitboards_layers')([i[1] for i in bitboards_input.items()])
print('bitboards_layers', bitboards_layers.shape)
tmp = Conv2D(filters=2 * group_features_coef, kernel_size=(8, 8), activation=activation,name='single_feature_layer',
             kernel_constraint=get_constraint(axis=[0,1,2]),bias_constraint=get_constraint(axis=[]))
single_feature_layer = tmp(bitboards_layers)
print('single_feature_layer', single_feature_layer.shape,'weights shape',get_all_shapes(tmp))
# веса вдоль первых 3 измерений
# изменяем форму чтобы использовать в Dense слоях
for_flatten_tmp = []
for_flatten_tmp += input_list
print('for_flatten_tmp len', len(for_flatten_tmp))
for_flatten_tmp += [i[1] for i in line8layers.items() if 'MAX' in i[0][0]]
print('for_flatten_tmp len', len(for_flatten_tmp))
for_flatten_tmp.append(vertical_lines_layer)
for_flatten_tmp.append(horizontal_lines_layer)
for_flatten_tmp.append(diagonal1_lines_layer)
for_flatten_tmp.append(diagonal2_lines_layer)
for_flatten_tmp.append(pawn_vertical_lines_layer)
# print('for_flatten_tmp len',len(for_flatten_tmp))
for_flatten_tmp += [i[1] for i in global_pooling_layers.items() if 'MAX' in i[0][0]]
# print('for_flatten_tmp len',len(for_flatten_tmp))
for_flatten_tmp += [i[1] for i in oblast_layers.items() if 'MAX' in i[0][0]]
# print('for_flatten_tmp len',len(for_flatten_tmp))
for_flatten_tmp += [i[1] for i in conv_layers.items()]
# print('for_flatten_tmp len',len(for_flatten_tmp))
for_flatten_tmp.append(locally_connected_layer)
for_flatten_tmp.append(single_feature_layer)
Flatten_layers = []
for i in for_flatten_tmp:
    if len(i.shape) == 1:
        Flatten_layers.append(i)
    else:
        Flatten_layers.append(Flatten()(i))
features_layer = Concatenate(name='features_layer')(Flatten_layers)
print('features_layer', features_layer.shape)

#dense_layers_size = 1024 базовая версия
dense_layers_size = 1024

tmp = Dense(dense_layers_size, activation=activation, name='complex_features_1',
            kernel_constraint=get_constraint(axis=0),bias_constraint=get_constraint(axis=[]))
complex_features_1 = tmp(features_layer)
print('complex_features_1',complex_features_1.shape,'weights shape',get_all_shapes(tmp))
tmp = Dense(dense_layers_size, activation=activation, name='complex_features_2',
            kernel_constraint=get_constraint(axis=0),bias_constraint=get_constraint(axis=[]))
complex_features_2 = tmp(complex_features_1)
print('complex_features_2',complex_features_2.shape,'weights shape',get_all_shapes(tmp))
tmp = Dense(dense_layers_size, activation=activation, name='evaluation_intermediate',
            kernel_constraint=get_constraint(axis=0),bias_constraint=get_constraint(axis=[]))
evaluation_intermediate = tmp(complex_features_2)
print('evaluation_intermediate',evaluation_intermediate.shape,'weights shape',get_all_shapes(tmp))
#tmp = Dense(1, activation='softsign', name='output',kernel_constraint=get_constraint(axis=0),bias_constraint=get_constraint(axis=[]))
tmp = Dense(1, name='output',kernel_constraint=get_constraint(axis=0),bias_constraint=get_constraint(axis=[]))

from keras.layers import BatchNormalization,Activation
output = Activation('softsign')(BatchNormalization()(tmp(evaluation_intermediate)))
#output = tmp(evaluation_intermediate)
print('output',output.shape,'weights shape',get_all_shapes(tmp))
# веса вдоль первого измерения
model = Model(inputs=input_list, outputs=output)
# tmp = Dense(1,activation=activation,trainable=False)
# x = Input(shape=(1,))
# tmp2 = tmp(x)
# weights = [np.array([[[[ 8000.0 ],[ 8000.0],[8000.0],[8000.0],[8000.0],[8000.0],[8000.0],[8000.0]]]]),
#           np.array([-500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0])]
# tmp.set_weights(dense_weights)
# m = Model(inputs=x,outputs=tmp2)
# print('tmp.get_weights()',tmp.get_weights())
# a = np.array([0.0])
# print(a.shape)
# print(m.predict(a))
# little_model = Model(inputs=[bitboards_input['WHITE'+' '+'ROOK'],bitboards_input['BLACK'+' '+'ROOK']],
# outputs=[line8layers[('VERTICAL_AVERAGE','WHITE', 'ROOK')],
#        line8layers[('VERTICAL_AVERAGE','BLACK', 'ROOK')]])
# proba = np.zeros(64)
# proba.shape = (8,8,1)

# little_model([proba,proba])
# for i in input_list:
#    print(i)

#if __name__ == '__main__':
#    print('model size',get_model_size(model))
#    print('recommended learning rate',get_recommended_learning_rate(model=model,time_for_learning=3600,time_for_step=5))
    #model.summary(show_trainable=True)
    #keras.utils.plot_model(model, show_shapes=True, show_layer_activations=True, dpi=96 * 16)
    #keras.utils.plot_model(model, show_shapes=False, show_layer_activations=True, dpi=96 * 16 * 16,
    #                       to_file='model_short.png')
