
import keras.utils
import numpy as np
from keras import Model, Input
from keras.layers import Dense, AveragePooling2D, Conv1D, Conv2D, Conv3D, Concatenate, DepthwiseConv2D, GlobalAveragePooling2D
from keras.layers import LocallyConnected2D, Flatten, BatchNormalization,Activation,Reshape,ZeroPadding1D, Add
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

conv_features_coef = 4
group_features_coef = 8 * conv_features_coef
inter_group_coef = 2
lines_coef = 64 # число нравится
oblast_coef = inter_group_coef * group_features_coef

kernel_coef = 8000.0
bias_coef = -500.0
average_to_max_weights = [np.array([[[kernel_coef]]]), np.array([bias_coef])]
average_to_max_oblast_weights = \
    [np.array([[[[kernel_coef] for i in range(oblast_coef)]]]), np.array([bias_coef for i in range(oblast_coef)])]
average_to_max_diagonal_weights = [np.array([[[[kernel_coef] for i in range(15)]]]),
                                   np.array([bias_coef for i in range(15)])]
dense_weights = [np.array([[kernel_coef]]), np.array([bias_coef])]
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

def get_line8layers_color_piece_names(name,pawn=False):
    if pawn:
        return [(i[0][0],i[0][1],i[0][2]) for i in line8layers.items() if i[0][0].startswith(name) and i[0][2] == 'PAWN']
    else:
        return [(i[0][1],i[0][2]) for i in line8layers.items() if i[0][0].startswith(name)]

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


tmp = Conv2D(filters=2 * lines_coef, kernel_size=(1, 1), name='vertical_lines_layer',
             kernel_constraint=get_constraint(axis=-2),bias_constraint=get_constraint(axis=[]))
batch_norm_tmp = BatchNormalization(name='batch_norm'+tmp.name)
vertical_lines_layer = Activation(activation)(batch_norm_tmp(tmp(vertical_lines_layers)))
print('vertical_lines_layer', vertical_lines_layer.shape,'weights shape',get_all_shapes(tmp))
tmp = Conv2D(filters=2 * lines_coef, kernel_size=(1, 1), name='horizontal_lines_layer',
             kernel_constraint=get_constraint(axis=-2),bias_constraint=get_constraint(axis=[]))
batch_norm_tmp = BatchNormalization(name='batch_norm'+tmp.name)
horizontal_lines_layer = Activation(activation)(batch_norm_tmp(tmp(horizontal_lines_layers)))
print('horizontal_lines_layer', horizontal_lines_layer.shape,'weights shape',get_all_shapes(tmp))
# для этих слоев веса нейрона расположены вдоль предпоследнего измерения
tmp = DepthwiseConv2D(depth_multiplier=2 * lines_coef, kernel_size=(1, 8),name='diagonal1_lines_layer',
                      kernel_constraint=get_constraint(axis=[0,1]),bias_constraint=get_constraint(axis=[]))
batch_norm_tmp = BatchNormalization(name='batch_norm'+tmp.name)
diagonal1_lines_layer = Activation(activation)(batch_norm_tmp(tmp(diagonal1_lines_layers)))  # depth_multiplier - количество фильтров
print('diagonal1_lines_layer', diagonal1_lines_layer.shape,'weights shape',get_all_shapes(tmp))
tmp = DepthwiseConv2D(depth_multiplier=2 * lines_coef, kernel_size=(1, 8),name='diagonal2_lines_layer',
                      kernel_constraint=get_constraint(axis=[0,1]),bias_constraint=get_constraint(axis=[]))
batch_norm_tmp = BatchNormalization(name='batch_norm'+tmp.name)
diagonal2_lines_layer = Activation(activation)(batch_norm_tmp(tmp(diagonal2_lines_layers)))
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
tmp = Conv2D(filters=2 * lines_coef, kernel_size=(1, 3),name='pawn_vertical_lines_layer',
             kernel_constraint=get_constraint(axis=[1,2]),bias_constraint=get_constraint(axis=[]))
batch_norm_tmp = BatchNormalization(name='batch_norm'+tmp.name)
pawn_vertical_lines_layer = Activation(activation)(batch_norm_tmp(tmp(pawn_vertical_lines_layers)))
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
        tmp = Conv2D(filters=oblast_coef, kernel_size=(8, 8),name='OBLAST_MAX' + color + piece,
                     kernel_constraint=get_constraint(axis=[0,1]),bias_constraint=get_constraint(axis=[]))
        batch_norm_tmp = BatchNormalization(name='batch_norm'+'OBLAST_MAX' + color + piece)
        oblast_layers[('MAX', color, piece)] = Activation(activation)(batch_norm_tmp(tmp(bitboards_input[color + ' ' + piece])))
        print('oblast_layers max', oblast_layers[('MAX', color, piece)].shape,
              'weights shape',get_all_shapes(tmp))
        print('batch normalization oblast_layers max','weights shape',get_all_shapes(batch_norm_tmp))

# делаем свертки
bitboards_for_conv_layers = Concatenate(name='bitboards_for_conv_layers') \
    ([i[1] for i in bitboards_input.items() if not i[0].endswith('PIECES')])
conv_layers = dict()
print('bitboards_layers', bitboards_for_conv_layers.shape)
for i in range(1, 5):
    for j in range(1, 5):
        tmp = Conv2D(filters=2 * conv_features_coef, kernel_size=(i, j),
                     name='conv_layer_' + str(i) + '_' + str(j),
                     kernel_constraint=get_constraint(axis=[0,1,2]),bias_constraint=get_constraint(axis=[]))
        batch_norm_tmp = BatchNormalization(name='batch_norm'+tmp.name)
        conv_layers[(i, j)] = Activation(activation)(batch_norm_tmp(tmp(bitboards_for_conv_layers)))
        print('conv_layers', i, j, conv_layers[(i, j)].shape,'weights shape',get_all_shapes(tmp))
        # веса вдоль первых 3 измерений
# делаем локально связанные слои
bitboards_pieces_layers = Concatenate(name='bitboards_pieces_layers')([i[1] for i in bitboards_input.items()
                                                                       if not i[0].startswith('EMPTY') and not i[
        0].endswith('PIECES')])
for i in bitboards_input.items():
    print(i[0])
print('bitboards_pieces_layers', bitboards_pieces_layers.shape)
tmp = LocallyConnected2D(filters=2, kernel_size=(1, 1),name='locally_connected_layer',
                         kernel_constraint=get_constraint(axis=1),bias_constraint=get_constraint(axis=[]))
batch_norm_tmp = BatchNormalization()
locally_connected_layer = Activation(activation)(batch_norm_tmp(tmp(bitboards_pieces_layers)))
# тут могут быть для разных фаз игры
print('locally_connected_layer', locally_connected_layer.shape,'weights shape',get_all_shapes(tmp))
# веса вдоль второго измерения
# добавляем нейроны, которые соединены со всеми входами
bitboards_layers = Concatenate(name='bitboards_layers')([i[1] for i in bitboards_input.items()])
print('bitboards_layers', bitboards_layers.shape)
tmp = Conv2D(filters=2 * group_features_coef, kernel_size=(8, 8),name='single_feature_layer',
             kernel_constraint=get_constraint(axis=[0,1,2]),bias_constraint=get_constraint(axis=[]))
batch_norm_tmp = BatchNormalization(name='batch_norm'+tmp.name)
single_feature_layer = Activation(activation)(batch_norm_tmp(tmp(bitboards_layers)))
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
reshape_layer1 = Reshape(target_shape=(features_layer.shape[1],1))(features_layer) # чтобы можно было применить Zero Padding
print('reshape_layer1', reshape_layer1.shape)
filter_mul1 = 3
filter_mul2 = 5
filter_mul3 = 16
conv_3d_filter_number = filter_mul1*filter_mul2*filter_mul3
dim_size = 4 # для 8 и больше слишком много весов в модели
conv_3d_height = dim_size
conv_3d_wight = dim_size
conv_3d_depth = dim_size
kernel_size = conv_3d_height*conv_3d_wight*conv_3d_depth
next_layers_size = conv_3d_filter_number*kernel_size
print('next_layers_size',next_layers_size)
zero_padding_layer = ZeroPadding1D(padding=(0,next_layers_size-features_layer.shape[1]))(reshape_layer1)
# форма (None,next_layers_size,1)
print('zero_padding_layer', zero_padding_layer.shape)
reshape_layer2 = Reshape(target_shape=(conv_3d_height,conv_3d_wight,conv_3d_depth,conv_3d_filter_number))(zero_padding_layer)
print('reshape_layer2', reshape_layer2.shape)
prev_layer = reshape_layer2
res_layers_number = 8 # при таком числе количество весов примерно такое же, как в исходной сети
for i in range(res_layers_number):
    if i%2==0:
        conv_3d_group_number = conv_3d_filter_number//filter_mul1
    else:
        conv_3d_group_number = conv_3d_filter_number//filter_mul2
    tmp = Conv3D(filters=next_layers_size,kernel_size=conv_3d_height,padding='valid',groups=conv_3d_group_number)
    batch_norm_tmp = BatchNormalization()
    conv3d_layer = Activation(activation)(batch_norm_tmp(tmp(prev_layer)))
    print('conv3d_layer',conv3d_layer.shape)
    reshape_layer_tmp = Reshape(target_shape=(conv_3d_height,conv_3d_wight,conv_3d_depth,conv_3d_filter_number))(conv3d_layer)
    add_layer = Add()([prev_layer,reshape_layer_tmp])
    print('add_layer',add_layer.shape)
    prev_layer = add_layer
prev_layer = Flatten()(prev_layer)
# dense_layers_size = 1024 #базовая версия
# #dense_layers_size = 128 работает
# tmp = Dense(dense_layers_size, name='complex_features_1',
#             kernel_constraint=get_constraint(axis=0),bias_constraint=get_constraint(axis=[]))
# batch_norm_tmp = BatchNormalization()
# complex_features_1 = Activation(activation)(batch_norm_tmp(tmp(features_layer)))
# print('complex_features_1',complex_features_1.shape,'weights shape',get_all_shapes(tmp))
# tmp = Dense(dense_layers_size, name='complex_features_2',
#             kernel_constraint=get_constraint(axis=0),bias_constraint=get_constraint(axis=[]))
# batch_norm_tmp = BatchNormalization()
# complex_features_2 = Activation(activation)(batch_norm_tmp(tmp(complex_features_1)))
# print('complex_features_2',complex_features_2.shape,'weights shape',get_all_shapes(tmp))
# tmp = Dense(dense_layers_size, name='evaluation_intermediate',
#             kernel_constraint=get_constraint(axis=0),bias_constraint=get_constraint(axis=[]))
# batch_norm_tmp = BatchNormalization()
# evaluation_intermediate = Activation(activation)(batch_norm_tmp(tmp(complex_features_2)))
# print('evaluation_intermediate',evaluation_intermediate.shape,'weights shape',get_all_shapes(tmp))
tmp = Dense(1, name='output',
            kernel_constraint=get_constraint(axis=0),bias_constraint=get_constraint(axis=[]))
batch_norm_tmp = BatchNormalization()
output = Activation('softsign')(batch_norm_tmp(tmp(prev_layer))) # работающий вариант
#output = Activation('tanh')(batch_norm_tmp(tmp(evaluation_intermediate))) # из-за малого разброса softsign меняем функцию
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

if __name__ == '__main__':
    pass
    #print('model size',get_model_size(model))
    #print('recommended learning rate',get_recommended_learning_rate(model=model,time_for_learning=3600,time_for_step=5))
    #model.summary(show_trainable=True)
    keras.utils.plot_model(model, show_shapes=True, show_layer_activations=True, dpi=96 * 16)
    keras.utils.plot_model(model, show_shapes=False, show_layer_activations=True, dpi=96 * 16 * 16,
                           to_file='model_short.png')
