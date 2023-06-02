from model_chooser import model  # работает только с моей моделью
from constants import visualization_model_weights_path
from keras.activations import sigmoid
import numpy as np
from models_code.model_batch_norm_split6 import get_line8layers_color_piece_names

print('loading model weights...')
model.load_weights(visualization_model_weights_path)
print('model weights loaded')
colors_names = ['_WHITE_', '_BLACK_']
pieces_names = ['PAWN', 'KNIGHT', 'BISHOP', 'ROOK', 'QUEEN', 'KING']
# визуализация областей
print('OBLAST VISUALIZATION')
for color in colors_names:
    for piece in pieces_names:
        print(color, piece)
        oblast_layer = model.get_layer(name='OBLAST_MAX' + color + piece)
        batch_norm_oblast_layer = model.get_layer(name='batch_norm' + 'OBLAST_MAX' + color + piece)
        weights = oblast_layer.get_weights()
        gamma = batch_norm_oblast_layer.gamma.numpy()
        beta = batch_norm_oblast_layer.beta.numpy()
        for layer_filter in range(weights[0].shape[-1]):
            representation = weights[0][:, :, :, layer_filter]
            tmp = representation.shape
            representation.shape = (8, 8)
            answers = sigmoid(representation * gamma[layer_filter] + beta[layer_filter]).numpy()
            answers = np.flip(answers, axis=0)  # для правильного отображения доски
            summ = np.sum(answers)  # для выхода
            std = np.std(answers)
            if std > 0.0021:
                # if abs(summ/np.size(answers)-0.5)>0.001:
                print('sum', summ, 'std', std, 'gamma', gamma[layer_filter], 'beta', beta[layer_filter])
                for i in range(8):
                    for j in range(8):
                        value = answers[i, j]
                        print(round(number=value, ndigits=3), end=' ')
                    print()
                print('bias', round(number=weights[1][layer_filter], ndigits=20),
                      'bias output',
                      round(number=gamma[layer_filter] * weights[1][layer_filter] + beta[layer_filter], ndigits=20))
                print()
            representation.shape = tmp
print('CONVOLUTION VISUALIZATION')
conv_layers = [layer for layer in model.layers if 'conv_layer_' in layer.name and not ('batch_norm' in layer.name)]
for layer in conv_layers:
    print(layer.name)
    weights = layer.get_weights()
    batch_norm_conv_layer = model.get_layer(name='batch_norm' + layer.name)
    gamma = batch_norm_conv_layer.gamma.numpy()
    beta = batch_norm_conv_layer.beta.numpy()
    for layer_filter in range(weights[0].shape[-1]):
        representation = weights[0][:, :, :, layer_filter]
        tmp = representation.shape
        answers = sigmoid(representation * gamma[layer_filter] + beta[layer_filter]).numpy()
        answers = np.flip(answers, axis=0)  # для правильного отображения доски
        summ = np.sum(answers)  # для выхода
        std = np.std(answers)
        if std > 0.05:
            # if abs(summ/np.size(answers)-0.5)>0.01:
            print('sum', summ, 'std', std, 'gamma', gamma[layer_filter], 'beta', beta[layer_filter])
            print(np.round(answers, decimals=3))
            print('bias', round(number=weights[1][layer_filter], ndigits=20),
                  'bias output',
                  round(number=gamma[layer_filter] * weights[1][layer_filter] + beta[layer_filter], ndigits=20))
        representation.shape = tmp
print('SINGLE FEATURE LAYER VISUALIZATION')
single_feature_layer = model.get_layer(name='single_feature_layer')
weights = single_feature_layer.get_weights()
batch_norm_single_feature_layer = model.get_layer(name='batch_norm' + single_feature_layer.name)
gamma = batch_norm_single_feature_layer.gamma.numpy()
beta = batch_norm_single_feature_layer.beta.numpy()
for layer_filter in range(weights[0].shape[-1]):
    representation = weights[0][:, :, :, layer_filter]
    tmp = representation.shape
    answers = sigmoid(representation * gamma[layer_filter] + beta[layer_filter]).numpy()
    answers = np.flip(answers, axis=0)  # для правильного отображения доски
    summ = np.sum(answers)  # для выхода
    std = np.std(answers)
    if std > 0.0023:
        # if abs(summ/np.size(answers)-0.5)>0.01:
        print('sum', summ, 'std', std, 'gamma', gamma[layer_filter], 'beta', beta[layer_filter])
        print(np.round(answers, decimals=3))
        print('bias', round(number=weights[1][layer_filter], ndigits=20),
              'bias output',
              round(number=gamma[layer_filter] * weights[1][layer_filter] + beta[layer_filter], ndigits=20))
    representation.shape = tmp
print('HORIZONTAL LINES LAYER VISUALIZATION')
horizontal_lines_layer = model.get_layer(name='horizontal_lines_layer')
weights = horizontal_lines_layer.get_weights()
batch_norm_horizontal_lines_layer = model.get_layer(name='batch_norm' + horizontal_lines_layer.name)
gamma = batch_norm_horizontal_lines_layer.gamma.numpy()
beta = batch_norm_horizontal_lines_layer.beta.numpy()

pieces_color_names = get_line8layers_color_piece_names('HORIZONTAL')
for layer_filter in range(weights[0].shape[-1]):
    representation = weights[0][:, :, :, layer_filter]
    tmp = representation.shape
    representation.shape = (representation.shape[2],)
    answers = sigmoid(representation * gamma[layer_filter] + beta[layer_filter]).numpy()
    summ = np.sum(answers)  # для выхода
    std = np.std(answers)
    if std > 0.045:
        print('sum', summ, 'std', std, 'gamma', gamma[layer_filter], 'beta', beta[layer_filter])
        for i in range(len(answers)):
            print(pieces_color_names[i], np.round(answers[i], decimals=3))
        print('bias', round(number=weights[1][layer_filter], ndigits=20),
              'bias output',
              round(number=gamma[layer_filter] * weights[1][layer_filter] + beta[layer_filter], ndigits=20))
    representation.shape = tmp
print('VERTICAL LINES LAYER VISUALIZATION')
vertical_lines_layer = model.get_layer(name='vertical_lines_layer')
weights = vertical_lines_layer.get_weights()
batch_norm_vertical_lines_layer = model.get_layer(name='batch_norm' + vertical_lines_layer.name)
gamma = batch_norm_vertical_lines_layer.gamma.numpy()
beta = batch_norm_vertical_lines_layer.beta.numpy()
pieces_color_names = get_line8layers_color_piece_names('VERTICAL')
for layer_filter in range(weights[0].shape[-1]):
    representation = weights[0][:, :, :, layer_filter]
    tmp = representation.shape
    representation.shape = (representation.shape[2],)
    answers = sigmoid(representation * gamma[layer_filter] + beta[layer_filter]).numpy()
    summ = np.sum(answers)  # для выхода
    std = np.std(answers)
    if std > 0.045:
        print('sum', summ, 'std', std, 'gamma', gamma[layer_filter], 'beta', beta[layer_filter])
        for i in range(len(answers)):
            print(pieces_color_names[i], np.round(answers[i], decimals=3))
        print('bias', round(number=weights[1][layer_filter], ndigits=20),
              'bias output',
              round(number=gamma[layer_filter] * weights[1][layer_filter] + beta[layer_filter], ndigits=20))
    representation.shape = tmp
# print('DIAGONAL1 LINES LAYER VISUALIZATION')
# diagonal1_lines_layer = model.get_layer(name='diagonal1_lines_layer')
# weights = diagonal1_lines_layer.get_weights()
# batch_norm_diagonal1_lines_layer = model.get_layer(name='batch_norm' + diagonal1_lines_layer.name)
# gamma = batch_norm_diagonal1_lines_layer.gamma.numpy()
# beta = batch_norm_diagonal1_lines_layer.beta.numpy()
# pieces_color_names = get_line8layers_color_piece_names('DIAGONAL1')
# for layer_filter in range(weights[0].shape[-1]):
#     for diagonal in range(weights[0].shape[-2]):
#         representation = weights[0][:, :, diagonal, layer_filter]
#         tmp = representation.shape
#         representation.shape = (representation.shape[1],)
#         answers = sigmoid(representation * gamma[layer_filter] + beta[layer_filter]).numpy()
#         summ = np.sum(answers)  # для выхода
#         std = np.std(answers)
#         if std > 0.018:
#             print('sum', summ, 'std', std, 'gamma', gamma[layer_filter], 'beta', beta[layer_filter])
#             for i in range(len(answers)):
#                 print(pieces_color_names[i], np.round(answers[i], decimals=3))
#             print('bias', round(number=weights[1][layer_filter], ndigits=20),
#                   'bias output',
#                   round(number=gamma[layer_filter] * weights[1][layer_filter] + beta[layer_filter], ndigits=20))
#         representation.shape = tmp
#
# print('DIAGONAL2 LINES LAYER VISUALIZATION')
# diagonal2_lines_layer = model.get_layer(name='diagonal2_lines_layer')
# weights = diagonal2_lines_layer.get_weights()
# batch_norm_diagonal2_lines_layer = model.get_layer(name='batch_norm' + diagonal2_lines_layer.name)
# gamma = batch_norm_diagonal2_lines_layer.gamma.numpy()
# beta = batch_norm_diagonal2_lines_layer.beta.numpy()
# pieces_color_names = get_line8layers_color_piece_names('DIAGONAL2')
# for layer_filter in range(weights[0].shape[-1]):
#     for diagonal in range(weights[0].shape[-2]):
#         representation = weights[0][:, :, diagonal, layer_filter]
#         tmp = representation.shape
#         representation.shape = (representation.shape[1],)
#         answers = sigmoid(representation * gamma[layer_filter] + beta[layer_filter]).numpy()
#         summ = np.sum(answers)  # для выхода
#         std = np.std(answers)
#         if std > 0.018:
#             print('sum', summ, 'std', std, 'gamma', gamma[layer_filter], 'beta', beta[layer_filter])
#             for i in range(len(answers)):
#                 print(pieces_color_names[i], np.round(answers[i], decimals=3))
#             print('bias', round(number=weights[1][layer_filter], ndigits=20),
#                   'bias output',
#                   round(number=gamma[layer_filter] * weights[1][layer_filter] + beta[layer_filter], ndigits=20))
#         representation.shape = tmp
print('PAWN VERTICAL LINES LAYER VISUALIZATION')
pawn_vertical_lines_layer = model.get_layer(name='pawn_vertical_lines_layer')
weights = pawn_vertical_lines_layer.get_weights()
batch_norm_pawn_vertical_lines_layer = model.get_layer(name='batch_norm' + pawn_vertical_lines_layer.name)
gamma = batch_norm_pawn_vertical_lines_layer.gamma.numpy()
beta = batch_norm_pawn_vertical_lines_layer.beta.numpy()
pieces_color_names = get_line8layers_color_piece_names('VERTICAL', pawn=True)
for layer_filter in range(weights[0].shape[-1]):
    representation = weights[0][:, :, :, layer_filter]
    tmp = representation.shape
    representation.shape = (representation.shape[1], representation.shape[2])
    answers = sigmoid(representation * gamma[layer_filter] + beta[layer_filter]).numpy()
    summ = np.sum(answers)  # для выхода
    std = np.std(answers)
    if std > 0.03:
        print('sum', summ, 'std', std, 'gamma', gamma[layer_filter], 'beta', beta[layer_filter])
        for i in range(answers.shape[0]):
            for j in range(answers.shape[1]):
                if i == 0:
                    print('LEFT', end=' ')
                elif i == 1:
                    print('CENTER', end=' ')
                elif i == 2:
                    print('RIGHT', end=' ')
                print(pieces_color_names[j], np.round(answers[i, j], decimals=3))
        print('bias', round(number=weights[1][layer_filter], ndigits=20),
              'bias output',
              round(number=gamma[layer_filter] * weights[1][layer_filter] + beta[layer_filter], ndigits=20))
    representation.shape = tmp
# при батч нормализации стандартное отклонение выхода ожидаемое равно примерно 0.25

print('big_dense_layer LAYER VISUALIZATION')
big_dense_layer = model.get_layer(name='big_dense_layer')
weights = big_dense_layer.get_weights()
print('weights')
print('weights size', weights[0].size)
print('non zero count', np.count_nonzero(weights[0]))
print('zero count', weights[0].size - np.count_nonzero(weights[0]))
# batch_norm_complex_features_1 = model.get_layer(name='batch_norm'+complex_features_1.name)
# gamma = batch_norm_complex_features_1.gamma.numpy()
# beta = batch_norm_complex_features_1.beta.numpy()
# if False:
#     for layer_filter in range(weights[0].shape[-1]):
#         representation = weights[0][:, :, :, layer_filter]
#         tmp = representation.shape
#         answers = sigmoid(representation * gamma[layer_filter] + beta[layer_filter]).numpy()
#         answers = np.flip(answers, axis=0)  # для правильного отображения доски
#         summ = np.sum(answers)  # для выхода
#         std = np.std(answers)
#         if std > 0.0023:
#             # if abs(summ/np.size(answers)-0.5)>0.01:
#             print('sum', summ, 'std', std, 'gamma', gamma[layer_filter], 'beta', beta[layer_filter])
#             print(np.round(answers, decimals=3))
#             print('bias', round(number=weights[1][layer_filter], ndigits=20),
#                   'bias output',
#                   round(number=gamma[layer_filter] * weights[1][layer_filter] + beta[layer_filter], ndigits=20))
#         representation.shape = tmp
