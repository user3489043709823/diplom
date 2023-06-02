# в этом файле будет подготовка набора данных к обучению. Из данных, полученных при поиске, будет готовиться датасет
import chess as c

# если будет добавляться много терминальных узлов, то датасет обучится под них и слабо обучится под все остальное
# обучится распознавать маты, паты, или окончания. Они все дают много полезной информации для сети, но нейросеть
# не должна подстроиться под них. Поэтому они должны быть, но их должно быть небольшое количество в общем количестве

terminal_samples_in_dataset = 0
non_terminal_samples_in_dataset = 0


def data_preparation(dataset_adding, learn_options):
    global terminal_samples_in_dataset, non_terminal_samples_in_dataset
    if learn_options.generate_endgame_position and learn_options.true_endgame_piece_number:
        dataset_adding = [i for i in dataset_adding if len(i[0].piece_map()) == learn_options.pieces_number]
    if not learn_options.use_game_over_positions:
        dataset_adding = [i for i in dataset_adding if not i[0].is_game_over(claim_draw=False)]
    if learn_options.limit_terminals_number_in_dataset:
        filter_mask = []
        for index in range(len(dataset_adding)):
            if dataset_adding[index][2]:  # эта запись - терминальный узел?
                if non_terminal_samples_in_dataset > 16 * terminal_samples_in_dataset:
                    add_to_dataset = True
                    terminal_samples_in_dataset += 1
                else:
                    add_to_dataset = False
            else:
                add_to_dataset = True
                non_terminal_samples_in_dataset += 1
            filter_mask.append(add_to_dataset)
        dataset_adding = [dataset_adding[i] for i in range(len(dataset_adding)) if filter_mask[i]]
    else:
        for index in range(len(dataset_adding)):
            if dataset_adding[index][2]:  # эта запись - терминальный узел?
                terminal_samples_in_dataset += 1
            else:
                non_terminal_samples_in_dataset += 1
    # кроме каждой позиции можно возвращать ее копию, отраженную по горизонтали,
    # а так же отраженную по вертикали с переменой цветов
    # при каждом отражении сохраняется правило, что белые пешки ходят вверх, а черные вниз
    adding_len = len(dataset_adding)
    if learn_options.mirror_vertically:
        dataset_adding.extend([(i[0].mirror(), -i[1], i[2]) for i in dataset_adding])
    if learn_options.flip_horizontally:
        dataset_adding.extend([(i[0].transform(c.flip_horizontal), i[1], i[2]) for i in dataset_adding])
    dataset_adding2 = []
    if learn_options.mirror_vertically and learn_options.flip_horizontally:
        for i in range(adding_len):  # в dataset_adding2 сумма значений 4 подряд позиций - 0
            # это может помочь в том, чтобы в каждом наборе была сумма значений 0, и не будет возникать смещение
            dataset_adding2.append(dataset_adding[i])
            dataset_adding2.append(dataset_adding[i + adding_len])
            dataset_adding2.append(dataset_adding[i + 2 * adding_len])
            dataset_adding2.append(dataset_adding[i + 3 * adding_len])
        dataset_adding = dataset_adding2
    # print('non-terminal samples', len([i for i in dataset_adding if not i[2]]))
    # print('terminal samples', len([i for i in dataset_adding if i[2]]))
    # print('terminal game over samples', len([i for i in dataset_adding if i[0].is_game_over(claim_draw=False)]))
    # print('terminal samples in dataset', terminal_samples_in_dataset)
    # print('non terminal samples in dataset', non_terminal_samples_in_dataset)
    return dataset_adding
