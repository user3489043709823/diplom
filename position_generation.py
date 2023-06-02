from random import randint, seed as rseed
from Node import Node
from search import c, pre_evaluation

white_squares = c.SquareSet(c.BB_LIGHT_SQUARES)
black_squares = c.SquareSet(c.BB_DARK_SQUARES)
colors = [c.WHITE, c.BLACK]
all_not_king_pieces = []  # все фигуры кроме королей, которые есть на доске в начале игры


# print('random number',randint(1,10))
def some_init():
    for color in colors:
        for iindex in range(8):
            all_not_king_pieces.append(c.Piece(piece_type=c.PAWN, color=color))
        for piece in [c.KNIGHT, c.BISHOP, c.ROOK]:
            for iindex in range(2):
                all_not_king_pieces.append(c.Piece(piece_type=piece, color=color))
        all_not_king_pieces.append(c.Piece(piece_type=c.QUEEN, color=color))


some_init()


def get_random_free_square(b):
    while True:
        rand_square = c.SQUARES[randint(0, 63)]
        if b.piece_at(rand_square) is None:
            break
    return rand_square


def generate_random_position(pieces_number, seed=None):
    if seed:
        rseed(seed)
    b = c.Board()
    while True:
        b.clear()
        b.turn = bool(randint(0, 1))
        # возможностей рокировки после b.clear() нет
        # устанавливаем королей
        rand_square = c.SQUARES[randint(0, 63)]
        b.set_piece_at(piece=c.Piece(piece_type=c.KING, color=c.WHITE), square=rand_square)
        b.set_piece_at(piece=c.Piece(piece_type=c.KING, color=c.BLACK), square=get_random_free_square(b))
        # устанавливаем другие фигуры. Для простоты устанавливаем в предположении, что не было прохождение пешек
        all_not_king_pieces2 = all_not_king_pieces.copy()
        for i in range(pieces_number - 2):
            b.set_piece_at(piece=all_not_king_pieces2.pop(randint(0, len(all_not_king_pieces2) - 1)),
                           square=get_random_free_square(b))
        ok = True
        for colorr in colors:
            bishops = b.pieces(piece_type=c.BISHOP, color=colorr)
            if len(bishops) == 2:
                bishops_list = list(bishops)
                if (bishops_list[0] in white_squares and bishops_list[1] in black_squares) or \
                        (bishops_list[1] in white_squares and bishops_list[0] in black_squares):
                    pass
                else:
                    ok = False
                    break
        if b.is_valid() and ok:
            break
    return b


def get_next_positions(b):
    next_positions = []
    for move in b.legal_moves:
        next_position = b.copy(stack=False)
        next_position.push(move)
        next_positions.append(next_position)
    return next_positions


def generate_positions_from_full_tree(b, dataset_len):
    # создает полное дерево из заданной позиции и возвращает позиции из него
    # корневая позиция эндпильная
    previous_positions = [b.copy(stack=False)]
    all_tree_levels = []
    all_tree_levels.extend(previous_positions)
    breaking = False
    while True:
        next_positions = []
        for position in previous_positions:
            next_positions.extend(get_next_positions(position))
            all_tree_levels.extend(next_positions)
            if len(all_tree_levels) + len(next_positions) >= dataset_len:
                all_tree_levels.extend(next_positions)
                print('positions generated from tree', len(all_tree_levels))
                breaking = True
                break
        if breaking:
            break
        if len(next_positions) == 0:
            print('tree limited by end of game, cannot fill all dataset')
            return None
        all_tree_levels.extend(next_positions)
        print('positions generated from tree', len(all_tree_levels))
        previous_positions = next_positions
    return all_tree_levels[:dataset_len]


def generate_dataset_from_full_tree(b, dataset_len, learn_options):
    # создает полное дерево из заданной позиции и возвращает датасет из него
    coef = 1
    if learn_options.mirror_vertically:
        coef *= 2
    if learn_options.flip_horizontally:
        coef *= 2
    positions = generate_positions_from_full_tree(b, dataset_len // coef)
    root = Node()
    dataset_adding = []
    for board in positions:
        pre_evaluation(root, board)
        dataset_adding.append((board, root.value, True))
    return dataset_adding


if __name__ == '__main__':
    for i in range(100):
        print(generate_random_position(5))
        print()
