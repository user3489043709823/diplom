import chess as c
import os.path
from GoOptions import GoOptions

board = None  # доска, на которой будет проводиться партия
chess_960 = False
engine_name = 'Diplom'
root = None
try:
    log = open("log.txt", 'w')
except:
    log = None  # для проверки в print_with_log
    print("could not create log file")


def new_game():  # создает доску с новой игрой
    global board
    if chess_960:
        board = c.Board(chess960=True)
    else:
        board = c.Board()


new_game()  # на случай если первой командой будет go, stockfish например при этом работает


def print_with_log(x):  # Каждый вывод выводится одновременно и в log файл
    print(x)
    if log:
        log.write('engine: ' + x + '\n')
        log.flush()


engine_ready = False
get_move_function = None


def preparation():  # создание модели, загрузка весов и т.д.
    global get_move_function
    from model_chooser import model
    from constants import model_path, model_weights_path
    from search import get_move
    get_move_function = get_move
    if os.path.exists(model_path):
        print('loading model weights...')
        model.load_weights(model_weights_path)
        print('model weights loaded')


def whitespace_ignoring(string):
    string = string.replace('\t', ' ')  # убираем символы табуляции
    string = string.strip()  # убираем лидирующие пробелы и пробелы в конце строки
    while True:
        tmp = string.replace('  ', ' ')
        if tmp == string:
            string = tmp
            break
        string = tmp
    return string


def board_from_fen(fen_string):
    if chess_960:
        new_board = c.Board(fen_string, chess960=True)
    else:
        new_board = c.Board(fen_string)
    return new_board


last_position_starts_with_startpos = None  # если False, то она начиналась с fen
while True:
    console_input = input()
    console_input = whitespace_ignoring(console_input)
    if console_input:  # если ввод пустой, то ничего не делаем
        if log:  # записываем то, что введено, в лог файл
            log.write('input: ' + console_input + '\n')
            log.flush()
        if console_input == 'xboard':
            continue  # не поддерживаем xboard пока что
        elif console_input == 'isready':
            if not board:
                new_game()
            if not engine_ready:
                preparation()
                engine_ready = True
            print_with_log("readyok")
        elif console_input == 'uci':
            print_with_log("id name " + engine_name)
            print_with_log("id author Belozerov Andrey")
            print_with_log("uciok")
        elif console_input == 'ucinewgame':
            new_game()
            root = None
            last_position_starts_with_startpos = True
        elif console_input.startswith('position startpos moves'):
            moves = console_input.split()[3:]
            moves_chess_type = [c.Move.from_uci(i) for i in moves]
            if not (root is None) and last_position_starts_with_startpos and len(moves_chess_type) > 0\
                    and board.move_stack == moves_chess_type[:-1]:
                # если позиции различаются одним ходом
                for i in root.children:
                    if i.move_to_this_node == moves_chess_type[-1]:
                        root = i
                        board.push(moves_chess_type[-1])
                        break
            else:
                new_game()
                for move in moves:
                    board.push_uci(move)
                root = None
            last_position_starts_with_startpos = True
        elif console_input.startswith('position fen'):
            console_input_split = console_input.split()
            fen = console_input_split[2:8]  # Shredder FEN, в нем 6 разделов
            fen = ' '.join(fen)
            if len(console_input_split) > 8 and console_input_split[8] == 'moves':
                moves = console_input_split[9:]
            else:
                moves = []
            moves_chess_type = [c.Move.from_uci(i) for i in moves]
            if not (root is None) and last_position_starts_with_startpos is False and fen == last_fen \
                    and len(moves_chess_type) > 0 and board.move_stack == moves_chess_type[:-1]:
                    # условие на длину чтобы не было случая когда длины 0 и 0
                for i in root.children:
                    if i.move_to_this_node == moves_chess_type[-1]:
                        root = i
                        board.push(moves_chess_type[-1])
                        break
            else:
                if not (board and board.fen() == fen):
                    board = board_from_fen(fen)
                    root = None
                for move in moves:
                    board.push_uci(move)
                    root = None
            last_position_starts_with_startpos = False
            last_fen = fen
        elif console_input.startswith('go'):
            if not board:
                new_game()
                last_position_starts_with_startpos = True
            go_options = GoOptions()
            console_input_split = console_input.split()
            if 'wtime' in console_input_split:
                index = console_input_split.index('wtime')
                go_options.wtime = int(console_input_split[index + 1])
            if 'btime' in console_input_split:
                index = console_input_split.index('btime')
                go_options.btime = int(console_input_split[index + 1])
            if 'nodes' in console_input_split:
                index = console_input_split.index('nodes')
                go_options.nodes = int(console_input_split[index + 1])
            if 'movetime' in console_input_split:
                index = console_input_split.index('movetime')
                go_options.movetime = int(console_input_split[index + 1])
            position_value, best_move, best_node = get_move_function(board, printing=False, root=root,
                                                                     go_options=go_options,
                                                                     print_info=True)[0:3]
            root = best_node.parent
            print_with_log("bestmove " + best_move)
        elif console_input == 'quit':
            break
        elif console_input == 'printpos':
            print(board)
        else:
            continue  # игнорируем неизвестную команду
