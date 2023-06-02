from position_generation import generate_random_position
import chess.svg
import os
if __name__ == '__main__':
    if not os.path.exists('svg'):
        os.mkdir('svg')
    for i in range(100):
        b = generate_random_position(9)
        # 12 - нет
        # 11 спорно
        # 9 точно нормально
        b_svg = chess.svg.board(board=b, size=8 * 64 * 2)
        with open('svg/' + str(i) + '.svg', 'w') as outputfile:
            outputfile.write(b_svg)
