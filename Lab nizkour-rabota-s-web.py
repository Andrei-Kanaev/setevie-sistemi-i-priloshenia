#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Стиль Google

import numpy as np

def add_row(board):
    """
    Добавляет новую строку к доске.

    Parameters:
    board (numpy.ndarray): Исходная доска.

    Returns:
    numpy.ndarray: Доска с добавленной строкой.
    """
    new_board = []
    for row in board:
        new_board += list(row)
    new_board = list(filter(lambda cell: cell != 'x', new_board))
    new_board = new_board + [' '] * (27 - len(new_board))

    board_rows = []
    for row in board:
        board_rows += list(row)
    board = board_rows

    board = list(filter(lambda cell: cell != ' ', board))
    board = board + new_board

    board = [board[i:i + 9] for i in range(0, len(board), 9)]
    return np.array(board)


def check(row1, col1, row2, col2):
    """
    Проверяет, возможен ли ход между двумя клетками.

    Parameters:
    row1 (int): Номер строки первой клетки.
    col1 (int): Номер столбца первой клетки.
    row2 (int): Номер строки второй клетки.
    col2 (int): Номер столбца второй клетки.

    Returns:
    bool: True, если ход возможен, иначе False.
    """
    if row1 == row2 and col1 != col2:
        space = set(board[row1, col1 + 1:col2])
    elif col1 == col2 and row1 != row2:
        space = set(board[row1 + 1:row2, :])
    elif col1 != col2 and abs(row1 - row2) == 1:
        if row1 < row2:
            up = set(board[row1, col1 + 1:])
            down = set(board[row2, :col2])
        else:
            up = set(board[row2, :col2])
            down = set(board[row1, col1 + 1:])
        space = up | down
    if len(space) == 1 and 'x' in space or len(space) == 0:
        return True
    return False


def del_empty_row(board):
    """
    Удаляет пустые строки с доски.

    Parameters:
    board (numpy.ndarray): Исходная доска.

    Returns:
    numpy.ndarray: Доска без пустых строк.
    """
    board = [list(row) for row in board]
    row = ['x'] * 9
    while row in board:
        board.remove(row)
    board = np.array(board)
    return board


def get_coords(cell):
    """
    Получает координаты клетки на доске.

    Parameters:
    cell (str): Координаты клетки в формате 'Б2'.

    Returns:
    tuple: Координаты клетки в формате (строка, столбец).
    """
    abc = 'АБВГДЕЖЗИ'
    letter, digit = cell
    letter = abc.index(letter)
    digit = int(digit) - 1
    return digit, letter


def print_board(board):
    """
    Выводит доску в консоль.

    Parameters:
    board (numpy.ndarray): Доска для вывода.
    """
    max_digits = len(str(len(board)))
    abc = ' ' * max_digits + '  А Б В Г Д Е Ж З И'
    pretty_board = [
        f'{i}){""}' + ' '.join(row) for i, row in enumerate(board, start=1)
    ]

    pretty_board = []
    for i, row in enumerate(board, start=1):
        row_num = f'{i}) ' + (' ' * (max_digits - len(str(i))))
        pretty_board.append(row_num + ' '.join(row))

    pretty_board = '\n'.join(pretty_board)
    pretty_board = f'\n{abc}\n{pretty_board}'
    print(pretty_board)


board = np.array([
    ['1', '2', '3', '4', '5', '6', '7', '8', '9'],
    ['1', '1', '1', '2', '1', '3', '1', '4', '1'],
    ['5', '1', '6', '1', '7', '1', '8', '1', '9'],
])


menu = '''
Для хода введите координаты клеток
Например, А1 Б2
Для добавления строк отправьте "+"
Для завершения игры нажмите CTRL+C
Ввод: '''

while True:
    print_board(board)
    try:
        coords = input(menu).upper()
    except KeyboardInterrupt:
        print('До свидания!')
        exit()
    if len(coords) == 5:
        cell_1, cell_2 = coords.split()
        row1, col1 = get_coords(cell_1)
        row2, col2 = get_coords(cell_2)
        value_1 = board[row1][col1]
        value_2 = board[row2][col2]
        if value_1 != 'x' and value_2 != 'x':
            value_1 = int(value_1)
            value_2 = int(value_2)
            if value_1 == value_2 or value_1 + value_2 == 10:
                if check(row1, col1, row2, col2):
                    board[row1][col1] = 'x'
                    board[row2][col2] = 'x'
                else:
                    print('Неправильный ход!')
        else:
            print('Неправильный ход!')
    elif coords == '+':
        board = add_row(board)
    else:
        print('Некорректная команда!')

    board = del_empty_row(board)

    if len(board) == 0:
        print('ПОЗДРАВЛЯЕМ! ВЫ ПОБЕДИЛИ!')
        break

