# самое затратное в обучении с подкреплением - это оценка позиции. Кроме этого, чем быстрее это, тем лучше поиск
# в файле оценивается время оценки пакетов нейронной сетью
from time import time
from parallel_evaluation import batch_evaluation

print('Предварительный запуск')
batch_evaluation()
print('Начало тестирования')
start = time()
for i in range(50):
    print('итерация', i)
    batch_evaluation()
print('Время', time() - start)
