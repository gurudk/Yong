import time

address = 'Four score and seven years ago'


def index_words_iter(text):
    if text:
        yield 0
    for index, letter in enumerate(text):
        if letter == ' ':
            yield index + 1


it = index_words_iter(address)
result = list(index_words_iter(address))
print(result)
print(next(it))
print(next(it))
print(next(it))