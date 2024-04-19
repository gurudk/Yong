def log(message, *values):
    if not values:
        print(message)
    else:
        values_str = ", ".join(str(x) for x in values)
        print(f'message: {values_str}')


log('My numbers is :', [1, 2, 3, 4])
log('Hi there')