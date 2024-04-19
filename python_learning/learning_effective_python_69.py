""""

第69条 在需要准确计算的场合， 使用decimal表示相应的数值

"""

rate = 1.45
seconds = 3*60 + 42
cost = rate * seconds / 60
print(cost)

from decimal import Decimal

rate = Decimal('1.45')
seconds = Decimal(3*60 + 42)
cost = rate * seconds / Decimal(60)
print(cost)

rate = Decimal('0.05')
seconds = Decimal(5)
cost = rate * seconds / Decimal(60)
print(cost)


from decimal import ROUND_UP

rounded = cost.quantize(Decimal('0.01'), rounding = ROUND_UP)
print(f'Rounded {cost} to {rounded}')

