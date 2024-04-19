"""

第17条 用defaultdict处理内部状态中缺失的元素，而不要用setdefault

"""

from collections import defaultdict

visits = {
    'Mexico': {'Tulum', 'Puerto vallarta'},
    'Japan': {'Hakone'}
}

visits.setdefault('France', set()).add('Arles')

if (japan := visits.get('Japan')) is None:
    visits['Japan'] = japan = set()
japan.add('Kyoto')
print(visits)


class Visits:
    def __init__(self):
        self.data = {}

    def add(self, country, city):
        city_set = self.data.setdefault(country, set())
        city_set.add(city)


visits = Visits()
visits.add('Russia', 'Yekaterinburg')
visits.add('Tanzania', 'Zanzibar')

print(visits.data)


class Visits:
    def __init__(self):
        self.data = defaultdict(set)

    def add(self, country, city):
        self.data[country].add(city)


visits = Visits()
visits.add('England', 'Bath')
visits.add('England', 'London')

print(visits.data)

