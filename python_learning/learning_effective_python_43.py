"""

第43条 自定义的容器类型应该从collections.abc继承

"""


from collections.abc import Sequence


class FrequencyList(list):
    def __init__(self, members):
        super().__init__(members)

    def frequency(self):
        counts = {}
        for item in self:
            counts[item] = counts.get(item, 0) + 1
        return counts


foo = FrequencyList(['a', 'b', 'a', 'c', 'b', 'a', 'd'])
print('Length is:', len(foo))

foo.pop()
print('After pop:', repr(foo))
print('Frequency:', foo.frequency())


class BinaryNode():
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right


bar = [1,2,3]

assert bar[0] == bar.__getitem__(0)


class IndexableNode(BinaryNode):
    def _traverse(self):
        if self.left is not None:
            yield from self.left._traverse()
        yield self
        if self.right is not None:
            yield from self.right._traverse()

    def __getitem__(self, index):
        for i, item in enumerate(self._traverse()):
            if i == index:
                return item.value
        raise IndexError(f'Index {index} is out of range')


tree = IndexableNode(10, left=IndexableNode(5, left=IndexableNode(2), right=IndexableNode(6, right=IndexableNode(7))),
                     right=IndexableNode(15, left=IndexableNode(11)))


print('LRR is', tree.left.right.right.value)
print('Index 0 is', tree[0])
print('Index 1 is', tree[1])
print('11 in the tree?', 11 in tree)
print('17 in the tree?', 17 in tree)
print('tree is ', list(tree))


class SequenceNode(IndexableNode):
    def __len__(self):
        for count, _ in enumerate(self._traverse(), 1):
            print(f'count:{count}')
        return count


seq_tree = SequenceNode(10, left=SequenceNode(5,
                                              left=SequenceNode(2),
                                              right=SequenceNode(6,
                                                                 right=SequenceNode(7))),
                                 right=SequenceNode(15, left=SequenceNode(11)))

print('the len is :', len(seq_tree))


class BadType(Sequence):
    pass

# foo = BadType()


class BetterTree(SequenceNode, Sequence):
    pass


better_tree = BetterTree(10, left=BetterTree(5, left=BetterTree(2), right=BetterTree(6, right=BetterTree(7))),
                          right=BetterTree(15, left=BetterTree(11)))


print('Index of 7 is', better_tree.index(7))
print('Count os 10 is', better_tree.count(10))
