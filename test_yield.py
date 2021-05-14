def make_iterables_to_chain():
    yield [1, 2, 3]
    yield ['a', 'b', 'c']

data_yield = make_iterables_to_chain()
data_yield = list(data_yield)
print(data_yield)