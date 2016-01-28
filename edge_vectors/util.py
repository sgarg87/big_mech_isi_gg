

def unique_list(seq):
    # todo: this code may be in accurate, verify the code
    seen = set()
    seen_add = seen.add
    sol = [x for x in seq if not (x in seen or seen_add(x))]
    if len(sol) != len(set(sol)):
        raise AssertionError
    return sol
