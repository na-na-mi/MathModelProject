def space(cargo_list, long, w):
    alist = []
    for soul in cargo_list:
        for j in cargo_list:
            if soul == j:
                continue
            else:
                u = (soul * j * (long // soul) * (w // j)) / (long * w)
                alist.append(u)
    return alist


my_list = [[1.32, 0.64, 0.84], [0.98, 0.42, 0.52], [1.5, 1]]
for i in my_list:
    x = space(i, 2.891, 2.338)
    x.sort()
    print(x.pop())
print("\n")
for i in my_list:
    z = space(i, 2.235, 2.643)
    z.sort()
    print(z.pop())
