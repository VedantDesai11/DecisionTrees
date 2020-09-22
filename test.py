d = {1:[1,2,2,2,3], 2:[4,5,5,5,6],3:[7,8,8,8,8,9]}

#print([d[a] for a in d.keys()])

for values in d.values():
	print(list(set(values)))