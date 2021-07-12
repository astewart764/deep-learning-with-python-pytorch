import numpy as np

list1 = [0, 0, 0, 1, 1, 0, 1, 0]
list2 = [1, 0, 1, 1, 0, 0, 0, 0]

list3 = [int(list1[idx] == list2[idx]) for idx in range(len(list1))]
print(list3)

print( int(sum(list3)/len(list3) * 100) )
