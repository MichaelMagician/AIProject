from typing import List


class MyClass(object):
    def __init__(self, word, count):
        self.word = word
        self.count = count

    def __lt__(self, other):
        return self.count.__lt__(other.count)
    
    def __str__(self) -> str:
        return f"{self.word}: {self.count}"
    
# cl1 = MyClass("a", 2)
# cl2 = MyClass("b", 3)

# li = [cl1, cl2]
# li.sort()
# print([str(l) for l in li])




# import pandas as pd
# from datetime import datetime, time
# data = pd.DataFrame(
# {
#  "name":["Bob", "Jacob"],
#  "first_appear":[datetime.strptime("12:10:00", '%H:%M:%S').time(), datetime.strptime('12:31:00', '%H:%M:%S').time()],
#  "last_appear":[datetime.strptime('12:33:49', '%H:%M:%S').time(), datetime.strptime('13:29:12', '%H:%M:%S').time()]
#  }
# )

import bisect
a=[['10', 'name_1'],['50','name_2'],['40','name_3'], ['60','name_4'], ['80', 'name_N'], ['90', 'name_N'], ['101', 'name_N']]
b=[(10,40),(40,60),(60,90),(90,100)]
# [10,40),[40,60),[60,90),[90,100)

# [['10', 'name_1'], [['50','name_2'],['40','name_3']]]
def group_by_intervals(a, b):
    bins = [ int(e[0]) for e in b]
    bins.sort()
    bins.append(int(b[-1][1]))
    element_map = {} 
    for data in a:
        num = int(data[0])
        index = bisect.bisect_left(bins, num)
        if index == len(bins) -1:
            continue
        element_map[index] = element_map.get(index, [])
        element_map[index].append(data)
    print(element_map.values())
group_by_intervals(a, b)

# []