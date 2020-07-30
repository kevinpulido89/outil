# from k_util import describe
# import numpy as np
#
# a = [1, 2]
# b = np.array([[1, 2, 0],
#               [3, 4, 9]], copy=True)
#
# describe(b)  # --> Type: <class 'numpy.ndarray'> || Size: (2, 3)

# d = {'c':2,
#      'b':4,
#      'a':19
# }

# print([(k,v) for k , v in sorted(d.items(), key=lambda x: x[1])])
# print(sorted(d.items(), key=lambda x: x[0]))

# print(d.items())

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# ll = [2, 3, 5, 7, 9]

# mx = float('-inf')
#
# # for idx, e in enumerate([99,88]):
# #     print(idx)
#
# for e, a in zip(ll[:-1], ll[1:]):
#     c = e*a
#
#     if c>mx:
#         mx = c
# print(mx)

# try:
#     print(max(e*a for e,a in zip(ll[:-1], ll[1:])))
# except Exception as e:
#     raise float('-inf')

# print([e*a for e,a in zip(ll[:-1], ll[1:])])

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# from datetime import datetime
# import pandas as pd
#
# def getm(number):
#     if number < 10 :
#         return '0' + str(number)
#     else:
#         return str(number)
#
# data  = pd.DataFrame([[datetime(year=2019, month=8, day=3),1],
#                       [datetime(year=2019, month=9, day=4),2],
#                       [datetime(year=2019, month=8, day=8),3]],
#                       columns=['created', 'id'])
#
# data['month'] = data['created'].apply(lambda fecha: '{y}-{m}'.format(y=fecha.year, m=getm(fecha.month)))
#
# # data.drop('created', inplace=True)
#
# data.sort_values(by=['month'], inplace=True)
#
# gp = data.groupby(by='month').aggregate({'id':'count'})
# print(gp)
#
# df = pd.DataFrame(gp)
#
# print(gp.loc['2019-08','id'])

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# import numpy as np
# import matplotlib.pyplot as plt
#
# d1 = np.random.normal(loc=0.1, scale=3, size=2000)
# d2 = np.random.normal(loc=-2,  scale=1, size=2000)
# d3 = np.random.normal(loc=3,   scale=2, size=2000)
# d4 = np.random.normal(loc=7,   scale=2, size=2000)
#
# kwargs = dict(histtype='stepfilled',
#               alpha=0.3,
#               density=True,
#               bins=40)
#
# plt.hist(d1, **kwargs)
# plt.hist(d2, **kwargs)
# plt.hist(d3, **kwargs)
# plt.hist(d4, **kwargs)
# plt.show()
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# import  numpy as np
#
# arr = np.array([2,3,4,3,2,1])
#
# def counter(array, n):
#     C = 0
#     for e in array:
#         if e == n:
#             C +=1
#         else:
#             pass
#
#     return C
#
# k = counter(arr,2)
#
# # print(k)
#
# def counter2(array, n):
#     U, C = np.unique(array, return_counts=True)
#     D = dict(zip(U, C))
#     return D[n]

# kk = counter2(arr,2)
#
# print(kk)
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Coding interview
# https://www.youtube.com/watch?v=GJdiM-muYqc

# def first_recurring(string):
#     letra = {}
#
#     for char in string:
#         if char in letra:
#             return char
#         letra[char] = 1
#
#     return None
#
# s = "ABCkfsbkfadfoasfjdljflajfl"
# print(first_recurring(s))


# https://www.youtube.com/watch?v=uQdy914JRKQ

# nums = [9,9,9]
#
# def unir_sumar(lista):
#     tmp = ''
#
#     for e in lista:
#         tmp = tmp+str(e)
#
#     ans = int(tmp)+1
#
#     tmp = []
#
#     for s in str(ans):
#         tmp.append(int(s))
#
#     return tmp
#
# print(unir_sumar(nums))

s = 'Bananae'

vowels = 'AEIOU'

stusc = 0
kevsc = 0

for i in range(len(s)):
    if s[i] in vowels:
        kevsc += (len(s)-i)
    else:
        stusc += (len(s)-i)

if kevsc > stusc:
    print("Kevin", kevsc)
elif kevsc < stusc:
    print ("Stuart", stusc)
else:
    print ("Draw")
