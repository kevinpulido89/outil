# from k_util import describe
# import numpy as np
#
# a = [1, 2]
# b = np.array([[1, 2, 0],
#               [3, 4, 9]], copy=True)
#
# describe(b)  # --> Type: <class 'numpy.ndarray'> || Size: (2, 3)

d = {'c':2,
     'b':4,
     'a':19
}

# print([(k,v) for k , v in sorted(d.items(), key=lambda x: x[1])])
print(sorted(d.items(), key=lambda x: x[0]))



# print(d.items())

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

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
