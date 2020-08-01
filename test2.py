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


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Split columns in pandas

# import pandas as pd
#
# df = pd.DataFrame({'AB': ['A1-B1', 'A2-B2']})
#
# df[['A', 'B']] = df['AB'].str.split('-', 1, expand=True)
# print(df)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# def nearest_square(num=int):
#     '''
#     Return the nearest square perfect square that is less than or equal to num
#     '''
#
#     root = 0
#     while (root + 1)**2 <= num:
#         root += 1
#     return root ** 2
#
# # TDD> Test driven development
# # https://www.linkedin.com/pulse/data-science-test-driven-development-sam-savage/
# # https://medium.com/uk-hydrographic-office/test-driven-development-is-essential-for-good-data-science-heres-why-db7975a03a44
# # https://engineering.pivotal.io/post/test-driven-development-for-data-science/
# # https://docs.python-guide.org/writing/tests/
#
# print(nearest_square(65))
#
# def validator_nearest_square():
#     assert nearest_square(5) == 4
#     assert nearest_square(9) == 9, 'the value should be 9'
#     assert nearest_square(-12) == 0

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# class Pants(object):
#     """docstring for Pants class with the following attributes:
#
#     - color (string) eg 'red', 'yellow', 'orange'
#     - waist_size (integer) eg 8, 9, 10, 32, 33, 34
#     - length (integer) eg 27, 28, 29, 30, 31
#     - price (float) eg 9.28
#     """
#
#     def __init__(self, color=str, waist_size=int, length=int, price=float):
#         super(Pants, self).__init__()
#         self.color = color
#         self.waist_size = waist_size
#         self.length = length
#         self.price = price
#
#     def change_price(self, new_price):
#         self.price = new_price
#
#     def discount(self, value_discount=float):
#         return self.price * (1-value_discount)
#
#
# class SalesPerson(object):
#     """The SalesPerson class represents an employee in the store
#
#     """
#
#     def __init__(self, first_name=str, last_name=str, employee_id=int, salary=float):
#         '''Method for initializing a SalesPerson object
#
#         Args:
#             first_name (str)
#             last_name (str)
#             employee_id (int)
#             salary (float)
#
#         Attributes:
#             first_name (str): first name of the employee
#             last_name (str): last name of the employee
#             employee_id (int): identification number of the employee
#             salary (float): yearly salary of the employee
#             pants_sold (list): a list of pants objects sold by the employee
#             total_sales (float): sum of all sales made by the employee
#         '''
#         super(SalesPerson, self).__init__()
#         self.first_name = first_name
#         self.last_name = last_name
#         self.employee_id = employee_id
#         self.salary = salary
#         self.pants_sold = list()
#         self.total_sales = 0.0
#
#     def sell_pants(self, pants_obj):
#         """The sell_pants method appends a pants object to the pants_sold attribute
#
#         Args:
#             pants_object (obj): a pants object that was sold
#
#         Returns: None
#         """
#         self.pants_sold.append(pants_obj)
#
#     def display_sales(self):
#         """The display_sales method prints out all pants that have been sold
#
#         Args: None
#
#         Returns: None
#         """
#         for sp in self.pants_sold:
#             print(f'color: {sp.color}, waist_size: {sp.waist_size},length: {sp.length}, price: {sp.price}')
#
#     def calculate_sales(self):
#         """The calculate_sales method sums the total price of all pants sold
#
#         Args: None
#
#         Returns:
#             float: sum of the price for all pants sold
#         """
#         self.total_sales = sum([sp.price for sp in self.pants_sold])
#         return self.total_sales
#
#     def calculate_commission(self, percentage=float):
#         """
#         The calculate_commission method outputs the commission based on sales
#
#         Args:
#             percentage (float): the commission percentage as a decimal
#
#         Returns:
#             float: the commission due
#         """
#         return self.calculate_sales() * percentage
#
#
# def check_results():
#     pants_one = Pants('red', 35, 36, 15.12)
#     pants_two = Pants('blue', 40, 38, 24.12)
#     pants_three = Pants('tan', 28, 30, 8.12)
#
#     salesperson = SalesPerson('Amy', 'Gonzalez', 2581923, 40000)
#
#     assert salesperson.first_name == 'Amy'
#     assert salesperson.last_name == 'Gonzalez'
#     assert salesperson.employee_id == 2581923
#     assert salesperson.salary == 40000
#     assert salesperson.pants_sold == []
#     assert salesperson.total_sales == 0
#
#     salesperson.sell_pants(pants_one)
#     salesperson.pants_sold[0] == pants_one.color
#
#     salesperson.sell_pants(pants_two)
#     salesperson.sell_pants(pants_three)
#
#     assert len(salesperson.pants_sold) == 3
#     assert round(salesperson.calculate_sales(),2) == 47.36
#     assert round(salesperson.calculate_commission(.1),2) == 4.74
#
#     print('Great job, you made it to the end of the code checks!')
#
# check_results()
# #
# pants_one = Pants('red', 35, 36, 115.12)
# pants_two = Pants('blue', 40, 38, 124.12)
# pants_three = Pants('tan', 28, 30, 118.12)
#
# new_salesperson = SalesPerson('Amy', 'Gonzalez', 2581923, 40000)
#
# new_salesperson.sell_pants(pants_one)
# new_salesperson.sell_pants(pants_two)
# new_salesperson.sell_pants(pants_three)
#
# new_salesperson.display_sales()

# > ###########################################################################
#
# import math
# import matplotlib.pyplot as plt
#
# class Gaussian():
#     """ Gaussian distribution class for calculating and
#     visualizing a Gaussian distribution.
#
#     Attributes:
#         mean (float) representing the mean value of the distribution
#         stdev (float) representing the standard deviation of the distribution
#         data_list (list of floats) a list of floats extracted from the data file
#
#     """
#     def __init__(self, mu = 0, sigma = 1):
#
#         self.mean = mu
#         self.stdev = sigma
#         self.data = []
#
#
#     def calculate_mean(self):
#
#         """Function to calculate the mean of the data set.
#
#         Args:
#             None
#
#         Returns:
#             float: mean of the data set
#
#         """
#
#         avg = 1.0 * sum(self.data) / len(self.data)
#
#         self.mean = avg
#
#         return self.mean
#
#
#
#     def calculate_stdev(self, sample=True):
#
#         """Function to calculate the standard deviation of the data set.
#
#         Args:
#             sample (bool): whether the data represents a sample or population
#
#         Returns:
#             float: standard deviation of the data set
#
#         """
#
#         if sample:
#             n = len(self.data) - 1
#         else:
#             n = len(self.data)
#
#         mean = self.mean
#
#         sigma = 0
#
#         for d in self.data:
#             sigma += (d - mean) ** 2
#
#         sigma = math.sqrt(sigma / n)
#
#         self.stdev = sigma
#
#         return self.stdev
#
#
#     def read_data_file(self, file_name, sample=True):
#
#         """Function to read in data from a txt file. The txt file should have
#         one number (float) per line. The numbers are stored in the data attribute.
#         After reading in the file, the mean and standard deviation are calculated
#
#         Args:
#             file_name (string): name of a file to read from
#
#         Returns:
#             None
#
#         """
#
#         with open(file_name) as file:
#             data_list = []
#             line = file.readline()
#             while line:
#                 data_list.append(int(line))
#                 line = file.readline()
#         file.close()
#
#         self.data = data_list
#         self.mean = self.calculate_mean()
#         self.stdev = self.calculate_stdev(sample)
#
#
#     def plot_histogram(self):
#         """Function to output a histogram of the instance variable data using
#         matplotlib pyplot library.
#
#         Args:
#             None
#
#         Returns:
#             None
#         """
#         plt.hist(self.data)
#         plt.title('Histogram of Data')
#         plt.xlabel('data')
#         plt.ylabel('count')
#
#
#
#     def pdf(self, x):
#         """Probability density function calculator for the gaussian distribution.
#
#         Args:
#             x (float): point for calculating the probability density function
#
#
#         Returns:
#             float: probability density function output
#         """
#
#         return (1.0 / (self.stdev * math.sqrt(2*math.pi))) * math.exp(-0.5*((x - self.mean) / self.stdev) ** 2)
#
#
#     def plot_histogram_pdf(self, n_spaces = 50):
#
#         """Function to plot the normalized histogram of the data and a plot of the
#         probability density function along the same range
#
#         Args:
#             n_spaces (int): number of data points
#
#         Returns:
#             list: x values for the pdf plot
#             list: y values for the pdf plot
#
#         """
#
#         mu = self.mean
#         sigma = self.stdev
#
#         min_range = min(self.data)
#         max_range = max(self.data)
#
#          # calculates the interval between x values
#         interval = 1.0 * (max_range - min_range) / n_spaces
#
#         x = []
#         y = []
#
#         # calculate the x values to visualize
#         for i in range(n_spaces):
#             tmp = min_range + interval*i
#             x.append(tmp)
#             y.append(self.pdf(tmp))
#
#         # make the plots
#         fig, axes = plt.subplots(2,sharex=True)
#         fig.subplots_adjust(hspace=.5)
#         axes[0].hist(self.data, density=True)
#         axes[0].set_title('Normed Histogram of Data')
#         axes[0].set_ylabel('Density')
#
#         axes[1].plot(x, y)
#         axes[1].set_title('Normal Distribution for \n Sample Mean and Sample Standard Deviation')
#         axes[0].set_ylabel('Density')
#         plt.show()
#
#         return x, y
#
#     def __add__(self, other):
#
#         """Function to add together two Gaussian distributions
#
#         Args:
#             other (Gaussian): Gaussian instance
#
#         Returns:
#             Gaussian: Gaussian distribution
#
#         """
#
#         result = Gaussian()
#         result.mean = self.mean + other.mean
#         result.stdev = math.sqrt(self.stdev ** 2 + other.stdev ** 2)
#
#         return result
#
#
#     def __repr__(self):
#
#         """Function to output the characteristics of the Gaussian instance
#
#         Args:
#             None
#
#         Returns:
#             string: characteristics of the Gaussian
#
#         """
#
#         return f"mean {self.mean}, standard deviation {self.stdev}"
#
# # Unit tests to check your solution
#
# import unittest
#
# class TestGaussianClass(unittest.TestCase):
#     def setUp(self):
#         self.gaussian = Gaussian(25, 2)
#
#     def test_initialization(self):
#         self.assertEqual(self.gaussian.mean, 25, 'incorrect mean')
#         self.assertEqual(self.gaussian.stdev, 2, 'incorrect standard deviation')
#
#     def test_pdf(self):
#         self.assertEqual(round(self.gaussian.pdf(25), 5), 0.19947,\
#          'pdf function does not give expected result')
#
#     def test_meancalculation(self):
#         self.gaussian.read_data_file('numbers.txt', True)
#         self.assertEqual(self.gaussian.calculate_mean(),\
#          sum(self.gaussian.data) / float(len(self.gaussian.data)), 'calculated mean not as expected')
#
#     def test_stdevcalculation(self):
#         self.gaussian.read_data_file('numbers.txt', True)
#         self.assertEqual(round(self.gaussian.stdev, 2), 92.87, 'sample standard deviation incorrect')
#         self.gaussian.read_data_file('numbers.txt', False)
#         self.assertEqual(round(self.gaussian.stdev, 2), 88.55, 'population standard deviation incorrect')
#
#     def test_add(self):
#         gaussian_one = Gaussian(25, 3)
#         gaussian_two = Gaussian(30, 4)
#         gaussian_sum = gaussian_one + gaussian_two
#
#         self.assertEqual(gaussian_sum.mean, 55)
#         self.assertEqual(gaussian_sum.stdev, 5)
#
#     def test_repr(self):
#         gaussian_one = Gaussian(25, 3)
#
#         self.assertEqual(str(gaussian_one), "mean 25, standard deviation 3")
#
# tests = TestGaussianClass()
#
# tests_loaded = unittest.TestLoader().loadTestsFromModule(tests)
#
# unittest.TextTestRunner().run(tests_loaded)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class Clothing(object):

    def __init__(self, color, size, style, price):
        self.color = color
        self.size = size
        self.style = style
        self.price = price

    def change_price(self, price):
        self.price = price

    def calculate_discount(self, discount):
        return self.price * (1 - discount)

    def calculate_shipping(self, weight, rate):
        self.weight = weight
        self.rate = rate
        return self.weight * self.rate

# TODO: Add a method to the clothing class called calculate_shipping.
#   The method has two inputs: weight and rate. Weight is a float
#   representing the weight of the article of clothing. Rate is a float
#   representing the shipping weight. The method returns weight * rate

class Shirt(Clothing):

    def __init__(self, color, size, style, price, long_or_short):

        Clothing.__init__(self, color, size, style, price)
        self.long_or_short = long_or_short

    def double_price(self):
        self.price = 2*self.price

class Pants(Clothing):

    def __init__(self, color, size, style, price, waist):

        Clothing.__init__(self, color, size, style, price)
        self.waist = waist

    def calculate_discount(self, discount):
        return self.price * (1 - discount / 2)

class Blouse(Clothing):
    """docstring for Blouse."""

    def __init__(self, color, size, style, price, country_of_origin):

        Clothing.__init__(self, color, size, style, price)
        self.country_of_origin = country_of_origin

    def triple_price(self):
        return self.price*3

import unittest

class TestClothingClass(unittest.TestCase):
    def setUp(self):
        self.clothing = Clothing('orange', 'M', 'stripes', 35)
        self.blouse = Blouse('blue', 'M', 'luxury', 40, 'Brazil')
        self.pants = Pants('black', 32, 'baggy', 60, 30)

    def test_initialization(self):
        self.assertEqual(self.clothing.color, 'orange', 'color should be orange')
        self.assertEqual(self.clothing.price, 35, 'incorrect price')

        self.assertEqual(self.blouse.color, 'blue', 'color should be blue')
        self.assertEqual(self.blouse.size, 'M', 'incorrect size')
        self.assertEqual(self.blouse.style, 'luxury', 'incorrect style')
        self.assertEqual(self.blouse.price, 40, 'incorrect price')
        self.assertEqual(self.blouse.country_of_origin, 'Brazil', 'incorrect country of origin')

    def test_calculateshipping(self):
        self.assertEqual(self.clothing.calculate_shipping(.5, 3), .5 * 3,\
         'Clothing shipping calculation not as expected')

        self.assertEqual(self.blouse.calculate_shipping(.5, 3), .5 * 3,\
         'Clothing shipping calculation not as expected')

tests = TestClothingClass()

tests_loaded = unittest.TestLoader().loadTestsFromModule(tests)

unittest.TextTestRunner().run(tests_loaded)
