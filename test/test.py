import numpy
import pandas
arr = numpy.array([1, 2, 3, 4, 5])

print(arr)


mydataset = {
  'cars': ["BMW", "Volvo", "Ford"],
  'passings': [3, 7, 2]
}

myvar = pandas.DataFrame(mydataset)

print(myvar)