import numpy as np
import warnings


warnings.filterwarnings('ignore')


def linear_function(x, p):
  '''Parameters: 1. p[0]: slope 2. p[1]: x-intercept (critical mass)'''
  x = np.array(x)
#   return p[0] + p[1]*x
  return p[0]*(x - p[1])


def quadratic_function(x, p):
  x = np.array(x)
#   return p[0] + p[1]*x + p[2]*np.square(x)
  return (p[2]*x + p[0])*(x - p[1])

def simple_exponential_function(x, p):

  x = np.array(x)
  return p[0]*np.exp(-p[1]*x)

