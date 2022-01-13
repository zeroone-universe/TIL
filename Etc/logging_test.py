import os
import logging


logging.basicConfig(filename='F:/TIL/Etc/test.log',level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def add(x,y):
    return x+y

def subtract(x,y):
    return x-y

def multiply(x,y):
    return x*y

def divide(x,y):
    return x/y

if __name__=='__main__':
    x=10
    y=5
    logging.debug(f'add: {x} + {y} = {add(x,y)}')
    logging.info(f'subtract: {x} + {y} = {subtract(x,y)}')
    logging.debug(f'multiply: {x} + {y} = {multiply(x,y)}')
    logging.debug(f'divide: {x} + {y} = {divide(x,y)}')
    print('')