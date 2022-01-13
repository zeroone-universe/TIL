import logging

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
    print(f'Add:{add(x,y)}')
    print(f'Sub:{subtract(x,y)}')
    print(f'Mul:{multiply(x,y)}')
    print(f'Div:{divide(x,y)}')