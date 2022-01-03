import sys

def maxProfit(prices):
    profit=0
    minimum_price=sys.maxsize

    for price in prices:
        minimum_price=min(minimum_price, price)
        profit=max(profit, price-minimum_price)

    return profit


if __name__ =='__main__':
    print(maxProfit([7,1,5,3,6,4]))