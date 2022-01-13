
import argparse

parser=argparse.ArgumentParser(description='숫자를 받아서 더할 겁니다.')

parser.add_argument('--a', required=True, help ='숫자 a', type=int)
parser.add_argument('--b', required=False, default='dev', help='숫자 b',type=int)

args=parser.parse_args()

print(args.a)
print(args.b)

print(f'더해진 숫자는 {args.a+args.b}')