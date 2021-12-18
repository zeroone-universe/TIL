def array_pair_sum(nums):
    sum=0
    pair=[]
    nums.sort()
    nums.reverse()

    for i in range(len(nums)//2):
        pair.append([nums[2*i],nums[2*i+1]])
    
    for j in range(len(pair)):
        sum+=min(pair[j])

    return sum

def array_pair_sum(nums):
    #거꾸로 정렬하고 짝수번째 찾는 방법. 생략
    #파이썬에서 숫자는 0부터 시작함을 유의
    pass

def array_pair_sum_pystyle(nums):
    return sum(sorted(nums)[::2])

if __name__=='__main__':
    nums=[1,4,3,2]
    print(array_pair_sum_pystyle(nums))
    