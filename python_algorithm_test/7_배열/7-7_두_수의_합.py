def two_sum_bf(nums:list, target:int)->list:
    for i in range(len(nums)):
        for j in range(i+1,len(nums)):
            if nums[i]+nums[j]==target:
                return [i,j]
    
def two_sum_in(nums:list, target:int)->list:
    for i, n in enumerate(nums):
        complement= target-n

        if complement in nums[i+1:]:
            return [nums.index(n), nums[i+1:].index(complement)+(i+1)]

def two_sum_dic(nums:list, target:int)->list:
    nums_dic={}

    for i, num in enumerate(nums):
        nums_dic[num]=i

    for i, num in enumerate(nums):
        if target-num in nums_dic and i!=nums_dic[target-num]:
            return(i, nums_dic[target-num])
#리스트의 index를 value로, 리스트의 값을 key로 갖는 딕셔너리 만들기

def two_sum_dic2(nums:list, target:int)->list:
    nums_dic={}
    for i, num in enumerate(nums):
        if target-num in nums_dic and i!=nums_dic[target-num]:
            return(i, nums_dic[target-num])
        nums_dic[num]=i   

if __name__=='__main__':
    nums=[2,2,7,15]
    target=9
    print(two_sum_dic2(nums,target))