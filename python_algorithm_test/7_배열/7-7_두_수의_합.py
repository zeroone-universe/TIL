def find_sum_bf(nums:list, target:int)->list:
    for i in range(len(nums)):
        for j in range(i+1,len(nums)):
            if nums[i]+nums[j]==target:
                return [i,j]
    return None


nums=[2,2,7,15]
target=9
print(find_sum(nums,target))