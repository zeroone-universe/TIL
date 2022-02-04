    
def permute(nums):
    result=[]
    
    def dfs(index, poss):
        if len(nums) == len(poss):
            result.append(poss)
            return
        
        for i in nums:
            if i not in poss:
                dfs(index+1,poss.append(i))
                
            
    a=list()
    dfs(0, a)
    
    return result

if __name__ == '__main__':
    nums=[1,2,3]
    print(permute(nums=nums))