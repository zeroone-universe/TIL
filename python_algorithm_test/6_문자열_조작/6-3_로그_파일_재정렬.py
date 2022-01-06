def reorderLogFiles(logs: list) -> list:
    digit_log_list=[]
    letter_log_list=[]
    for log in logs:
        if log.split()[1].isdigit():
            digit_log_list.append(log)
        else :
            letter_log_list.append(log)
                
    letter_log_list.sort(key=lambda x: (x.split()[1:], x.split()[0]))
    ans=letter_log_list+digit_log_list
    return ans
        



if __name__=='__main__':
    logs=['dig1 8 1 5 1 ','let1 art can', 'dig2 3 6', 'let own kilt dig', 'let3 art zero']
    result=reorderLogFiles(logs)
    print(result)

