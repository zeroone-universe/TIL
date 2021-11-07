import sys
import os
import importlib.util

def get_py_paths(paths):
    py_paths=[]
    if type(paths)==str:
        paths=[paths]

#1. get py paths
    for path in paths:
        for root,dir,files in os.walk(path):
            py_paths+=[os.path.join(root,file) for file in files if os.path.splitext(file)[-1]=='.py']

#2. sort by filenames
    py_paths.sort(key=lambda x: os.path.split(x)[-1])

    return py_paths

path='/home/class/ec2206/submit/HW03/' #HWo3 이름 바꿔주세요
py_paths=get_py_paths(path)


for module_path in py_paths:
    filename=os.path.split(module_path)[1]
    try:
        module_name='maxScore' #함수 이름 바꿔주세요
        spec=importlib.util.spec_from_file_location(module_name, module_path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        #input 바꿔주세요
        input1=[1,3]
        input2=[5,6,7]
        input3=[3,1,5,8]
        input4=[1,3,5,6,2]



        output1=module.maxScore(input1)
        output2=module.maxScore(input2)
        output3=module.maxScore(input3)
        output4=module.maxScore(input4)
        output5=module.maxScore(input5)

        print(f'{filename} : {output1} {output2} {output3} {output4} {output5}')
    except:
        print(f'{filename} : Error!!!')