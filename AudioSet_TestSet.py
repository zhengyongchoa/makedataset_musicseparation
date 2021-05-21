import numpy as np
import os,sys
import pickle
import shutil
project_dir = os.path.dirname(os.path.abspath(__file__))  # 工程根目
sys.path.append(project_dir)







if __name__ == '__main__':
    # 测试集每个年龄段有M=200条音频
    M= 5
    input_dir = os.path.join(project_dir , 'AudioSet' )
    output_dir = os.path.join(project_dir , 'TestSet')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir )
    indexlist = np.array(range(11, 61))

    indexlist =[58]
    for age in  indexlist:
        age_in_dir = os.path.join(input_dir,str(age))
        names = [na for na in os.listdir(age_in_dir) if na.endswith(".caf")]
        age_out_dir = os.path.join(output_dir,str(age))
        if not os.path.exists(age_out_dir):
            os.makedirs(age_out_dir)

        L =len(names)

        sizelist =[]
        filelist =[]
        for j in names:
            filepath = os.path.join(age_in_dir , j)
            sizelist.append(os.path.getsize(filepath))
            filelist.append(j)
        listmax = np.argsort(sizelist)
        listmax = list(listmax[-M:])

        for k in listmax:
            filepath = os.path.join(age_in_dir ,filelist[k])
            newfile = filelist[k][:-4] + '_test.wav'
            newpath = os.path.join(age_out_dir ,newfile )
            shutil.copy(filepath, newpath)
            # os.remove(filepath)

        print('测试集：年龄{}复制完成！'.format(age))








