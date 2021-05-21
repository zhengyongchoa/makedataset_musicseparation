# step1：检索伴奏和背景音两文件内文件是否一一对应
# step2: vocal加载混响，
# 本脚本可处理44k 16k音频
# MakeDatasetSeparate.py 放在 分离语料集里
# 语料集和 混响RIR集合在一个路径下
import os
import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
import random
from scipy import signal
import argparse

root_dir = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)



def transfer(path_wav, path_mp3, files):
    """将wav转换成mp3格式
    参数：wav文件夹路径，mp3文件夹路径，files 文件夹下的文件名称
    """
    cmd_str = "ffmpeg -i " +  os.path.join(path_wav, files) + " -f mp3 -acodec libmp3lame -y " + os.path.join(path_mp3,  files[0:-4] + ".mp3")
    # print(cmd_str)
    os.system(cmd_str)
    os.remove(os.path.join(path_wav, files))



def reverb(win , Vwave):
    sig = Vwave
    filtered = signal.fftconvolve(sig , win ,mode ='full')
    filtered = filtered[0: len(sig)]
    return filtered

def writemp3(Newname , NewVOCpath , NewBGMpath , MIXpath , Vwave ,BGMwave, Mixwave , sample ):
    """
    音频文件名称， 【干声 ，背景声，混合声 的路径 ，变量】 采样率 ，
    """
    librosa.output.write_wav(os.path.join(NewVOCpath, Newname), Vwave, sample)  # create mp3
    librosa.output.write_wav(os.path.join(NewBGMpath, Newname), BGMwave, sample)
    librosa.output.write_wav(os.path.join(MIXpath, Newname), Mixwave, sample)

    transfer(NewVOCpath, NewVOCpath, Newname)
    transfer(NewBGMpath, NewBGMpath, Newname)
    transfer(MIXpath, MIXpath, Newname)

def AddDirectSound(rir):
    maxvalue = max(abs(rir))
    if maxvalue == 0 :
        print("line 66:  RIR sequence  is O !")
    else:
        cha = np.floor(0.5 / (maxvalue + 1E-7))
        if cha > 20:
            mi = random.randint(2,10)
            mixrate = maxvalue * mi
        else:
            mi = random.randint(1, cha)
            mixrate = maxvalue * mi

    rir[0] = mixrate

    # plt.plot(rir)
    # plt.show()

    return rir

def makedata(args):
    sample = args.sample
    path = args.path
    Newpath = args.Newpath
    segment = args.segment

    VOCpath = os.path.join(root_dir, path, args.vocal_data)
    BGMpath = os.path.join(root_dir, path, args.acc_data)
    if os.path.exists(os.path.join(VOCpath, '.DS_Store')):
        os.remove(os.path.join(VOCpath, '.DS_Store'))
    if os.path.exists(os.path.join(BGMpath, '.DS_Store')):
        os.remove(os.path.join(BGMpath, '.DS_Store'))

    NewVOCpath = os.path.join(root_dir, Newpath, 's2')
    NewBGMpath = os.path.join(root_dir, Newpath, 's1')
    MIXpath = os.path.join(root_dir, Newpath, 'mix')
    if not os.path.exists(os.path.join(root_dir, Newpath)):
        os.mkdir( os.path.join(root_dir, Newpath) )
    if not os.path.exists(MIXpath):
        os.mkdir(MIXpath)
    if not os.path.exists(NewVOCpath):
        os.mkdir(NewVOCpath)
    if not os.path.exists(NewBGMpath):
        os.mkdir(NewBGMpath)


    # step1：检测语料文件夹下的干声和背景声文件夹中mp3文件数目是否一致

    VOCfileList =[]
    BGMpathList = []
    for _ ,_ , files in os.walk(VOCpath):
        for file in files:
            if file.endswith('.mp3') and (file[0] != '.'):
                VOCfileList.append(file)
    VOCfileList.sort()

    for _,_,files in os.walk(BGMpath):
        for file in files:
            if file.endswith('.mp3') and (file[0] != '.'):
                BGMpathList.append(file)
    BGMpathList.sort()



    if len(VOCfileList) == len(BGMpathList):
        for VOCfile in  VOCfileList:
            filenum =0
            name = VOCfile.split('-')
            VOC2BGM = name[0]
            for i in range(1, len(name)-1):
                VOC2BGM = VOC2BGM + '-'+ name[i]
            VOC2BGM = VOC2BGM + '-m.mp3'
            if VOC2BGM in BGMpathList:
                filenum = filenum+1
            else:
                print("bgm不包含vocal ！\n")
                print('vocal: ' , VOCfile)
                exit()
    else:
        print('voc ',len(VOCfileList))
        print('BGM ' , len(BGMpathList))
        print("两文件夹内文件数目不对应！")
        exit()


    # step2:干声做混响， 重采样输出

    numflag = 0
    for VOCfile  in  VOCfileList:
        numflag =numflag+1
        print('注意,运行到第 {} 条音频: {}!'.format(numflag , VOCfile))
        name = VOCfile.split('-')
        Newname = name[0]
        for i in range(1, len(name) - 1):
            Newname = Newname + '-' + name[i]
        VOC2BGM = Newname + '-m.mp3'
        # Newname = Newname + '.wav'
        Vwave ,fs1 = librosa.load( os.path.join(VOCpath ,VOCfile ), 44100 )
        BGMwave ,fs2 = librosa.load(os.path.join(BGMpath, VOC2BGM ) , 44100)



        if sample == 44100 :
            RIRdirwav44k = librosa.resample(RIRdirwav , 16000, 44100)
            VwaveRIR =  reverb(RIRdirwav44k , Vwave )
            lengthWave = min(len(Vwave), len(BGMwave))
            # Mixwave = VwaveRIR[0:lengthWave]/max(abs(VwaveRIR[0:lengthWave])) * 0.9  + BGMwave[0:lengthWave]/ max(abs(BGMwave[0:lengthWave])) *0.6
            Mixwave = VwaveRIR[0:lengthWave] / (max(abs(VwaveRIR[0:lengthWave])) + 1E-7) * 0.9 + \
                      BGMwave[0:lengthWave] / (max(abs(BGMwave[0:lengthWave])) + 1E-7) * 0.6


            segment_len = int(segment * sample)
            numframe = int(np.floor(lengthWave / segment_len))
            if numframe < 1:
                print('音频长度小于片段定制长度！')
            else:
                tlist = np.arange(numframe+1) * segment_len
                for j in range(numframe):
                    Newname_j = Newname + '-'+ str(j+1) + '.wav'
                    print('正在处理音频: {} !'.format(Newname_j))
                    ind1 = tlist[j]
                    ind2 = tlist[j+1]

                    Vwaveframe = Vwave[ind1:ind2]
                    BGMwaveframe = BGMwave[ind1:ind2]
                    Mixwaveframe  = Mixwave[ind1:ind2]

                    writemp3(Newname_j, NewVOCpath, NewBGMpath, MIXpath, Vwaveframe, BGMwaveframe , Mixwaveframe, sample)


        else:
            Vwave16k = librosa.resample(Vwave , 44100 , 16000 )
            BGMwave16k = librosa.resample(BGMwave , 44100 , 16000 )
            # VwaveRIR = reverb( RIRdirwav , Vwave16k)
            VwaveRIR = Vwave16k
            lengthWave = min(len(BGMwave16k) , len(VwaveRIR) )
            # Mixwave= VwaveRIR[0: lengthWave]/max(abs(VwaveRIR[0: lengthWave]))*0.9 +  BGMwave16k[0: lengthWave]/max(abs(BGMwave16k[0: lengthWave]))*0.6
            Mixwave = VwaveRIR[0: lengthWave] / (max(abs(VwaveRIR[0: lengthWave])) + 1E-7) * 0.9 + \
                      BGMwave16k[0: lengthWave] / (max(abs(BGMwave16k[0: lengthWave])) + 1E-7) * 0.6

            segment_len = int(segment * sample)
            numframe = int(np.floor(lengthWave / segment_len))
            if numframe < 1:
                print('音频长度小于片段定制长度！')
            else:
                tlist = np.arange(numframe + 1) * segment_len
                for j in range(numframe):
                    Newname_j = Newname + '-' + str(j + 1) + '.wav'
                    print('正在处理音频: {} !'.format(Newname_j))
                    ind1 = tlist[j]
                    ind2 = tlist[j + 1]

                    Vwaveframe = Vwave16k[ind1:ind2]
                    BGMwaveframe = BGMwave16k[ind1:ind2]
                    Mixwaveframe = Mixwave[ind1:ind2]

                    writemp3(Newname_j, NewVOCpath, NewBGMpath, MIXpath, Vwaveframe, BGMwaveframe, Mixwaveframe, sample)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("将语料制作成数据集 ")
    parser.add_argument('--path',type=str, default= '声伴分离训练',help = '干声和背景声所在的语料文件夹名称')
    parser.add_argument('--Newpath', type=str, default= 'train5_set' , help=' 新生成的数据集所在的文件夹名称')
    parser.add_argument('--vocal_data', type=str, default= 'vocal_singer' , help='语料文件夹下干声文件夹的名称')
    parser.add_argument('--acc_data', type=str, default= 'acc_singer' , help='语料文件夹下背景声文件夹的名称')
    parser.add_argument('--sample', type= int, default= 16000 , help='目标数据集的音频采样率，可以选择"44100" ， "16000"')
    parser.add_argument('--segment', type=int, default= 20.01 , help='从音频中以20s为一片段，进行切片')

    args = parser.parse_args()
    print(args)
    makedata(args)
    print(" 程序运行完毕 ！")


































