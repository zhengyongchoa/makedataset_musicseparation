# -*- coding:utf-8 -*-
# 使用 spleeter 将音频批量分离'''
# 代码必须放在PYTHON_ALL_CODE 的spleet下面
# by zyc
import numpy as np
import os, sys
from scipy.io import wavfile
import shutil
import collections
import contextlib
import webrtcvad
import wave
# project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 工程根目录
# sys.path.append(project_dir)
import fnmatch
from third_party.spleeter.spleeter.__main__ import spleeter_2stems, spleeter_4stems

def transfer(path_caf, path_wav, files):
    """将caf转换成wav格式
    参数：caf文件夹路径，wav文件夹路径，caf文件夹下的文件名称
    """
    cmd_str = "ffmpeg -y -i " + os.path.join(path_caf, files) + " -acodec pcm_s16le -ac 1 -ar 16000 " + \
                os.path.join(path_wav, files[0:-4] +'_16k'+ ".wav" )
    # print(cmd_str)
    os.system(cmd_str)


def read_wave(path):
    """读取.wav文件。
     参数：采取路径，
     返回：（PCM音频数据，采样率）。
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """写一个.wav文件。
     参数：获取路径，PCM音频数据和采样率。
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """音频的帧数据。
    """
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """从PCM音频数据生成音频帧。
    参数：帧长（毫秒），音频（pcm），采样率
    Yield出每帧音频
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        #sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                #sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                #sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []

    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])

def Separate(caf_dir , test_output_dir ,age):

    caf_list = os.listdir(caf_dir)
    num = 0
    for caf_file in fnmatch.filter(caf_list, '*.caf'):
        num = num + 1
        print('年龄{}岁，第{}条音频进行分离'.format(age ,num))
        # 将caf转成wav
        tmppath = os.path.join(test_output_dir, 'caf2wav')
        if not os.path.exists(tmppath):
            os.makedirs(tmppath)
        wav_file = os.path.join(test_output_dir, 'caf2wav/caf2wav.wav')
        if 0:
            os.system('ffmpeg -y -i ' + os.path.join(caf_dir, caf_file) + ' -ar 44100 -ac 1 ' + wav_file)
        else:
            cmd = 'python -m third_party.spleeter.spleeter separate -i ' + wav_file + ' -o ' + test_output_dir + ' -p spleeter:2stems'
            print(cmd)
            os.system(cmd)
        # 截取长度大于60s
        sample_rate, audio = wavfile.read(wav_file)
        if len(audio) > sample_rate * 60:
            wavfile.write(wav_file, sample_rate, audio[0:sample_rate * 60])

        # 运行 spleeter 分离人声和伴奏
        wav_sep_file = os.path.join(test_output_dir, 'caf2wav/vocals.wav')
        wav_sep_file_16k = os.path.join(test_output_dir, 'caf2wav/vocals_16k.wav')
        if os.path.exists(wav_sep_file):
            os.system('rm -f ' + wav_sep_file)
        if os.path.exists(wav_sep_file_16k):
            os.system('rm -f ' + wav_sep_file_16k)
        os.system(
            'python -m third_party.spleeter.spleeter separate -i ' + wav_file +
            ' -o ' + test_output_dir + ' -p spleeter:2stems')

        # 将 vocals.wav 转成16k
        transfer(tmppath, tmppath, 'vocals.wav')

        try:
            audio, sample_rate = read_wave(wav_sep_file_16k)
            vad = webrtcvad.Vad(3)  # 侵略性 可选1 2 3
            frames = frame_generator(30, audio, sample_rate)
            frames = list(frames)
            segments = vad_collector(sample_rate, 30, 300, vad, frames)
            print(segments)
            seg = []
            seg_num = 0
            for j, segment in enumerate(segments):
                seg_num = seg_num + 1
                seg.append(segment)
            seg = b"".join(seg)
            path = os.path.join(test_output_dir, str(age) + '_' + caf_file)
            write_wave(path, seg, sample_rate)

        except:
            print('文件转换错误')
            continue

    print('该岁数运行完毕！')
    shutil.rmtree(tmppath)

if __name__ == '__main__':

    # Docinput_dir = os.path.join(project_dir, 'signal_processors/age_classifier/wgetSound')
    # Docoutput_dir = os.path.join(project_dir, 'signal_processors/age_classifier/AudioSet')
    Docinput_dir = '/Users/zyc/Desktop/PYTHON_ALL_CODE/signal_processors/age_classifier/wgetSound'
    Docoutput_dir = '/Users/zyc/Desktop/PYTHON_ALL_CODE/signal_processors/age_classifier/AudioSet'
    index = 1  # 选 【1 ，2 ，3 ，4 ， 5】
    indexlist = np.array(range(11,21)) + (index - 1) * 10

    indexlist=[58]
    for age in  indexlist:
        print('开始分离年龄:{}岁的音频'.format(age))
        input_dir = os.path.join(Docinput_dir , str(age))
        output_dir = os.path.join(Docoutput_dir ,str(age))
        Separate(input_dir, output_dir,age)

    print('the program is over！')


