from __future__ import print_function

from collections import deque
from itertools import cycle
from threading import Thread

import numpy as np
import pyaudio


# import soundcard


class Audio:
    def __init__(self):
        self._streams = []
        self._paudio = None
        self.name = None
        self.sample_rate = None

    def start_acquisition(self,
                          sample_rate=44100,
                          buffer_size=8192,
                          output_data=(),
                          callback=None):

        output_cycle = cycle(output_data.flatten())

        # def paudio_callback(in_data, frame_count, time_info, status):
        #     if callback:
        #         data = np.frombuffer(in_data, 'f')
        #         print('in:', data[:10])
        #         callback(data)
        #     return in_data, pyaudio.paContinue

        def input_callback(in_data, frame_count, time_info, status):
            if callback:
                data = np.frombuffer(in_data, 'f')
                callback(data)
            return in_data, pyaudio.paContinue

        def output_callback(in_data, frame_count, time_info, status):
            data = np.fromiter(output_cycle, 'f', count=2 * frame_count)
            return data.tobytes(), pyaudio.paContinue

        self._paudio = pyaudio.PyAudio()

        # info = self._paudio.get_default_input_device_info()
        # self.name, self.sample_rate = info['name'], info['defaultSampleRate']
        # supported = self._paudio.is_format_supported(32000,
        #                                  input_device=default_info['index'],
        #                                  input_channels=2,
        #                                  input_format=pyaudio.paInt16)
        # print('Supported =', supported)
        input_stream = self._paudio.open(format=pyaudio.paFloat32,
                                         channels=1,
                                         rate=sample_rate,
                                         input=True,
                                         stream_callback=input_callback,
                                         frames_per_buffer=buffer_size)

        output_stream = self._paudio.open(format=pyaudio.paFloat32,
                                          channels=2,
                                          rate=sample_rate,
                                          output=True,
                                          stream_callback=output_callback,
                                          frames_per_buffer=buffer_size)
        print('output latency:', output_stream.get_output_latency())
        print('input latency:', input_stream.get_input_latency())
        self._streams = [output_stream, input_stream]
        for stream in self._streams:
            stream.start_stream()

    def stop_acquisition(self):
        for stream in self._streams:
            stream.start_stream()
            stream.close()
        self._streams = []
        if self._paudio:
            self._paudio.terminate()


class AudioInOut:
    def __init__(self):
        # self._stream = None
        # self._paudio = None
        self.name = None
        self.sample_rate = None
        self._playthread = None
        self.play = False

    def readwrite(self, output_cycle, sample_rate, buffer_size, callback):
        p = pyaudio.PyAudio()

        stream = p.open(format=pyaudio.paFloat32,
                        channels=2,
                        rate=sample_rate,
                        input=True,
                        output=True,
                        frames_per_buffer=buffer_size,
                        # buffer_size=2*buffer_size
                        )

        print("* recording")
        # first buffer always seems to be zeros, so throw it away
        print('read available:', stream.get_read_available())
        for chunk in output_cycle:
            if not self.play:
                break
            stream.write(chunk)
            raw = stream.read(stream.get_read_available())
            if callback:
                data = np.frombuffer(raw, 'f')
                callback(data[::2])

        print("* done")

        stream.stop_stream()
        stream.close()

        p.terminate()

    def start_acquisition(self,
                          sample_rate=48000,
                          buffer_size=9600,
                          output_data=(),
                          output_channel=0,
                          callback=None):
        # copy output data into left or right channel
        stereo_data = np.zeros((2, len(output_data)), 'f')
        stereo_data[output_channel] = output_data
        # interleave channels and break into chunks of buffer_size frames
        chunks = [ary.tobytes() for ary in
                  stereo_data.transpose().reshape(-1, buffer_size * 2)]
        # repeat forever!
        chunk_cycle = cycle(chunks)
        self.play = True
        self._playthread = Thread(target=self.readwrite,
                                  args=(chunk_cycle,
                                        sample_rate,
                                        buffer_size,
                                        callback))
        self._playthread.start()

    def stop_acquisition(self):
        self.play = False
        if self._playthread and self._playthread.is_alive():
            self._playthread.join()


class PlayRecord:
    def __init__(self):
        # self._stream = None
        # self._paudio = None
        self.name = None
        self.sample_rate = None
        self._playthread = None
        self.play = False

    def readwrite(self,
                  output_data,
                  output_channel,
                  sample_rate,
                  buffer_size,
                  callback):

        speaker = soundcard.default_speaker()
        mic = soundcard.default_microphone()

        chunks = output_data.reshape(-1, buffer_size)
        with mic.recorder(samplerate=sample_rate) as recorder, \
                speaker.player(samplerate=sample_rate,
                               channels=output_channel) as player:
            for chunk in cycle(chunks):
                if not self.play:
                    break
                player.play(chunk)
                data = recorder.record(numframes=buffer_size)
                callback(data.flatten())

    def start_acquisition(self,
                          sample_rate=48000,
                          buffer_size=9600,
                          output_data=(),
                          output_channel=0,
                          callback=None):
        self.play = True
        self._playthread = Thread(target=self.readwrite,
                                  args=(output_data,
                                        output_channel,
                                        sample_rate,
                                        buffer_size,
                                        callback))
        self._playthread.start()

        # # handler for full input buffer
        # def paudio_callback(in_data, frame_count, time_info, status):
        #     if callback:
        #         data = np.frombuffer(in_data, 'f')
        #         print('in:', data[:10])
        #         callback(data[input_channel::2])
        #     out_data = chunk_cycle.next()
        #     return out_data, pyaudio.paContinue
        #
        # self._paudio = pyaudio.PyAudio()
        #
        # info = self._paudio.get_default_input_device_info()
        # self.name, self.sample_rate = info['name'], info['defaultSampleRate']
        # # supported = self._paudio.is_format_supported(32000,
        # #                                  input_device=default_info['index'],
        # #                                  input_channels=2,
        # #                                  input_format=pyaudio.paInt16)
        # # print('Supported =', supported)
        # self._stream = self._paudio.open(format=pyaudio.paFloat32,
        #                                  channels=2,
        #                                  rate=sample_rate,
        #                                  input=True,
        #                                  output=True,
        #                                  # stream_callback=paudio_callback,
        #                                  frames_per_buffer=buffer_size)
        # self._stream.start_stream()

    def stop_acquisition(self):
        self.play = False
        if self._playthread and self._playthread.is_alive():
            self._playthread.join()
        # if self._stream:
        #     self._stream.stop_stream()
        #     self._stream.close()
        # if self._paudio:
        #     self._paudio.terminate()


def main():
    import time
    d = deque(maxlen=5)

    def callback(data):
        d.append(data)
        if len(d) < d.maxlen:
            return
        x = np.concatenate(d)

    audio = Audio()
    audio.start_acquisition(sample_rate=48000,
                            buffer_size=48000 // 5,
                            callback=callback)
    time.sleep(5)
    audio.stop_acquisition()


if __name__ == '__main__':
    main()
