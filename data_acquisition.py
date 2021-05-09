from __future__ import print_function

from collections import deque
from itertools import cycle

import numpy as np
import pyaudio


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

        def input_callback(in_data, frame_count, time_info, status):
            if callback:
                data = np.frombuffer(in_data, 'f')
                callback(data)
            return in_data, pyaudio.paContinue

        def output_callback(in_data, frame_count, time_info, status):
            data = next(output_data)
            return data.astype('f').flatten().tobytes(), pyaudio.paContinue

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
            stream.stop_stream()
            stream.close()
        self._streams = []
        if self._paudio:
            self._paudio.terminate()


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
