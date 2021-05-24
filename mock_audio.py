import os
import time
import threading

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

WAVFILE = os.path.expanduser('~/Music/vandertones/04 - Track04.mp3.wav')


class MockAudio(threading.Thread):

    def __init__(self):
        self.sample_rate = None
        self.buffer_size = None
        self.callback = None
        self.samples = None
        self.stop = False
        self.offset = 0
        super().__init__()

    def start_acquisition(self,
                          sample_rate=44100,
                          buffer_size=8192,
                          callback=None):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.callback = callback
        rate, samples = wavfile.read(WAVFILE)
        print(rate, sample_rate, samples.shape, samples[:10])
        samples = samples[:, 0] / 32768.0
        if rate != sample_rate:
            samples = resample_poly(samples, up=sample_rate, down=rate)
        print(samples.shape, samples.dtype)
        self.samples = samples
        self.start()

    def run(self):
        while not self.stop:
            time.sleep(self.buffer_size/self.sample_rate)
            chunk = self.samples[self.offset:self.offset+self.buffer_size]
            self.offset += self.buffer_size
            self.offset %= len(self.samples)
            if len(chunk) < self.buffer_size:
                chunk = np.concatenate((chunk, self.samples[:self.offset]))
            if self.callback:
                self.callback(chunk)

    def stop_acquisition(self):
        self.stop = True
        self.join()
