import os
import time
import threading
from glob import glob

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

VANDERTONE_DIR = os.path.expanduser('~/Music/vandertones')
DURATION = 3.0
BAND_ORDER = [9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
              1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

class MockAudio:

    def __init__(self):
        self.sample_rate = None
        self.buffer_size = None
        self.callback = None
        self.samples = None
        self.stop = False
        self.offset = 0
        self.thread = None
        super().__init__()

    def start_acquisition(self,
                          sample_rate=44100,
                          buffer_size=8192,
                          callback=None):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.callback = callback
        all_samples = []
        for band in BAND_ORDER:
            wav = glob(os.path.join(VANDERTONE_DIR, f'{band:02d}*.wav'))
            if not wav:
                raise ValueError(f'No wav file found for band {band}')
            rate, samples = wavfile.read(wav[0])
            samples = samples[:, 0] / 32768.0
            if rate != sample_rate:
                samples = resample_poly(samples, up=sample_rate, down=rate)
            n = int(sample_rate * DURATION)
            all_samples.append(samples[:n])
        self.samples = np.concatenate(all_samples)
        self.thread = threading.Thread(target=self.acquire)
        self.thread.start()

    def acquire(self):
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
        if self.thread:
            self.thread.join()
        self.stop = False
        self.thread = None
