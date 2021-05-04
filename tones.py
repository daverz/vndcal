import inspect

import numpy as np
from scipy.signal import chirp


def sp_sweep(rate=48000,
             level=-12.0,  # dBFS
             hz_start=10.0,
             hz_end=240.0,
             duration=2.0,  # seconds
             lead_in_octaves=1.0 / 3,  # fade-in length in octaves
             lead_out_octaves=1.0 / 12):  # fade-out length in octaves
    octave_length = duration / np.log2(hz_end / hz_start)
    lead_in = lead_in_octaves * octave_length
    lead_out = lead_out_octaves * octave_length
    amplitude = 10 ** (level / 20.0)
    sweep_len = int(duration * rate)
    # time steps for the sweep
    t = np.arange(sweep_len) / float(rate)
    # create sweep.  Scipy's chirp() is a cosine sweep, so we start at -90 deg.
    sweep_samples = amplitude * chirp(t, hz_start, duration, hz_end,
                                      method='logarithmic',
                                      phi=-90)
    # compute Blackman tapers
    lead_in_len = int(lead_in * sweep_len)
    lead_out_len = int(lead_out * sweep_len)
    lead_in_taper = np.blackman(2 * lead_in_len - 1)
    lead_out_taper = np.blackman(2 * lead_out_len - 1)
    # apply tapers
    sweep_samples[:lead_in_len] *= lead_in_taper[:lead_in_len]
    sweep_samples[-lead_out_len:] *= lead_out_taper[-lead_out_len:]
    return sweep_samples


# def sweep_cycle(sweep_samples, channels=(0,), chunk_size=9600):
#     """Returns a repeating cycle of stereo frames
#     suitable for passing to pyaudio."""
#     sweep_len = len(sweep_samples)
#     assert sweep_len % chunk_size == 0
#     data = np.zeros((sweep_len, 2), 'f')
#     for ch in channels:
#         data[:, ch] = sweep_samples
#     chunk_bytes = [chunk.tobytes() for chunk in
#                    data.reshape(-1, 2 * chunk_size)]
#     return cycle(chunk_bytes)


def make_chunk_generator(data_source, channels=(0,), chunk_size=8192):
    parts = []
    while True:
        while sum(len(p) for p in parts) < chunk_size:
            try:  # assume it's a generator
                next_part = next(data_source)
            except TypeError:  # assume it's an array
                next_part = data_source
            parts.append(next_part)
        joined = np.concatenate(parts)
        chunk = joined[:chunk_size]
        remainder = joined[chunk_size:]
        parts = []
        if len(remainder):
            parts.append(remainder)
        stereo = np.zeros((2, chunk_size))
        for channel in channels:
            stereo[channel] = chunk
        yield stereo.transpose()


def test_chunk_generator():
    data_source = sp_sweep()
    chunk_gen = make_chunk_generator(data_source=data_source)
    for _ in range(5):
        chunk = next(chunk_gen)
        print(chunk.shape)


if __name__ == '__main__':
    test_chunk_generator()
