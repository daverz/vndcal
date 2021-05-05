import numpy as np
import scipy.signal as sig


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
    sweep_samples = amplitude * sig.chirp(t, hz_start, duration, hz_end,
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


def warble_tone(center,
                amplitude=0.5,  # amplitude of warble tone
                width=1.0 / 3,  # octaves
                period=0.25,  # seconds
                phi=-90,  # sine wave
                sample_rate=48000):
    """Generate a warble tone"""
    # find min and max frequencies for warble tone
    alpha = 2 ** (0.5 * width)
    f0, f1 = center / alpha, center * alpha
    t1 = 0.5 * period
    t = np.arange(t1 * sample_rate) / float(sample_rate)
    # accumulated phase over half period
    phase = 180 * (f0 + f1) * t1
    offset = phi
    while True:
        yield amplitude * sig.chirp(t, f0, t1, f1, phi=offset)
        # make sure we start at the next sample after where we left off
        offset += phase
        # reverse direction
        f0, f1 = f1, f0


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
