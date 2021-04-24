import math
import numpy as np
from scipy.signal import butter, lfilter, lfiltic, freqz, firwin2, periodogram, \
    welch, bilinear
from scipy.io import wavfile

BANDS_PER_OCTAVE = 4
SAMPLE_RATE = 48000
LOWPASS = 0.8  # of Nyquist
DECIMATORS = (5, 4, 3, 2)
# DECIMATION = (5, 5, 4)
# DECIMATION = (5, 4, 3)  # decimate down to frequencies of interest
DECIMATION = (4, 4, 4)  # decimate down to frequencies of interest
BANDS = 11
NYQUIST = 0.5 * SAMPLE_RATE / np.prod(DECIMATION)
TOP_BAND_CENTER = 120
TOP_BAND = 120 / NYQUIST
ORDER = 8

# filter_func = filtfilt

_octave_band_cache = {}


def mean_power(x):
    return np.dot(x, x) / len(x)


def db_power(x):
    return 10 * np.log10(mean_power(x))


def make_filter_func(b, a):
    zi = lfiltic(b, a, [])

    def filter_func(x):
        y, zi[:] = lfilter(b, a, x, zi=zi)
        return y

    return filter_func


# def make_decimator(decimation=DECIMATION):
#     filters = []
#     coeffs = {}
#     for factor in set(decimation):
#         coeffs[factor] = cheby1(8, 0.05, 0.8 / factor)
#     for factor in decimation:
#         filters.append(make_filter_func(*coeffs[factor]))
#
#     def decimator(x):
#         y = x
#         for filt, q in zip(filters, decimation):
#             y = filt(y)[::q]
#         return y
#
#     return decimator
#
#
# def make_filters(critical_freqs=CRITICAL_FREQS, nyquist=NYQUIST):
#     filters = []
#     c = 2 ** (1. / 6)
#     for center, (lower, upper) in zip(CENTER_FREQS, critical_freqs):
#         # wn = lower/nyquist, upper/nyquist
#         wn = center / c / nyquist, center * c / nyquist
#         # b, a = octave_filter_coefficients(center, lower, upper, nyquist)
#         b, a = butter(2, wn, btype='bandpass')
#         print(center, filter_enbw(b, a))
#         # w, h = freqz(b, a)  # returns the frequencies and transfer function
#         # pwr = h.real ** 2 + h.imag ** 2
#         # print('overall', center, pwr.sum()**.5)
#         filters.append(make_filter_func(b, a))
#     return filters


def filter_enbw(b, a):
    """Equivalent noise bandwidth of the filter, scaled by Nyquist frequency.
    This is the integral of the normalized power response over frequency."""
    w, h = freqz(b, a)  # returns the frequencies and transfer function
    # bl, al = butter(ORDER, LOWPASS, btype='lowpass')
    # _, hl = freqz(bl, al)
    # h *= hl
    pwr = h.real ** 2 + h.imag ** 2
    return pwr.sum() * w[1] / np.pi


def octave_filter_coefficients(fc,
                               bands_per_octave,
                               order):
    """Given center frequencies the top octave bands
    normalized by the Nyquist rate, and the filter name and order,
    produce IIR filter coefficients."""
    alpha = 2.0 ** (0.5 / bands_per_octave)
    lower = fc / alpha
    upper = fc * alpha
    # band Q
    qr = fc / (upper - lower)
    b, a = butter(order, [lower, upper], btype='bandpass')
    # We adjust the band Q so that band power matches an ideal filter...
    # Compute the equivalent noise bandwidth for a
    # "naive" choice of band edges.
    # For high-order filters this correction is negligible
    enbw = filter_enbw(b, a)
    qd = enbw / fc * qr * qr
    # Solve the quadratic for the adjusted multiplication factor.
    alpha = 0.5 * (1 + (1 + 4 * qd ** 2) ** .5) / qd
    adjusted_edges = fc / alpha, fc * alpha
    b, a = butter(order, adjusted_edges, btype='bandpass')
    return b, a


# def make_filter(b, a):
#     def filter_(x):
#         return filter_func(b, a, x)
#
#     return filter_


def make_decimator(q, order=8):
    b, a = butter(order, LOWPASS / q, btype='lowpass')
    _filter_func = make_filter_func(b, a)

    def decimate(x):
        y = _filter_func(x)
        return y[::q]

    return decimate


def make_filter_bank(top,
                     bands_per_octave,
                     nbands,
                     order):
    bank = []
    assert 0 < nbands <= bands_per_octave
    exp = 2.0 ** (1.0 / bands_per_octave)
    for i in range(0, nbands):
        fc = top / exp ** i
        b, a = octave_filter_coefficients(fc,
                                          bands_per_octave=bands_per_octave,
                                          order=order)
        filter_ = make_filter_func(b, a)
        bank.append(filter_)
    return bank


# # init decimators
# _decimator_map = {i: make_decimator(i) for i in DECIMATORS}
#
# # init filter bank
# _filter_bank = make_filter_bank(TOP_BAND, BANDS_PER_OCTAVE, ORDER)


# def decimate(x, *decimation):
#     y = x
#     for q in decimation:
#         y = _decimator_map[q](y)
#     return y


# def apply_octave_banks(x, decimation=DECIMATION, nbands=BANDS):
#     # decimate to effective rate
#     y = decimate(x, *decimation)
#     i = 0
#     db_values = []
#     overall = db_power(x)
#     db_values.append(overall)
#     while len(y):
#         for f in _filter_bank:
#             band = f(y)
#             db_pwr = db_power(band)
#             db_values.append(db_pwr)
#             i += 1
#             if i >= nbands:
#                 return np.array(db_values[::-1])
#         y = decimate(y, 2)


def c_weighting(sample_rate):
    """Return digital filter coefficients
    for C weighting for given sample_rate"""
    f1 = 20.598997
    f4 = 12194.217
    C1000 = 0.0619
    numerators = [(2 * np.pi * f4) ** 2 * (10 ** (C1000 / 20)), 0, 0]
    denominators = np.convolve([1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2],
                [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2])
    b, a = bilinear(numerators, denominators, sample_rate)
    return b, a


def mic_correction(mic_response, nyquist):
    """Turn a mic frequency response into a fir filter to flatten response"""
    top_freq = LOWPASS*nyquist
    useful_response_index = mic_response[:, 0].searchsorted(top_freq)
    useful_response = mic_response[:useful_response_index]
    freqs = useful_response[:, 0] / nyquist
    gains = 10 ** (-useful_response[:, 1] / 20)
    freqs = np.r_[0, freqs, 1.0]
    gains = np.r_[0, gains, 0]
    print(gains[-10:])
    numtaps = 2**int(np.log2(len(freqs))) - 1
    taps = firwin2(numtaps, freqs, gains)
    return taps, 1


class OctaveFilter(object):
    def __init__(self,
                 bands_per_octave=BANDS_PER_OCTAVE,
                 top_center_frequency=TOP_BAND_CENTER,
                 nbands=BANDS,
                 input_rate=SAMPLE_RATE,
                 decimation=DECIMATION,
                 order=ORDER,
                 use_c_weighting=False,
                 mic_response=None):
        self._decimators = [make_decimator(q, order) for q in decimation]
        n_octaves = int(math.ceil(nbands / float(bands_per_octave)))
        self._octave_decimators = []
        # self._decimation = decimation
        self._nbands = nbands
        self.nyquist = 0.25 * input_rate / np.prod(decimation)
        self._pre_band_filters = []
        if use_c_weighting:
            b, a = c_weighting(self.nyquist * 4)
            self._pre_band_filters.append(make_filter_func(b, a))
        if mic_response is not None:
            b, a = mic_correction(mic_response, self.nyquist * 2)
            self._pre_band_filters.append(make_filter_func(b, a))
        top = top_center_frequency / self.nyquist
        self._input_rate = input_rate
        self._octave_band_filters = []
        # octave_band_count = [bands_per_octave] * (nbands//bands_per_octave)
        # if nbands % bands_per_octave:
        #     octave_band_count += [nbands % bands_per_octave]
        i = nbands
        for _ in range(n_octaves):
            decimator = make_decimator(2, order)
            self._octave_decimators.append(decimator)
            count = min(bands_per_octave, i)
            print(i, n_octaves, count)
            fb = make_filter_bank(top,
                                  bands_per_octave,
                                  count,
                                  order)
            self._octave_band_filters.append(fb)
            i -= bands_per_octave
        self._amplitude_corrections = np.zeros(nbands + 1, dtype=float)

    def set_decimation(self, decimation):
        self._decimation = decimation


    def __call__(self, x):
        band_power = []
        y = x
        for decimator in self._decimators:
            y = decimator(y)
        for f in self._pre_band_filters:
            y = f(y)
        y0 = y
        for decimator, filter_bank in zip(self._octave_decimators,
                                          self._octave_band_filters):
            y = decimator(y)
            for f in filter_bank:
                band = f(y)
                band_power.append(mean_power(band))
        band_power = band_power[::-1]
        # band_power.append(mean_power(y0))
        band_power = np.array(band_power)
        return band_power


def main():
    from time import clock
    # from scipy.signal import chirp, tukey
    # t = np.arange(SAMPLE_RATE) / float(SAMPLE_RATE)
    # data = chirp(t, 10., 1.0, 160., method='logarithmic')
    # data2 = chirp(t, 20., 1.0, 320., method='logarithmic')
    # data3 = chirp(t, 40., 1.0, 640., method='logarithmic')
    # # data = data * tukey(len(t), alpha=1/32.)
    # # for f in (20, 24, 30, 36, 42, 50, 60, 72, 84, 100, 120):
    # #     data = 0.5 * np.sin(2*np.pi*f*t)
    # # rms = 10 * np.log10(np.dot(data, data) / len(data))
    # # print(rms)
    # # print(data.shape)
    # octave_filter = OctaveFilter(decimation=(5, 3, 2), top_center_frequency=480)
    # octave_filter.compute_corrections(data3)
    # print(octave_filter._amplitude_corrections)
    # octave_filter = OctaveFilter(decimation=(5, 4, 3), top_center_frequency=240)
    # # octave_filter.set_decimation((5, 4, 3))
    # octave_filter.compute_corrections(data2)
    # print(octave_filter._amplitude_corrections)
    # octave_filter = OctaveFilter()
    # # octave_filter.set_decimation((5, 4, 3))
    # octave_filter.compute_corrections(data)
    # print(octave_filter._amplitude_corrections)
    t0 = clock()
    reps = 100
    for _ in range(reps):
        make_filter_bank(TOP_BAND, BANDS_PER_OCTAVE, ORDER)
    # values = octave_filter(data)
    dt = clock() - t0
    print(dt / reps)
    # print(values)


if __name__ == '__main__':
    main()
