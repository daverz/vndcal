from __future__ import print_function
import os
import re
import time
from collections import deque
from itertools import cycle

import numpy as np
from scipy.io import wavfile
import wx
import wx.lib.plot as plot

from data_acquisition import Audio
from filterbank import OctaveFilter
import tones

SINE_TONE_FREQUENCY = 1000.0

MIC_CALIBRATION_FILE = os.path.expanduser('~/Documents/REW/7032857.txt')
TARGET_FRACTION = 0.30
VOLUME_ADJUSTMENT_BANDS = [8, 9, 10]  # bands 9, 10 and 11
RMS_CORRECTION = 3.01
SNAPSHOT_DIR = os.path.expanduser('~/Desktop')
VOLUME_ADJUSTMENT_ORDER = 3  # fourth lowest band
TOP_FREQ = 120  # Hz
DECIMATION_EQ = 5, 4, 3  # decimation steps based on 48000 samples/sec
# DECIMATION_EQ = 4, 4, 4
DECIMATION_BASS = 5, 3, 2
SWEEP_START = TOP_FREQ / 12
SWEEP_STOP = TOP_FREQ * 2
SWEEP_DURATION = 2.0  # seconds
# DECIMATION_BASS = 4, 4, 4
# CAL_AUDIO_FILE_EQ = '/Users/dcook/Documents/Rew/vanderpink/Pink_PN_32768_10_300_48k_16-bit_L.wav'
# CAL_AUDIO_FILE_EQ = 'chirps/01_1_sec_log_chirp_10-160Hz_left.wav'
CAL_AUDIO_FILE_EQ = 'chirps/LogSweep_10_240_44.1k_PCM16_L.wav'
CAL_AUDIO_FILE_BASS_LEVEL = 'chirps/LogSweep_20-320Hz_44.1_stereo.wav'

BARWIDTH = 0.07
# LEVEL_BAR_POSITION = 150
# RMS_LEVEL = 10 * math.log10(2)
DB_REFERENCE = 70.0
SAMPLE_RATE = 48000
# SAMPLE_RATE = 44100
# NOVERLAPS = 16
UPDATE_RATE = 8
SAMPLE_FIFO_SIZE = 8
LEVEL_FIFO_SIZE = 4
MEASUREMENT_THRESHOLD_DB = 0.05
BLOCK_SIZE = SAMPLE_RATE // UPDATE_RATE
# BLOCK_SIZE = 32768 // 4
# BLOCK_SIZE = 2 ** 14
# DURATION = 2.4
# CHIRP_LEN = DURATION * SAMPLE_RATE
# SAMPLE_FIFO_SIZE = int(math.ceil(DURATION * SAMPLE_RATE / float(BLOCK_SIZE))) + 10
# CENTER_FREQUENCIES = [TOP_FREQ / 2.0 ** (0.25 * i) for i in range(10, -1, -1)]
CENTER_FREQUENCIES = (20, 24, 30, 36, 42, 50, 60, 72, 84, 100, 120)
BANDS = list(range(len(CENTER_FREQUENCIES)))

LEFT_EDGES = [f / 2.0 ** (1. / 8) for f in CENTER_FREQUENCIES]
LEFT_EDGES.append(CENTER_FREQUENCIES[-1] * 2 ** (1. / 8))
# CENTER_FREQUENCIES += [LEVEL_BAR_POSITION]
# CENTER_NAMES_EQ = ('20', '24', '30', '36', '42', '50',
#                    '60', '72', '84', '100', '120')
CENTER_NAMES_EQ = ["%.0f" % f for f in CENTER_FREQUENCIES]
CENTER_NAMES_BASS = ('42', '50', '60', '72', '84', '100', '120',
                     '140', '170', '200', '240')
# XSPEC = list(zip(CENTER_FREQUENCIES, CENTER_NAMES))
# print(XSPEC)
YAXIS_LIMITS = (50, 95)
XAXIS_LIMITS = (19, 133)
VOL_ADJ_FORMAT = 'Vol Adj: {:=+4.1f} \N{SQUARE DB}'
XLABEL = u'Frequency (\N{SQUARE HZ})'
YLABEL = u'Sound Pressure Level (\N{SQUARE DB})'
# TITLE = u'\N{vulgar fraction one quarter} Octave Spectrum'
TITLE = u'11-Band Subwoofer EQ'
RESPONSE_FORMAT = u'Response: \N{Plus-Minus Sign}{:4.1f} \N{SQUARE DB}'

mode_map = {
    'vol_adj': {'frequencies': CENTER_FREQUENCIES[-3:],
                'decimation': DECIMATION_EQ,
                'edges': LEFT_EDGES,
                'names': CENTER_NAMES_EQ,
                'xlimits': XAXIS_LIMITS,
                'sweep_range': (SWEEP_START, SWEEP_STOP),
                'title': TITLE,
                },
    'eq': {
        'frequencies': CENTER_FREQUENCIES,
        'decimation': DECIMATION_EQ,
        'edges': LEFT_EDGES,
        'names': CENTER_NAMES_EQ,
        'xlimits': XAXIS_LIMITS,
        'sweep_range': (SWEEP_START, SWEEP_STOP),
        'title': TITLE,
    },
    'bass_level': {
        'frequencies': [2 * f for f in CENTER_FREQUENCIES],
        'decimation': DECIMATION_BASS,
        'edges': [2 * e for e in LEFT_EDGES],
        'names': CENTER_NAMES_BASS,
        'xlimits': (2 * XAXIS_LIMITS[0], 2 * XAXIS_LIMITS[1]),
        'sweep_range': (2 * SWEEP_START, 2 * SWEEP_STOP),
        'title': 'Bass Level Adjustment'
    }
}

channel_text = {(0,): 'Left Speaker', (1,): 'Right Speaker',
                (0, 1): 'Both Speakers'}

cal_steps = [
    {'mode': 'spl',
     'channels': [0],
     'signal': 'sine_tone',
     'frequency': SINE_TONE_FREQUENCY},
    {'mode': 'measure',
     'channels': [0],
     'signal': 'warble_tone',
     'bands': BANDS[-3:]},
    {'mode': 'measure',
     'channels': [0],
     'signal': 'warble_tone',
     'bands': BANDS}
]

cal_steps += [{'mode': 'measure',
               'channels': [0],
               'signal': 'warble_tone',
               'frequencies': [f]}
              for f in CENTER_FREQUENCIES]

# now add right channel steps
for step in cal_steps[1:]:
    right_step = step.copy()
    right_step['channels'] = [1]
    cal_steps.append(right_step)

cal_step_cycle = cycle(cal_steps)

sens_regex = re.compile('Sens.*=(.+)dB')


def c_weighting(
        frequencies):  # from http://en.wikipedia.org/wiki/A-weighting#C
    f = np.array(frequencies)
    r_c = 12200. ** 2 * f ** 2 / (
            (f ** 2 + 20.6 ** 2) * (f ** 2 + 12200. ** 2))
    c = 0.06 + 20 * np.log10(r_c)
    return c


class PolyText(plot.PolyPoints):
    _attributes = {'foreground': 'blue',
                   # 'background':
                   'font': wx.SWISS_FONT,
                   'textList': (),
                   'adjust': (-0.5, -1.0)
                   }

    def __init__(self, points, **attr):
        plot.PolyPoints.__init__(self, points, attr)

    def draw(self, dc, printerScale, coord=None):
        font = self.attributes['font']
        # font.SetPointSize(20)
        text_list = self.attributes['textList']
        foreground = self.attributes['foreground']
        x_adjust, y_adjust = self.attributes['adjust']
        if foreground and not isinstance(foreground, wx.Colour):
            foreground = wx.Colour(foreground)
        if foreground:
            foregrounds = [foreground] * len(text_list)
        else:
            foregrounds = None
        dc.SetFont(font)
        if coord is None:
            if len(self.scaled):  # bugfix for Mac OS X
                if text_list:
                    extents = [dc.GetTextExtent(s) for s in text_list]
                    # widths = [dc.GetTextExtent(s).width for s in text_list]
                    centered = [[p[0] + x_adjust * e.width,
                                 p[1] + y_adjust * e.height]
                                for (p, e) in zip(self.scaled, extents)]
                    dc.DrawTextList(text_list,
                                    centered,
                                    foregrounds=foregrounds)
        else:
            pass


class MyCanvas(plot.PlotCanvas):
    def _xticks(self, *args):
        return [(np.log10(t), label) for (t, label) in self._xSpec]


def clip_db_rms(samples):
    mean_square = np.dot(samples, samples) / len(samples)
    return np.clip(10 * np.log10(mean_square), -120.0, 0.0)


def clip_db_power(pwr):
    return np.clip(10 * np.log10(pwr), -120.0, 0.0)


def find_center_frequency(x, samplerate=48000,
                          low_cutoff=17,
                          high_cutoff=135):
    bin_width = samplerate / len(x)
    min_bin = int(low_cutoff / bin_width)
    max_bin = int(high_cutoff / bin_width)
    ft = np.fft.rfft(x)
    power = ft.real ** 2 + ft.imag ** 2
    # cut off the spectrum to avoid noise skewing result
    power = power[min_bin:max_bin+1]
    # frequencies = np.arange(min_bin, max_bin+1) * bin_width
    # compute centroid of power spectrum
    # centroid = np.sum(frequencies * power) / np.sum(power)
    peak = np.argmax(power) + min_bin
    center = peak * bin_width
    # find the closest of our EQ frequencies
    # distances = abs(np.array(CENTER_FREQUENCIES) - centroid)
    distances = abs(np.array(CENTER_FREQUENCIES) - center)
    index = np.argmin(distances)
    # if distances[index] < 2:
    return index, center
    # return -1, center


class MainWindow(wx.Frame):
    def __init__(self, parent, title):
        wx.Frame.__init__(self, parent, title=title, size=(800, 640))
        self.volume_adjustment = 0
        wx.GetApp().SetAppName('VanderCal')
        self.config = wx.Config(wx.GetApp().AppName)
        self.canvas = MyCanvas(self)
        self.canvas.enableTitle = True
        self.canvas.title = 'Foo'
        self.canvas.logScale = True, False
        self.canvas.useScientificNotation = False
        self.canvas.enableGrid = False, True
        grid_pen = self.canvas.gridPen
        grid_pen.SetColour(wx.Colour(50, 50, 50, 255))
        # self.canvas.enableCenterLines = 'Horizontal'
        self.canvas.centerLinePen = wx.Pen('red')
        self.canvas.fontSizeAxis = 20
        self.canvas.fontSizeTitle = 20
        self.canvas.enableAntiAliasing = False
        wx.FontInfo()
        self.plot_font = wx.Font(wx.FontInfo(30))
        self.bass_mode = False

        self.CreateStatusBar()  # A Statusbar in the bottom of the window

        # Setting up the menu.
        filemenu = wx.Menu()

        # wx.ID_ABOUT and wx.ID_EXIT are standard IDs provided by wxWidgets.
        filemenu.Append(wx.ID_ABOUT, "&About",
                        " Information about this program")
        filemenu.AppendSeparator()
        filemenu.Append(wx.ID_EXIT, "E&xit", " Terminate the program")

        # Creating the menubar.
        menubar = wx.MenuBar()
        menubar.Append(filemenu,
                       "&File")  # Adding the "filemenu" to the MenuBar
        self.SetMenuBar(menubar)  # Adding the MenuBar to the Frame content.
        self.level_menu = wx.Menu()
        item = self.level_menu.AppendRadioItem(-1, 'Current',
                                               ' Display the current levels')
        self.level_menu.Bind(wx.EVT_MENU, self.memory_item_selected, item)
        self.current_item = item
        self.level_menu.AppendSeparator()
        item = self.level_menu.Append(wx.ID_CLEAR, 'Clear',
                                      " Clear all entries")
        self.level_menu.Bind(wx.EVT_MENU, self.clear_memory, item)
        self.toolmap = {}
        self.toolbar = self.make_toolbar()
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.canvas, 1, wx.EXPAND)

        self.SetAutoLayout(True)
        self.SetSizer(sizer)
        self.Layout()

        self.Show(True)
        self.hold = False
        self.set_reference_requested = False
        self.target_mode = False
        self.targets = None
        self.save_requested = False
        self.uncalibrated = None
        self.memory_index = -1
        self.level_memory = []
        self.memory_add_requested = False
        mic_sensitivity, mic_response = self.load_mic_calibration(
            MIC_CALIBRATION_FILE)
        self.mic_response = mic_response
        self.channel = 0
        self.channels = (0,)
        print('config db_reference:', self.config.ReadFloat('db_reference'))
        self.db_reference = self.config.ReadFloat('db_reference')
        # self.db_reference = 94 - mic_sensitivity
        print('db_reference:', self.db_reference)
        self.use_c_weighting = self.config.ReadBool('use_c_weighting', False)
        self.weighting = 0
        self.set_weighting()
        # self.toolbar.ToggleTool(self.toolmap['C Weighting'].Id,
        #                         self.use_c_weighting)
        # points = np.zeros((len(CENTER_FREQUENCIES), 2))
        # points[:, 0] = CENTER_FREQUENCIES
        # points[:, 1] = 70
        self.alpha = 0.2
        self.avg_power = 0
        self.avg_power_spectrum = None
        self.last_level = 0
        self.last_level_diff = 1e6
        self.measurement_ready = False
        self.last_points = ()
        # self.audio = AudioInOut()
        self.audio = Audio()

        # fifo_size = int(SWEEP_DURATION * SAMPLE_RATE / BLOCK_SIZE)
        self.sample_fifo = deque(maxlen=SAMPLE_FIFO_SIZE)
        # self.bin_width = SAMPLE_RATE / (SAMPLE_FIFO_SIZE * BLOCK_SIZE)
        # self.power_fifo = deque(maxlen=self.sample_fifo.maxlen)

        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.step_data = {}
        self.last_stage = None
        wx.CallLater(100, self.next_step)

    def make_toolbar(self):
        toolbar = self.CreateToolBar(wx.TB_HORIZONTAL | wx.TB_TEXT, wx.ID_ANY)
        tool_spec = (
            # bitmap filename, button type, label, callback, tooltip
            ('icons/Radio-Shack-SPL-small.png', 'normal', 'dB Ref',
             self.set_db_reference,
             'Click here when your SPL meter reads '
             '70 dB for a 1kHz test tone.'),
            # ('add-a-bar-chart-icone-8472-48.png', 'dropdown', 'Level Memory',
            #  self.level_memory_add,
            #  'Add the current levels to memory.  Right click to recall.'),
            # ('target.png', 'check', 'Target Mode', self.set_target_mode,
            #  'Show the difference between the current level '
            #  'and the target level.'),
            ('icons/pause-player-icone-5157-48.png', 'check', 'Hold Levels',
             self.set_hold, 'Hold all levels.'),
            ('icons/instagram_icon_48.png', 'normal', 'Save snapshot',
             self.save_snapshot,
             'Save an image of the plot to a file.'),
            # ('5210.png', 'check', 'C Weighting', self.set_c_weighting,
            #  'Use C weighting. For use with measurement microphones, '
            #  'not the Radio Shack meter.'),
            # ('icons8-bass-clef-48.png', 'check', 'Bass Level Mode',
            #  self.set_bass_mode,
            #  'Change to display for 20-320 Hz sweep.'),
            ('icons/Next-icon-48.png', 'normal', 'Next',
             self.next_step,
             'Move to the next calibration step.'),
        )
        for filename, button_type, label, callback, tip in tool_spec:
            bitmap = wx.Bitmap(filename, wx.BITMAP_TYPE_ANY)
            kind = getattr(wx, 'ITEM_' + button_type.upper())
            tool = toolbar.AddTool(-1, label, bitmap, wx.NullBitmap, kind=kind,
                                   # shortHelpString=tip
                                   )
            self.toolmap[label] = tool
            toolbar.Bind(wx.EVT_TOOL, callback, tool)
        # toolbar.AddSeparator()
        # self.volume_adjust_text = wx.StaticText(toolbar,
        #                                         label=VOL_ADJ_FORMAT.format(0),
        #                                         style=wx.SIMPLE_BORDER)
        # tooltip = ('Before starting EQ, adjust volume until '
        #            'this is as close to zero as possible')
        # self.volume_adjust_text.SetToolTip(tooltip)
        # font = wx.Font(18, wx.SWISS, wx.NORMAL, wx.NORMAL)
        # self.volume_adjust_text.SetFont(font)
        # toolbar.AddControl(self.volume_adjust_text)
        # toolbar.AddSeparator()
        # toolbar.AddSeparator()
        # self.response_text = wx.StaticText(toolbar,
        #                                    label=RESPONSE_FORMAT.format(0),
        #                                    style=wx.SIMPLE_BORDER)
        # tooltip = 'Frequency response range in dB'
        # self.response_text.SetToolTip(tooltip)
        # self.response_text.SetFont(font)
        # toolbar.AddControl(self.response_text)
        # dropdown = self.toolmap['Level Memory']
        # toolbar.SetDropdownMenu(dropdown.Id,
        #                         self.level_menu)
        # toolbar.Bind(wx.EVT_TOOL_DROPDOWN,
        #              self.set_memory_dropdown,
        #              dropdown)
        toolbar.Realize()
        return toolbar

    def set_db_reference(self, event):
        self.db_reference = 70 - self.last_level
        print('SETTING... set_db_reference:', self.db_reference, self.last_level)
        self.config.WriteFloat('db_reference', self.db_reference)
        self.config.Flush()
        print('Read back:', self.config.ReadFloat('db_reference'))
        print(dir(self.config))

    def compute_corrections(self, top_freq, decimation):
        """Compute amplitude corrections for an ideal input"""
        # rate, samples = wavfile.read(filename)
        # rate = SAMPLE_RATE
        # ideal_input = self.sweep
        # assert rate == SAMPLE_RATE
        # sample_width = 8 * samples.dtype.itemsize
        # ideal_input = samples[:, 0] / 2.0 ** (sample_width - 1)
        peak_level = 20 * np.log10(max(self.sweep))
        if self.mic_response is not None:
            mic_response = np.copy(self.mic_response)
            mic_response[:, 1] *= -1
        else:
            mic_response = None
        octave_filter = OctaveFilter(top_center_frequency=top_freq,
                                     decimation=decimation,
                                     mic_response=mic_response)
        band_power = octave_filter(self.sweep)
        correction = -10 * np.log10(band_power) + peak_level
        return correction

    def compute_targets(self):
        # targets are 25 to 30% of deviation from 70 dB
        targets = [70 + TARGET_FRACTION * (level - 70) for level in self.levels]
        targets.append(targets[-1])  # add right top edge
        return targets

    def set_target_mode(self, mode=True):
        if mode:
            self.targets = self.compute_targets()
            self.target_mode = True
            self.draw(self.levels)
            filename = '{} uncalibrated.png'.format(time.ctime())
            filename = os.path.join(SNAPSHOT_DIR, filename)
            self.save_snapshot(filename=filename)
            self.uncalibrated = self.levels
        else:
            filename = '{} calibrated.png'.format(time.ctime())
            filename = os.path.join(SNAPSHOT_DIR, filename)
            self.save_snapshot(filename=filename)
            self.target_mode = False
            self.uncalibrated = None
            self.targets = None

    def set_bass_mode(self, event):
        self.bass_mode = event.Selection
        self._set_bass_mode()

    # def _set_bass_mode(self):
    #     if self.bass_mode:
    #         # cal_file = CAL_AUDIO_FILE_BASS_LEVEL
    #         top_freq = 2 * TOP_FREQ
    #         decimation = DECIMATION_BASS
    #         self.center_frequencies = [2 * f for f in CENTER_FREQUENCIES]
    #         self.left_edges = [2 * e for e in LEFT_EDGES]
    #         self.frequency_names = CENTER_NAMES_BASS
    #         self.xlimits = tuple(2 * l for l in XAXIS_LIMITS)
    #     else:
    #         # cal_file = CAL_AUDIO_FILE_EQ
    #         top_freq = TOP_FREQ
    #         decimation = DECIMATION_EQ
    #         self.center_frequencies = CENTER_FREQUENCIES
    #         self.left_edges = LEFT_EDGES
    #         self.frequency_names = CENTER_NAMES_EQ
    #         self.xlimits = XAXIS_LIMITS
    #     # sweep extends 1 octave above and below the top and bottom band centers
    #     self.sweep = sp_sweep(hz_start=top_freq / 12, hz_end=2 * top_freq)
    #     self.output = np.zeros((len(self.sweep), 2), 'f')
    #     self.output[:, self.channel] = self.sweep
    #     self.octave_filter = OctaveFilter(top_center_frequency=top_freq,
    #                                       decimation=decimation)
    #     self.amplitude_corrections = self.compute_corrections(top_freq,
    #                                                           decimation)
    #     self.canvas.xSpec = list(zip(self.center_frequencies,
    #                                  self.frequency_names))

    def set_hold(self, event):
        self.hold = event.Selection
        if not self.hold:
            self.memory_index = -1
            self.level_menu.Check(self.current_item.Id, True)

    def save_snapshot(self, event=None, filename=''):
        self.hold = True
        self.canvas.SaveFile(filename)
        self.hold = False

    def next_step(self, event=None):
        # last_mode = self.step_data.get('mode')
        # last_stage = self.step_data.get('stage')
        # last_channels = self.step_data.get('channels')
        # self.step_data = next(cal_step_cycle)
        # self.stage = self.step_data['stage']
        # mode = self.step_data['mode']
        # mode_data = mode_map[mode]
        # self.channels = self.step_data['channels']
        # if self.stage == 'target':
        #     self.set_target_mode(True)
        # if last_stage == 'target':
        #     self.set_target_mode(False)
        # if mode != last_mode:
        #     center_frequencies = mode_data['frequencies']
        #     top_freq = center_frequencies[-1]
        #     decimation = mode_data['decimation']
        #     self.center_frequencies = mode_data['frequencies']
        #     self.left_edges = mode_data['edges']
        #     self.frequency_names = mode_data['names']
        #     self.xlimits = mode_data['xlimits']
        #     sweep_start, sweep_end = mode_data['sweep_range']
        #     # self.sweep = tones.sp_sweep(hz_start=sweep_start, hz_end=sweep_end)
        self.sweep = tones.warble_tone(CENTER_FREQUENCIES[-1])
        #     # self.octave_filter = OctaveFilter(top_center_frequency=top_freq,
        #     #                                   decimation=decimation)
        #     # self.amplitude_corrections = self.compute_corrections(top_freq,
        #     #                                                       decimation)
        self.canvas.xSpec = list(zip(CENTER_FREQUENCIES,
                                     CENTER_NAMES_EQ))
        # if self.channels != last_channels:
        self.output = tones.make_chunk_generator(self.sweep,
                                                 self.channels,
                                                 BLOCK_SIZE)
        self.audio.stop_acquisition()
        self.audio.start_acquisition(SAMPLE_RATE,
                                     BLOCK_SIZE,
                                     self.output,
                                     self._audio_callback)

    def set_c_weighting(self, event=None):
        print('tool toggled')
        self.use_c_weighting = event.Selection
        self.set_weighting()

    def set_weighting(self):
        if self.use_c_weighting:
            self.weighting = c_weighting(CENTER_FREQUENCIES)
        else:
            self.weighting = 0

    def level_memory_add(self, event=None):
        # self.memory_add_requested = True
        last_draw = self.canvas.last_draw
        if last_draw:
            graphics, _, _ = last_draw
            self.level_memory.append(graphics)

    def set_memory_dropdown(self, event):
        start = self.level_menu.MenuItemCount - 2
        end = len(self.level_memory) + 1
        print(start, end)
        for i in range(start, end):
            item = self.level_menu.InsertRadioItem(i, -1, str(i))
            self.level_menu.Bind(wx.EVT_MENU, self.memory_item_selected, item)
        event.Skip()

    def memory_item_selected(self, event):
        item = self.level_menu.FindItemById(event.Id)
        label = item.Label
        if label != 'Current':
            self.memory_index = int(item.Label) - 1
        else:
            self.memory_index = -1

    def clear_memory(self, event):
        self.memory_index = -1
        n = len(self.level_memory)
        self.level_memory.clear()
        for _ in range(n):
            item = self.level_menu.FindItemByPosition(1)
            self.level_menu.Delete(item)
        # current_item = self.level_menu.FindItemByPosition(0)
        self.level_menu.Check(self.current_item.Id, True)

    def draw(self, levels):
        points = list(zip(CENTER_FREQUENCIES, levels))
        fillcolour = 'green' if self.measurement_ready else 'light blue'
        objects = [plot.PolyBars(points,
                                 barwidth=BARWIDTH,
                                 # edgecolour='blue',
                                 fillcolour=fillcolour,
                                 fillstyle=wx.BRUSHSTYLE_SOLID
                                 ),
                   ]
        # if self.stage != 'bass_level':
        bar_numbers = ['{}'.format(i + 1) for i in range(len(points))]
        number_points = [[x, YAXIS_LIMITS[0] + .1]
                         for (x, _) in points]
        bar_number_object = PolyText(number_points,
                                     textList=bar_numbers,
                                     font=self.plot_font)
        objects.append(bar_number_object)

        # if self.targets:
        #     target_lines = list(zip(self.left_edges, self.targets))
        #     objects.append(plot.PolyLine(target_lines,
        #                                  drawstyle='steps-post',
        #                                  colour='brown',
        #                                  width=5,
        #                                  style=wx.PENSTYLE_SOLID
        #                                  ))
        # if self.stage in ['target', 'bass_level']:
        #     response = 0.5 * np.ptp(levels)
        #     response_txt = RESPONSE_FORMAT.format(response)
        #     response_pos = [points[-2][0], YAXIS_LIMITS[1] - 0.2]
        #     response_object = PolyText([response_pos],
        #                                textList=[response_txt],
        #                                adjust=(-0.5, 0.0),
        #                                font=self.plot_font)
        #     objects.append(response_object)
        # if not self.targets:
        #     # included_levels = levels[:-1]
        #     # vol_adj_index = np.argsort(included_levels)[VOLUME_ADJUSTMENT_ORDER]
        #     # vol_adj = DB_REFERENCE - included_levels[vol_adj_index]
        #     vol_adj = DB_REFERENCE - np.mean(levels[VOLUME_ADJUSTMENT_BANDS])
        #     vol_adj_txt = VOL_ADJ_FORMAT.format(vol_adj)
        #     vol_adj_pos = [points[5][0], YAXIS_LIMITS[1] - 0.2]
        #     vol_adj_object = PolyText([vol_adj_pos],
        #                               textList=[vol_adj_txt],
        #                               adjust=(-0.5, 0.0),
        #                               font=self.plot_font)
        #     objects.append(vol_adj_object)

        channel_txt = channel_text[self.channels]
        channel_pos = [points[0][0], YAXIS_LIMITS[1] - 0.2]
        channel_object = PolyText([channel_pos],
                                  textList=[channel_txt],
                                  adjust=(0.0, 0.0),
                                  font=self.plot_font)
        objects.append(channel_object)
        plot_graphics = plot.PlotGraphics(objects,
                                          xLabel=XLABEL,
                                          yLabel=YLABEL,
                                          title=TITLE)
        self.canvas.Draw(plot_graphics,
                         xAxis=XAXIS_LIMITS,
                         yAxis=YAXIS_LIMITS)
        if self.save_requested:
            self.save_snapshot()

    def _audio_callback(self, data):
        wx.CallAfter(self.process_audio, data)

    def process_audio(self, block):
        # center = find_center_frequency(block)
        # print('center:', center)
        # self.avg_power = np.mean(self.sample_fifo)
        # if self.avg_power == 0:
        #     self.avg_power = pwr
        # else:
        #     self.avg_power += self.alpha * (pwr - self.avg_power)
        self.sample_fifo.append(block)
        samples = np.concatenate(self.sample_fifo)
        # mean-square
        power = np.dot(samples, samples) / len(samples)
        # if self.avg_power_spectrum is None:
        #     self.avg_power_spectrum = pwr_spectrum
        # else:
        #     self.avg_power_spectrum += self.alpha * (pwr_spectrum
        #                                              - self.avg_power_spectrum)
        index, center = find_center_frequency(samples)
        level = clip_db_power(power)
        level_diff = abs(level - self.last_level)
        if (self.last_level_diff < MEASUREMENT_THRESHOLD_DB
                and level_diff < MEASUREMENT_THRESHOLD_DB):
            self.measurement_ready = True
        else:
            self.measurement_ready = False
        print('freq: %.1f, level: %.1f, level diffs: last=%.2f, this=%.2f'
              % (center, level, self.last_level_diff, level_diff))
        self.last_level_diff = level_diff
        print('process db_reference:', self.db_reference)
        self.last_level = level
        level += self.db_reference
        # level += self.db_reference
        # levels += self.db_reference + self.amplitude_corrections
        levels = np.zeros(len(CENTER_FREQUENCIES))
        if index > -1:
            levels[index] = level
        points = list(zip(CENTER_FREQUENCIES, levels))
        self.last_points = points
        self.levels = levels
        if not self.hold:
            self.draw(levels)

    def on_close(self, event):
        print("on_close")
        if self.audio:
            self.audio.stop_acquisition()
        self.config.WriteFloat('db_reference', self.db_reference)
        self.config.WriteBool('use_c_weighting', self.use_c_weighting)
        event.Skip()

    def load_mic_calibration(self, MIC_CALIBRATION_FILE):
        with open(MIC_CALIBRATION_FILE, 'r') as infile:
            line1 = infile.readline()
            match = sens_regex.search(line1)
            sensitivity = float(match.group(1))
            mic_response = np.loadtxt(infile)
        return sensitivity, mic_response


if __name__ == '__main__':
    app = wx.App(False)
    frame = MainWindow(None, "Real Time Analyzer")
    # filename = '/Users/dcook/Documents/REW/Pink_PN_65536_10_200_48.0k_PCM16_L.wav'
    # filename = '/Users/dcook/Documents/REW/LogSweep_10_240_48k_PCM16_L.wav'
    # corrections1 = frame.compute_corrections(filename,
    #                                          top_freq=TOP_FREQ,
    #                                          decimation=DECIMATION_EQ)
    # frame.mic_response = None
    # corrections = frame.compute_corrections(filename,
    #                                         top_freq=TOP_FREQ,
    #                                         decimation=DECIMATION_EQ)
    # print(corrections)
    # print(corrections1)
    # # print(corrections-3.01)
    # print(corrections - corrections1)
    app.MainLoop()
