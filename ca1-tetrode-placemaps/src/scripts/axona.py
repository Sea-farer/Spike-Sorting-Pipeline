#! /usr/bin/python3

"""
Axona session loader
"""

import numpy as np
from os.path import splitext, isfile
from collections import namedtuple
import re
import pathlib
from glob import glob
from datetime import datetime
from .utils import handle_string, advance_until, parse_rate


Spikes = namedtuple('Spikes', ['header', 'timestamps', 'n_spikes', 'n_channels'])
Position = namedtuple('Position', ['header', 'timestamps', 'coordinates', 'sample_rate'])
Cut = namedtuple('Cut', ['index'])
EEG = namedtuple('EEG', ['header', 'signal', 'sample_rate', 'type'])


def is_end_reached(f, end_string="data_end"):
    c = f.read()
    # put back the seeker
    f.seek(-len(c), 2)
    return c.decode('latin1') == end_string, c


def parse_header(text):
    """
    Expects `text` to be a multiline text of key value pair where each key value pair occurs on separate
    line and they are separated by a space.

    Example:

    text = '''
    key1 value1
    key2 value2
    '''
    """

    header = {}

    for line in text.split('\n'):
        line = line.strip()

        # skip blank line
        if line == '':
            continue

        parts = line.split(' ', 1)

        if len(parts) == 1:
            header[parts[0]] = None
        else:
            key, value = parts
            header[key] = handle_string(value)

    return header


def extract_header(f):
    header_str = advance_until(f, "data_start")
    if not header_str:
        raise ValueError("data_start was not found in the file")

    return parse_header(header_str)


def decode_digital_state(value):
    """
    Take a big-endian, unsigned, 16 bit number and produce a binary representation
    of the 16 channels involved
    e.g.: 1 -> channel 0 is on, channel 1-15 is off

    Returns a little-endian array of channel states, i.e. the 0th element of the
    array is channel 1, the final element is channel 16
    """
    value = int(value)
    assert value >= 0
    state = np.zeros(16).astype(bool)
    for i, s in enumerate("{0:b}".format(value)):
        state[i] = bool(int(s))
    return state
    

def load_inp_file(path):
    """ 
    Load the contents of the .inp file - this records digital IO events and keypresses
    Digital input/output is recorded as a 16 bit number based on the status 
    of all 16 digital channels, and therefore may require further decoding.
    Data format is documented here:
        http://space-memory-navigation.org/DacqUSBFileFormats.pdf
    """
    pass
    with open(path, "rb") as f:
        header = extract_header(f)
        bytes_per_timestamp = header.get('bytes_per_timestamp', 4)
        bytes_per_type = header.get("bytes_per_type", 1)
        bytes_per_value = header.get("bytes_per_value", 2)
        timebase = int(header.get("timebase", "16000 hz").split(" ")[0])

        dt = np.dtype([
                ("time", '>i{}'.format(bytes_per_timestamp)),
                ('type', np.string_, bytes_per_type),
                # ("value", '>u{}'.format(bytes_per_value)),
                ("value", '>u2'),
                ])
        data = np.fromfile(f, dtype=dt)
    
    # Change time from int to float, in order to switch from samples to seconds
    data = data.astype([
                ("time", float),
                ("type", dt[1]),  # Note, this is still a binary, not a true string
                ("value", dt[2]),
                ])
    data["time"] = data["time"] / timebase
    
    # Extract the different types
    # Remember - binary strings, hence the b flag
    digital_input = (data[data["type"] == b"I"])
    digital_output = (data[data["type"] == b"O"])
    keypress = (data[data["type"] == b"K"])
    
    # Convert the digital events to arrays of channel state
    # inputs[0] - time stamps in s
    # inputs[x] - values of channel X
    # inputs[1:, t] - values of all channels at time index t
    inputs = np.zeros((17, len(digital_input)))
    inputs[0] = digital_input["time"]
    for i in range(len(digital_input)):
        inputs[1:, i] = decode_digital_state(digital_input[i]["value"])
    
    outputs = np.zeros((17, len(digital_output)))
    outputs[0] = digital_output["time"]
    for i in range(len(digital_output)):
        inputs[1:, i] = decode_digital_state(digital_output[i]["value"])

    return inputs, outputs, keypress


def load_set_file(path):
    """
    Load the content of the SET file
    """
    with open(path, 'r', encoding='latin1') as f:
        text = f.read()

    return parse_header(text)


def load_spikes(path):
    # increment before, because channel_groups start at 1
    basename, ext = splitext(path)
    cgid = int(ext[1:])

    with open(path, "rb") as f:
        header = extract_header(f)
        bytes_per_timestamp = header.get('bytes_per_timestamp', 4)
        bytes_per_sample = header.get('bytes_per_sample', 1)
        timebase = int(header.get("timebase", "96000 hz").split(" ")[0])
        num_spikes = header.get('num_spikes', 0)
        num_chans = header.get('num_chans', 1)
        samples_per_spike = header.get('samples_per_spike', 50)
        raw_rate = header.get('rawrate', 48000)

        dtype = np.dtype([
            ('timestamp', '>u{}'.format(bytes_per_timestamp), 1),
            ("waveform", '<i{}'.format(bytes_per_sample), samples_per_spike)
        ])

        data = np.fromfile(f, dtype=dtype, count=num_chans * num_spikes)
        assert len(data) == num_chans * num_spikes, 'End of file reached before loading all data'
        assert is_end_reached(f), "Data still remaining in the file"

        timestamps = data['timestamp'][::num_chans] / timebase  # get timestamps for only first channel

        # TODO: handle waveform loading

        return header, timestamps, num_spikes, num_chans


def load_pos(path, threshold=0.95):
    with open(path, "rb") as f:
        header = extract_header(f)

        sample_rate = parse_rate(header["sample_rate"])  # sample_rate 50.0 hz

        num_colors = int(header["num_colours"])

        # TODO: use num_pos_samples
        num_pos_samples = int(header["num_pos_samples"])
        bytes_per_timestamp = int(header["bytes_per_timestamp"])
        bytes_per_coord = int(header["bytes_per_coord"])

        timestamp_dtype = ">i" + str(bytes_per_timestamp)
        coord_dtype = ">i" + str(bytes_per_coord)

        bytes_per_pixel_count = 4
        pixel_count_dtype = ">i" + str(bytes_per_pixel_count)

        pattern = ','.join(['t'] + ['x{n},y{n}'.format(n=n + 1) for n in range(num_colors)])

        pos_format = header['pos_format']
        if re.match(pattern, pos_format):
            two_spot = False
        elif re.match('t,x1,y1,x2,y2,numpix1,numpix2', pos_format):
            two_spot = True
        else:
            raise ValueError('pos file is encoded in unknown format "{}"'.format(pos_format))

        if two_spot:
            # TODO: check if should use "tracked_spots_count" instead of hard coded value
            dtype = np.dtype([("t", (timestamp_dtype, 1)),
                              ("coords", coord_dtype, 2 * 2),
                              ("pixel_count", (pixel_count_dtype, 1), 2)])
        else:
            dtype = np.dtype([("t", (timestamp_dtype, 1)),
                              ("coords", coord_dtype, 2 * num_colors)])

        data = np.fromfile(f, dtype=dtype)

    dt = 1 / sample_rate

    # time stamps are constructed based on the sampling rate
    timestamps = np.arange(len(data)) * dt

    coords = data['coords'].astype(float)

    # replace 1023 with nan
    coords[coords == 1023] = np.nan

    N = coords.shape[1]

    if N == 4:
        # handle old format with zero padded endings
        good_t = ~np.all(coords == 0, axis=1)
        timestamps = timestamps[good_t]
        coords = coords[good_t]

    if N > 4:
        # count the number of functional color channels
        valid_colors = np.any(~np.isnan(coords[:, 0:2 * N:2]), axis=0)
        n_valid_colors = valid_colors.sum()

        # select up to n_max_colors to return coordinates for
        n_max_colors = 2
        color_pos = np.where(valid_colors)[0][:n_max_colors]

        pos = [(coords[:, 2 * p] + header['window_min_x'], coords[:, 2 * p + 1] + header['window_min_y']) for p in
               color_pos]
    elif N == 4:  # 2-spot recording
        # not entirely sure if this really has to be handled as special case
        color_pos = [0, 1]
        pos = [(coords[:, 2 * p] + header['window_min_x'], coords[:, 2 * p + 1] + header['window_min_y']) for p in
               color_pos]
    else:
        raise ValueError("Expected at least 4 tracking coordinates but found only {}".format(N))

    # filter out position that is nan for >= threshold of samples
    # check if guaranteed inclusion of first coordinate is necessary
    pos_filtered = [p for i, p in enumerate(pos) if i == 0 or np.isnan(p[0]).mean() < threshold]

    # stack all coordinates into an array
    all_pos = np.concatenate([np.stack(xy, axis=-1) for xy in pos_filtered], axis=1)

    # TODO Remove NaN values at the endpoints here or elsewhere?

    return header, timestamps, all_pos, sample_rate


def load_eeg(path, data_type='EEG'):
    """
    Load and return the content of `.eeg/egf` or `.eeg/egf[0-9]+` files
    Args:
        path: Path to a single EEG/EGF file
        data_type: either EEG or EGF to indicate the type of data

    Returns:
        header, data, sample_rate, bytes_per_sample
    """
    with open(path, 'rb') as f:
        header = extract_header(f)
        n_samples = header['num_{}_samples'.format(data_type.upper())]

        sample_rate = parse_rate(header['sample_rate'])  # should be 250 Hz
        bytes_per_sample = header['bytes_per_sample']
        num_chans = header['num_chans']

        sample_dtype = ('<i{}'.format(bytes_per_sample), num_chans)
        data = np.fromfile(f, dtype=sample_dtype, count=n_samples)

        return header, data, sample_rate, bytes_per_sample


def scale_signal(data, adc_fullscale, bytes_per_sample, gain):
    value_range = 2 ** (8 * bytes_per_sample - 1)
    return (data / value_range) * (adc_fullscale / gain)


def extrapolate_nan_endpoints(all_pos):
    filled_pos = all_pos.copy()
    for c in range(filled_pos.shape[1]):
        r = filled_pos.shape[0] - 1
        if np.isnan(filled_pos[0, c]):
            j = 1
            while (j < r) & np.isnan(filled_pos[j, c]):
                j += 1
            if j < 1000:
                # Per BNT: % do not scale if j is too big, this will create artefacts
                filled_pos[0, c] = filled_pos[j, c] - 0.02 * j
            else:
                filled_pos[0, c] = filled_pos[j, c]
        if np.isnan(filled_pos[r, c]):
            k = r
            while (0 < k) & np.isnan(filled_pos[k, c]):
                k -= 1
            if r - k < 1000:
                filled_pos[r, c] = filled_pos[k, c] + 0.02 * (r - k - 1)
            else:
                filled_pos[r, c] = filled_pos[k, c]
    return filled_pos


class AxonaRecording:

    def __init__(self, basepath):
        self.basepath = str(basepath)
        self._position = False
        self._spikes = False
        self._cuts = False
        self._eeg = False
        self._egf = False
        self._digital_input = False
        self._digital_output = False
        self._keypresses = False

        self.settings = load_set_file(self.basepath + '.set')
        adc = self.settings['ADC_fullscale_mv']
        assert adc == 3680 or adc == 1500, 'Invalid value of ADC fullscale in mv: {}'.format(adc)
        self.adc_fullscale = adc
        self.eeg_gains = None
        self._prepare_gainmap()

        recording_time = ' '.join([self.settings['trial_date'], self.settings['trial_time']])
        self.recording_time = datetime.strptime(recording_time, '%A, %d %b %Y %H:%M:%S')
        self.recording_duration = self.settings['duration']
        tetrode_list = sorted(
            [int(f.name.split('.')[-1])
             for f in pathlib.Path(self.basepath).parent.glob('{}.[0-9]*'.format(pathlib.Path(self.basepath).name))]
        )
        self.tetrode_list = tetrode_list

    def _prepare_gainmap(self):
        """
        Create mapping between various channels and the gain
        Returns: gain map
        """
        attr = self.settings
        eeg_cand = [int(k.split('_')[-1]) for k, v in attr.items() if k.startswith('saveEEG') and v == 1]

        self.eeg_gains = {ch: attr['gain_ch_{}'.format(attr['EEG_ch_{}'.format(ch)])] for ch in eeg_cand}

    @property
    def position(self):
        if self._position is False:
            self._position = self.load_position()
        return self._position

    @property
    def cuts(self):
        if self._cuts is False:
            self._cuts = self.load_cuts()
        return self._cuts

    @property
    def spikes(self):
        if self._spikes is False:
            self._spikes = self.load_spikes()
        return self._spikes

    @property
    def eeg(self):
        if self._eeg is False:
            self._eeg = self.load_eeg()
        return self._eeg

    @property
    def egf(self):
        if self._egf is False:
            self._egf = self.load_egf()
        return self._egf

    @property
    def digital_input(self):
        if self._digital_input is False:
            self.load_digital_channels()
        return self._digital_input
    
    @property
    def digital_output(self):
        if self._digital_output is False:
            self.load_digital_channels()
        return self._digital_output

    @property
    def keypresses(self):
        if self._keypresses is False:
            self.load_digital_channels()
        return self._keypresses

    def load_digital_channels(self):
        # Unlike the other loads, doing the assignment here, to reduce duplication
        inp_path = self.basepath + ".inp"
        if not isfile(inp_path):
            return None
        self._digital_input, self._digital_output, self._keypresses = load_inp_file(inp_path)

    def load_position(self):
        pos_path = self.basepath + '.pos'
        if not isfile(pos_path):
            return None

        header, timestamps, coordinates, sample_rate = load_pos(pos_path)
        return Position(header, timestamps, coordinates, sample_rate)

    def load_spikes(self):
        spike_paths = glob('{}.[0-9]*'.format(self.basepath))

        spikes = {}
        for path in spike_paths:
            sgid = int(splitext(path)[1][1:])

            spikes[sgid] = Spikes(*load_spikes(path))

        return spikes

    def load_cuts(self):
        raise NotImplementedError

    def load_eeg(self):
        """
        Returns: EEG in mV
        """
        # do case insensitive search
        eeg_paths = glob('{}.EEG*'.format(self.basepath)) + glob('{}.eeg*'.format(self.basepath))

        eeg_map = {}
        for path in eeg_paths:
            ext = splitext(path)[1][1:]
            eeg_number = ext[3:]

            # if no trailing number, defaults to 1
            eeg_number = int(eeg_number) if eeg_number else 1
            header, data, sample_rate, bytes_per_sample = load_eeg(path)
            eeg_signal = scale_signal(data, self.adc_fullscale, bytes_per_sample, self.eeg_gains[eeg_number])

            eeg_map[eeg_number] = EEG(header, eeg_signal, sample_rate, 'EEG')

        return eeg_map

    def load_egf(self):
        """
        Returns: EGF in mV
        """
        # do case insensitive search
        egf_paths = glob('{}.EGF*'.format(self.basepath)) + glob('{}.egf*'.format(self.basepath))

        egf_map = {}
        for path in egf_paths:
            ext = splitext(path)[1][1:]
            egf_number = ext[3:]

            # if no trailing number, defaults to 1
            egf_number = int(egf_number) if egf_number else 1
            header, data, sample_rate, bytes_per_sample = load_eeg(path, data_type='EGF')
            egf_signal = scale_signal(data, self.adc_fullscale, bytes_per_sample, self.eeg_gains[egf_number])

            egf_map[egf_number] = EEG(header, egf_signal, sample_rate, 'EGF')

        return egf_map


def get_recordings(input_dir: pathlib.Path):
    """
    Given a path to a directory, identify all Axona sessions within that directory

    returns a list of tuple (name, AxonaRecording), one per recording found in this session
    """
    # ----- try AXONA (indicated by .set files) -----
    axona_recordings = [(axona_set.stem, AxonaRecording(
        axona_set.as_posix().replace('.set', '')))
                        for axona_set in input_dir.glob('*.set')]
    if len(axona_recordings) == 0:
        return None
    # sort by time
    axona_recordings = sorted(axona_recordings, key=lambda a: a[1].recording_time)
    return axona_recordings


    