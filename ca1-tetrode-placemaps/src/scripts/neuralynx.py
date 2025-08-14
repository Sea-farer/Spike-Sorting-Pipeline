# neuralynx file loaders

import re
import logging
import pathlib
import warnings
import numpy as np
from os import path
from glob import glob
from collections import defaultdict
from datetime import datetime

from . import mclust
from .utils import handle_string  # str() -> {int(), float(), str}

log = logging.getLogger(__name__)


#
# Neuralynx File Parsers
#


class NeuralynxBase(object):

    HEADER_SIZE = 16 * 1024                         # 16kB

    def __init__(self, f):
        '''
        create a neuralynx reader
        '''
        if type(f) == str:
            self.fh = open(f, 'rb')
        else:
            self.fh = f

        self.fh_end = self.fh.seek(0, 2)

        self.header = None
        self.header = self.get_cooked_header()

        self.fh.seek(0, 0)

    def get_raw_header(self):
        ''' get the raw neuralynx file header '''
        self.fh.seek(0, 0)
        return self.fh.read(self.HEADER_SIZE).strip(b'\x00').decode('latin-1')

    def get_cooked_header(self, hdr=None):
        '''
        parse a raw header,
        returns dict of arrays-of-arrays keyed by header tokens.
        '''

        if self.header:
            return self.header

        res = defaultdict(list)

        if hdr is None:
            hdr = self.get_raw_header()

        for line in hdr.split('\r\n'):
            if re.match('^\s*#', line):  # skip comments
                continue

            items = re.split('\s+', line)
            items = [i for i in items if i != '']
            if len(items):
                k = items[0].lstrip('-')
                # XXX: handle CheetahVersion as str?, other special cases?
                v = [handle_string(s) for s in items[1:]]
                res[k].append(v)

        return res

    def __iter__(self):
        return self

    def __next__(self):
        ''' get next event '''
        pos = self.fh.tell()

        if pos < self.HEADER_SIZE:
            pos = self.fh.seek(self.HEADER_SIZE, 0)

        if pos < self.fh_end:
            return np.frombuffer(self.fh.read(self.REC_SIZE), self.dtype)
        else:
            raise StopIteration


class NeuralynxEvents(NeuralynxBase):
    '''
    NEV Parsing
    '''

    REC_SIZE = 184

    dtype = np.dtype([
        ('nstx', np.int16),                         # reserved
        ('npkt_id', np.int16),                      # ID for packet's system
        ('npkt_data_size', np.int16),               # == 2
        ('qwTimeStamp', np.uint64),                 # cheetah timestamp (uS)
        ('nevent_id', np.int16),                    # id for this event
        ('nttl', np.int16),                         # TTL value from input port
        ('ncrc', np.int16),                         # record CRC from cheetah
        ('ndummy1', np.int16),                      # reserved
        ('ndummy2', np.int16),                      # reserved
        ('dnExtra', (np.int32, 8)),                 # Extra bit values
        ('EventString', (np.string_, 128)),         # Event string
    ])
    # assert(len(np.empty(1, dtype)[0].tobytes()) == REC_SIZE)

    def __init__(self, f):
        '''
        create a neuralynx events reader
        '''
        super().__init__(f)


class NeuralynxVideoTracking(NeuralynxBase):
    '''
    NVT Stuff
    '''

    REC_SIZE = 1828  # XXX: consistent?

    dtype = np.dtype([
        ('swstx', np.uint16),                   # record start == 2048 / 0x800
        ('swid', np.uint16),                    # system ID
        ('swdata_size', np.uint16),             # size of videorec (bytes)
        ('qwTimeStamp', np.uint64),             # cheeta timestamp (uS)
        ('dwPoints', (np.uint32, 400)),         # color hitfield points
        ('sncrc', np.int16),                    # unused
        ('dnextracted_x', np.int32),            # extracted X location
        ('dnextracted_y', np.int32),            # extracted Y location
        ('dnextracted_angle', np.int32),        # angle (cw ° from y; else 0)
        ('dntargets', (np.int32, 50))           # colored targets
    ])

    def __init__(self, f):
        super().__init__(f)

    # TODO: verify/integrate bitfield stuff
    def do_bits(self, rec):
        targets = rec['dntargets'][0]
        nztargets = np.nonzero(targets)[0]
        n_nztargets = len(nztargets)

        recs = []

        if n_nztargets != 0:
            log.debug('{} targets'.format(n_nztargets))
            for g in targets[nztargets]:

                uh = np.int16((g & 0xFFFF0000) >> 16)   # upper half
                # ___ = uh & 0b100000000000000          # reserved
                pur_r = (uh & 0b010000000000000) >> 13  # pure red
                pur_g = (uh & 0b001000000000000) >> 12  # pure green
                pur_b = (uh & 0b000100000000000) >> 11  # pure blue
                loc_y = uh & 0b000011111111111          # y location

                lh = np.int16(g & 0x0000FFFF)           # lower half
                lumin = (lh & 0b100000000000000) >> 14  # luminance / intensity
                raw_r = (lh & 0b010000000000000) >> 13  # raw red
                raw_g = (lh & 0b001000000000000) >> 12  # raw green
                raw_b = (lh & 0b000100000000000) >> 11  # raw blue
                loc_x = lh & 0b000011111111111          # x location

                recs.append({
                    'lumin': lumin,
                    'raw_r': raw_r,
                    'raw_g': raw_g,
                    'raw_b': raw_b,
                    'pur_r': pur_r,
                    'pur_g': pur_g,
                    'pur_b': pur_b,
                    'loc_x': loc_x,
                    'loc_y': loc_y,
                })
        else:
            log.debug('no targets')

        return recs

    def ext_pos(self, targets, tracking=('pur_r', 'pur_g',)):
        '''
        Extract position information for targets.

        Tracking values are extrapolated from the first 2 colors detected in
        the file. Preference will be given to Red and Green LED's if
        available.

        The result will be the first x and y position pair found for each
        record of the corresponding LED in in the target list ('targets').

        In the special case that luminance ('lumin') is the only type detetced,
        the first 2 values in the 'targets' data will be returned.
        '''

        # XXX: include luminance??
        # rgbprio = ('pur_r', 'raw_r', 'pur_g', 'raw_g', 'pur_b', 'raw_b',)
        rgbprio = ('pur_r', 'raw_r', 'pur_g', 'raw_g', 'pur_b', 'raw_b',
                   'lumin',)

        # list and set of observed LED's:
        led_list = [k for i in targets for k in i.keys() if i[k] == 1]
        led_set = set(led_list)

        front_x = front_y = back_x = back_y = None  # return variables

        if led_set == {'lumin'}:  # luminance-only special case

            # XXX: no length check in .m code but len() == 1 files exist
            if len(targets) >= 1:
                front = targets[0]
                front_x = front['loc_x']
                front_y = front['loc_y']

            if len(targets) >= 2:
                back = targets[1]
                back_x = back['loc_x']
                back_y = back['loc_y']

            return front_x, front_y, back_x, back_y

        if len(led_set) < 2:  # insufficient information; fail
            return front_x, front_y, back_x, back_y

        # determine first and second led's to use
        frontLED = backLED = None
        led_prio = [i for i in rgbprio if i in led_list]  # prio-sorted subset

        for fl in ['pur_r', 'raw_r', led_prio[0]]:
            if fl in led_list:
                frontLED = fl

        for bl in ['pur_g', 'raw_g', led_prio[1]]:
            if bl in led_list:
                backLED = bl

        try:  # extract first selected led position
            front = next((i for i in targets if i[frontLED]))
            front_x = front['loc_x']
            front_y = front['loc_y']
        except StopIteration:
            front_x = front_y = None

        try:  # extract second selected led position
            back = next((i for i in targets if i[backLED]))
            back_x = back['loc_x']
            back_y = back['loc_y']
        except StopIteration:
            back_x = back_y = None

        return front_x, front_y, back_x, back_y

    def __next__(self):
        '''
        get next event
        currenly returning a 3-list of: (high, med, low)
        level data as returned from np reading, do_bits, and get_pos.
        '''
        low = super().__next__()
        med = self.do_bits(low)
        high = self.ext_pos(med)
        return (high, med, low)


class NeuralynxTetrode(NeuralynxBase):

    REC_SIZE = 304  # XXX: consistent

    dtype = np.dtype([
        ('qwTimeStamp', np.uint64),             # cheetah timestamp (uS)
        ('dwScNumber', np.uint32),              # spike acq entity no
        ('dwCellNumber', np.uint32),            # classified cell no or 0
        ('dnParams', (np.uint32, 8)),           # selected feature data
        ('snData', (np.int16, (32, 4))),        # data points [points, channel]
    ])

    def __init__(self, f):
        super().__init__(f)


class NeuralynxSampled(NeuralynxBase):

    REC_SIZE = 1044

    dtype = ([
        ('qwTimeStamp', np.uint64),             # cheetah timestamp (uS)
        ('dwChannelNumber', np.uint32),         # channel no (not A/D channel)
        ('dwSampleFreq', np.uint32),            # sampling frequency (Hz)
        ('dwNumValidSamples', np.uint32),       # samples with valid data
        ('snSamples', (np.int16, (512,))),      # sampled data points
    ])

    def __init__(self, f):
        super().__init__(f)


class NeuralynxEeg(NeuralynxSampled):

    ret_dtype = ([
        ('qwTimeStamp', np.uint64),             # cheetah timestamp (uS)
        ('dwChannelNumber', np.uint32),         # channel no (not A/D channel)
        ('dwSampleFreq', np.uint32),            # sampling frequency (Hz)
        ('dwNumValidSamples', np.uint32),       # samples with valid data
        # XXX: not actually 'sn'; but prefer consistency vs hu notation.
        ('snSamples', (np.double, (512,))),     # sampled data points
    ])

    def __init__(self, f):
        super().__init__(f)

    def __next__(self):
        '''
        Interpret sampled data as EEG;
        '''
        hdr = self.header
        low = super().__next__()

        nsamp = low['dwNumValidSamples'][0]

        try:
            volts = hdr.get('ADBitVolts')[0][0]  # IndexError

            if type(volts) is not float:

                raise ValueError()

        except (IndexError, ValueError):

                msg = "{}: header '{}' 'ADBitVolts' data problem".format(
                    self, hdr)

                log.error(msg)
                raise ValueError(msg)

        ret = np.zeros(low.shape, dtype=np.dtype(self.ret_dtype))

        for f in ret.dtype.fields:
            if f != 'snSamples':
                ret[f] = low[f]

        ret['snSamples'][:nsamp] = low['snSamples'][:nsamp] * volts
        log.debug(volts, low['snSamples'][:nsamp], ret['snSamples'][:nsamp])
        return ret


#
# Session Folder API
#


class NeuralynxSession:

    def __init__(self, basepath):
        self.basepath = basepath
        self._position = False
        self._spikes = False
        self._cuts = False
        self._eeg = False
        self._events = False
        self._start_time = None
        self._stop_time = None

        nttfiles = list(pathlib.Path(self.basepath).glob('*.[Nn][Tt][Tt]'))
        if len(nttfiles) == 0:
            raise FileNotFoundError('No NTT (.ntt) file found - not a Neuralynx data directory')
        f = NeuralynxEeg(nttfiles[0].as_posix())
        ncs_header = f.get_cooked_header()

        try:
            recording_starttime = np.ravel(ncs_header['TimeCreated'])
            recording_endtime = np.ravel(ncs_header['TimeClosed'])

            recording_starttime = datetime.strptime(' '.join(recording_starttime), '%Y/%m/%d %H:%M:%S')
            recording_endtime = datetime.strptime(' '.join(recording_endtime), '%Y/%m/%d %H:%M:%S')
            self.recording_time = recording_starttime
            self.recording_duration = (recording_endtime - recording_starttime).seconds
        except ValueError:
            # For some reason, the date information is missing from the header
            warnings.warn("Attribute `TimeCreated` missing from file header. Your data files may be corrupted, or an incompatible version")
            self.recording_time = None
            self.recording_duration = None

        try:
            self.recording_name = re.split(r'\\|/', np.ravel(ncs_header['OriginalFileName'])[0])[-2]
        except IndexError:
            warnings.warn("Attribute `OriginalFileName` missing from file header. Your data files may be corrupted, or an incompatible version")
            self.recording_name = None

        tetrode_list = sorted([int(re.match('.*?(\d+)', f.stem).groups()[0]) for f in nttfiles])
        self.tetrode_list = tetrode_list




    @property
    def position(self):
        if self._position is False:
            self._position = self.load_position()
        return self._position

    @property
    def spikes(self):
        if self._spikes is False:
            self._spikes = self.load_spikes()
        return self._spikes

    @property
    def cuts(self):
        if self._cuts is False:
            self._cuts = self.load_cuts()
        return self._cuts

    @property
    def eeg(self):
        if self._eeg is False:
            self._eeg = self.load_eeg()
        return self._eeg

    @property
    def events(self):
        if self._events is False:
            self._events = self.load_events()
        return self._events

    @property
    def start_time(self):
        if self._start_time is None:
            self.parse_events()
        return self._start_time

    @property
    def stop_time(self):
        if self._stop_time is None:
            self.parse_events()
        return self._stop_time

    def load_position(self):
        '''
        load .nvt files

        returns:

          HeaderDict{}, [np.array(NeuralynxVideoTracking.dtype), ...]

        '''
        nvtfiles = glob(path.join(self.basepath, '*.[Nn][Vv][Tt]'))
        if len(nvtfiles) != 1:
            log.warning('number of .nvt for {} != 1'.format(self.basepath))

        nvt = NeuralynxVideoTracking(nvtfiles[0])
        return nvt.get_cooked_header(), [evt for evt in nvt]

    def load_spikes(self):
        '''
        load .ntt files.

        return format:

          {N: 
           {'header': {}, 'data' : [np.array(NeuralynxTetrode.dtype), ...]}

        where:

          N == tetrode number

        '''
        nttfiles = glob(path.join(self.basepath, '*.[Nn][Tt][Tt]'))
        if len(nttfiles) == 0:
            log.warning('no .ntt files found in {}'.format(self.basepath))
            return False

        log.info('found {} .ntt files: {}'.format(len(nttfiles), nttfiles))

        nttids = []
        for f in nttfiles:
            try:
                # 'SC1.ntt' -> 'SC1' ; 'SC1' -> int('1') -> 1
                fbase = path.basename(f).split('.')[0]
                fid = int(re.match('.*?(\d+)', fbase).groups()[0])
                nttids.append(fid)
            except (IndexError, ValueError):
                log.error('.ntt filename problem for {}'.format(f))
                raise

        # {1: 'basepath\SC1.ntt', ... }
        filemap = {k: v for k, v in zip(nttids, nttfiles)}

        # {N: {'header' : {k: v},
        #  'data': [np.array(NeuralynxTetrode.dtype), ...]}, ...}
        res = defaultdict(dict)
        for k in filemap:
            f = NeuralynxTetrode(filemap[k])
            res[k]['header'] = f.get_cooked_header()
            res[k]['data'] = list(f)

        return dict(res)

    def load_cuts(self):

        return mclust.load_all_cuts(self.basepath)

    def load_eeg(self):
        '''
        load .ncs files.

        return format:

          {N : {'header': {}, 'data': [np.array(NeuralynxEeg.dtype), ...]}}

        where:

          N == channel number

        '''
        ncsfiles = glob(path.join(self.basepath, '*.[Nn][Cc][Ss]'))
        if len(ncsfiles) == 0:
            log.warning('no .ncs files found in {}'.format(self.basepath))
            return False

        log.info('found {} .ncs files: {}'.format(len(ncsfiles), ncsfiles))

        ncsids = []
        for f in ncsfiles:
            try:
                # 'CSC1.ncs' -> 'CSC1' ; 'CSC1' -> int('1') -> 1
                fbase = path.basename(f).split('.')[0]
                fid = int(re.match('.*?(\d+)', fbase).groups()[0])
                ncsids.append(fid)
            except (IndexError, ValueError):
                log.error('.ncs filename problem for {}'.format(f))
                raise

        # {1: 'basepath\SC1.ntt', ... }
        filemap = {k: v for k, v in zip(ncsids, ncsfiles)}

        # {N: {'header' : {k: v},
        #  'data': [np.array(NeuralynxEeg.dtype), ...]}, ...}
        res = defaultdict(dict)
        for k in filemap:
            f = NeuralynxEeg(filemap[k])
            res[k]['header'] = f.get_cooked_header()
            res[k]['data'] = list(f)

        return dict(res)

    def load_events(self):
        '''
        load .nev files.

        returns:

          HeaderDict{}, [np.array(NeuralynxEvents.dtype), ...]

        '''
        nevfiles = glob(path.join(self.basepath, '*.[Nn][Ee][Vv]'))
        if len(nevfiles) != 1:
            log.warning('number of .nev for {} != 1'.format(self.basepath))

        nev = NeuralynxEvents(nevfiles[0])
        return nev.get_cooked_header(), [evt for evt in nev]

    def parse_events(self):
        """
        Return a dictionary of events - (event_start, event_stop): event_name
        Parsing through "events" in .nev
        If an even is named "end" -> stop time for the previous event
        If an event is named "eventA_end" -> stop time for the 'eventA'
        """
        events = {}
        current_event = None
        for event in self.events[1]:
            name = event["EventString"].astype(str)[0]
            time_s = event["qwTimeStamp"][0] * 1e-6
            invalid_event_names = ("Starting Recording", "Stopping Recording")
            # if the name is not "end", does not end with "_end", and is not an invalid event name, it's the start of an event
            # If the name is one of the above, and an event has started, then that event stops
            # If the name is stopping_recording, but no event has taken place, add an event over the whole duration
            # If the name is stopping_recording and an event is still in progress, finish that event with the recording_stopping timestamp
            if name == "Starting Recording":
                self._start_time = time_s
            elif name == "Stopping Recording":
                self._stop_time = time_s
                if current_event is not None:
                    # There was no "end" signal to the final event
                    t1 = time_s
                    events[(t0, t1)] = current_event
                    current_event = None
                    t0 = None
            elif name == "end" or name.endswith("_end"):
                t1 = time_s
                events[(t0, t1)] = current_event
                current_event = None
                t0 = None
            else:
                current_event = name
                t0 = time_s
        if current_event is not None:
            # Potentially, in the event of computer trouble, there may be recordings which have neither an "end" nor a "stopping recording" value
            # This is a one-off catch-all for those cases.
            # Create a timestamp from the start/end file dates in the header
            def convert_date(list):
                """ example: list = ['2019/01/23', '12:59:29'] """
                return datetime.datetime.strptime("T".join(list), "%Y/%m/%dT%H:%M:%S")

            create_dt = convert_date(self.events[0]["TimeCreated"][0])
            end_dt = convert_date(self.events[0]["TimeClosed"][0])
            t1 = (end_dt - create_dt).total_seconds()
            events[(t0, t1)] = current_event
        return events


def get_recordings(input_dir):
    """
    returning a list of a single tuple (name, NeuralynxSession)
    - assuming there is always a single recording for a Neuralynx session
    - Neuralyx directory are identified by [ntt] file
    """
    # ----- try Neuralynx (indicated by .ntt files) -----
    try:
        rec = NeuralynxSession(input_dir)
        return [(rec.recording_name, rec)]
    except FileNotFoundError:
        return None
