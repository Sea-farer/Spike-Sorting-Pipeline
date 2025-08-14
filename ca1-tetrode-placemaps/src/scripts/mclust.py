# load mclust
import numpy as np
from .utils import advance_until
from os.path import splitext, join
import logging
from os import path
from glob import glob
from collections import defaultdict
import re

dtype_map = {
    't32': '>u4',
    't64': '>u8'
}

log = logging.getLogger(__name__)


def extract_header(f):
    header_string = advance_until(f, '%%ENDHEADER')
    header = [x.strip(' %') for x in header_string.split('\n')]
    kv = [e.split(':', 1) for e in header]
    kv = [p for p in kv if len(p) > 1]
    return {k.strip(): v.strip() for k, v in kv}


def load_tfile(path, ttype=None):
    with open(path, 'rb') as f:
        header = extract_header(f)

        if ttype is None:
            ttype = splitext(path)[1][1:]

        dtype = dtype_map.get(ttype)

        if not dtype:  # load and figure out the dtype
            pos = f.tell()

            # read up to 20 values and check if every other value increments
            val = np.fromfile(f, '>u4', 50)
            if np.all(np.diff(val[::2]) > 0):
                dtype = '>u4'
            else:
                dtype = '>u8'

            f.seek(pos)
        timestamps = np.fromfile(f, dtype) / 10000
    return header, timestamps


def load_all_cuts(basepath):
    '''
    load .t files.

    return format:

      {channel : {unit : mclust.load_tfile(), ...}, ...}

    where:

      load_tfile() -> (header, np.array('>u4|>u8'))

    '''

    tfiles = [*glob(path.join(basepath, '*.[Tt]')),
              *glob(path.join(basepath, '*.[Tt]32')),
              *glob(path.join(basepath, '*.[Tt]64'))]

    if len(tfiles) == 0:
        log.warning('no .t files found in {}'.format(basepath))
        return False

    log.info('found {} .t files: {}'.format(len(tfiles), tfiles))

    res = defaultdict(dict)
    for f in tfiles:
        try:
            # filename -> channel, unit
            # 'Sc8_1.t' -> 'Sc8_1' -> ['8','1'] -> [8, 1]
            fbase = path.basename(f).split('.')[0]
            fid = re.match(r'.*?(\d+)_.*?(\d+)$', fbase).groups()
            channel, unit = [int(i) for i in fid]

        except (IndexError, ValueError):
            log.error('.t filename problem for {}'.format(f))
            raise

        if channel in res and unit in res[channel]:
            raise ValueError(
                f'Found duplicated set of tetrode and cluster in the specified folder: '
            )

        try:
            res[channel][unit] = load_tfile(f)
        except Exception:
            log.error('error loading .t file {}'.format(f))

    return dict(res)
