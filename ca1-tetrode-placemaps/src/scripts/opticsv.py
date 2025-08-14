# optitrack csv data file loader

import logging
import pandas as pd
from datetime import datetime
import numpy as np
import csv

from .utils import handle_string

log = logging.getLogger(__name__)

SUPPORTED_VERSIONS = ("1.22", "1.23")


class OptiCSV:
    """
    Read optitrack csv - tested on 1.22, 1.23
    """

    def __init__(self, fname):
        '''
        Create Optitrack CSV reader from file name.
        '''
        self.fname = fname
        self.meta = self._load_header(fname)
        self.recording_time = parse_capture_start_time(self.meta['Capture Start Time'])
        self.sample_rate = float(self.meta['Export Frame Rate'])
        self.tracking_name = self.meta['Take Name']
        self._data = None
        self._secondary_data = None

    @property
    def data(self):
        if self._data is None:
            self._data = self._load_data(self.fname)
        return self._data

    @property
    def secondary_data(self):
        if self._secondary_data is None:
            self._secondary_data = self._load_data(self.fname, tier='secondary')
        return self._secondary_data

    @staticmethod
    def _load_header(fname):
        with open(fname, 'r') as f:
            for line_idx, line in enumerate(f):
                line = line.strip().strip('"')
                if line.startswith('Format Version'):
                    # [k1, v1, ..., ..., kN, vN] -> {k1: v1, ..., kN: vN}
                    li = iter(line.split(','))
                    header = dict(zip(li, li))
                    return header

    @staticmethod
    def _load_data(fname, tier='primary'):
        '''
        Load Optitrack CSV file.

        Loads CSV data lines according to the heading provided in 'Frame' line.
        Returns parsed 'format' dictionary from 'Format' line and list of
        data values.

        e.g:

            [Start of file]
            Format Version,1.22,Take Name,Take 2018-10-04 01.43.11 PM_001,Capture Frame Rate,240,Export Frame Rate,240,Capture Start Time,2018-10-05 12.05.03.445 P.M.,Total Frames in Take,864989,Total Exported Frames,864989,Rotation Type,Quaternion,Length Units,Meters,Coordinate Space,Global
            ...
            Frame,Time (Seconds),X,Y,Z,W,X,Y,Z,,,,,,,,,,,
            0,0,0.040908,0.936177,0.023746,0.348331,0.219875,0.077455,-1.918324,,,,,,,,,,,
            1,0.004167,0.038748,0.940821,0.033869,0.334973,0.220214,0.078433,-1.919889,,,,,,,,,,,

        will be interpreted/returned as:

            {'Format Version': '1.22', ... },  # format dict

            [{'Frame': 0, 'Time (seconds)': 0, ... },
                      {'Frame': 1, 'Time (seconds)': 0.004167, ...}]
        '''
        meta = OptiCSV._load_header(fname)
        version = meta.get("Format Version", None)
        if version not in SUPPORTED_VERSIONS:
            raise NotImplementedError(f"Opticsv does not currently support Format Version `{version}`")

        # --- Parse CSV file for the heading rows ---
        # "heading rows" are rows containing the naming and type of the data in the columns
        # E.g.: type/name of rigid bodies or markers, datatype (X, Y, Z or rotations)
        # This parsing step does the following:
        #   + Skip any blank rows before the header (with "Format Version", etc.)
        #   + Skip any blank rows after the header
        #   + Read the heading rows in fixed order (see below)
        heading_rows = ['marker_types', 'marker_names',
                        'marker_hashes', 'data_types', 'data_comps']
        header_passed = False
        heading_row_count = 0
        raw_heading_rows = {}
        with open(fname, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                heading_row_count += 1
                if len(heading_rows) == 0:
                    heading_row_count -= 1
                    break
                if row and row[0] == 'Format Version':
                    header_passed = True
                    continue
                if row and np.any([bool(val) for val in row]) and header_passed:
                    raw_heading_rows[heading_rows.pop(0)] = row
        # ---

        marker_types = raw_heading_rows['marker_types']
        marker_names = raw_heading_rows['marker_names']
        marker_hashes = raw_heading_rows['marker_hashes']
        data_types = raw_heading_rows['data_types']
        data_comps = raw_heading_rows['data_comps']

        d_type_mapper = {('Rotation', 'X'): 'qx',
                         ('Rotation', 'Y'): 'qy',
                         ('Rotation', 'Z'): 'qz',
                         ('Rotation', 'W'): 'qw',
                         ('Position', 'X'): 'x',
                         ('Position', 'Y'): 'y',
                         ('Position', 'Z'): 'z',
                         ('Speed', ''): 'speed'}
        m_type_mapper = {'Rigid Body': 'rigid_bodies',
                         'Rigid Body Marker': 'markers',
                         'Marker': 'markers'}

        data_s = {}
        if tier == 'primary':
            data_s = {'rigid_bodies': {}, 'markers': {}}

        for col_id, (m_type, m_name, m_hash, d_type, d_comp) in enumerate(
                zip(marker_types, marker_names, marker_hashes, data_types, data_comps)):

            # skip the first 2 columns: 'Frame' and 'Time'
            if col_id < 2:
                continue

            # skip any unrecognized column (perhaps different version)
            if tier == 'primary':
                if m_type not in m_type_mapper or (d_type, d_comp) not in d_type_mapper:
                    continue
            elif tier == 'secondary':
                if m_type in m_type_mapper or (d_type, d_comp) not in d_type_mapper:
                    continue

            if m_type == 'Rigid Body':
                marker_name = m_name
                if marker_name not in data_s['rigid_bodies']:
                    data_s['rigid_bodies'].update({marker_name: {}})
            elif m_type in ('Rigid Body Marker', 'Marker'):
                marker_name = m_type.replace('"', '') + '-' + m_name.replace('"', '')
                if marker_name not in data_s['markers']:
                    data_s['markers'].update({marker_name: {}})
            else:
                marker_name = m_name if m_name else m_type
                m_type_mapper.update({m_type: m_type})
                if m_type not in data_s:
                    data_s[m_type] = {}
                if marker_name not in data_s[m_type]:
                    data_s[m_type].update({marker_name: {}})

            data_s[m_type_mapper[m_type]][marker_name][d_type_mapper[(d_type, d_comp)]] = pd.read_csv(
                fname, usecols=[col_id], skiprows=heading_row_count).values.astype(float).flatten()

        data_s['inds'] = pd.read_csv(
                fname, usecols=[0], skiprows=heading_row_count).values.astype(int).flatten()
        data_s['t'] = pd.read_csv(
                fname, usecols=[1], skiprows=heading_row_count).values.astype(float).flatten()

        return data_s


# ============== HELPERS =====================

possible_formats = ['%Y-%m-%d %I.%M.%S.%f %p', '%Y-%m-%d %I.%M.%S %p', '%d %b %Y %H:%M:%S %z']


def parse_capture_start_time(capture_time):
    for dt_format in possible_formats:
        try:
            return datetime.strptime(capture_time.replace('P.M.', 'PM').replace('A.M.', 'AM'), dt_format)
        except:
            pass
    raise ValueError(f'Unknown "Capture Start Time" format {capture_time} - parsable formats are: {possible_formats}')


def extract_rb_position(rb):
    """ 'rb' being a rigid body or a marker - dictionary with x, y, z
    Restructure from right handed to left handed coordinate frame
    x = -x
    swapping y and z

    based on Torgeir Waaga script '+optitrack/@RigidBody/RigidBody.m'

    Return x, y, z
    """
    return -rb['x'], rb['z'], rb['y']


def extract_rb_rotation(rb):
    """ 'rb' being a rigid body - dictionary with:
    + qx, qy, qz, qw - rotation_type: 'xyzw' (quaternion)
    + qx, qy, qz - rotation_type: 'xyz'

    Process rotation based on Torgeir Waaga script '+optitrack/@RigidBody/RigidBody.m'
    Return yaw, pitch, roll
    """

    rotation_matrix = compute_rotation_matrix(rb)

    dvx = rotation_matrix[:, :, 0]

    # calculate pitch yaw (Azimuth)
    yaw = (np.arctan2(dvx[:, 1], dvx[:, 0]) + 2 * np.pi) % (2 * np.pi)

    # calculate pitch
    hypotxy = np.hypot(dvx[:, 0], dvx[:, 1])
    pitch = np.arctan2(dvx[:, 2], hypotxy)

    # calculate roll
    dvy = rotation_matrix[:, :, 2]
    hypotyxy = np.hypot(dvy[:, 0], dvy[:, 1])
    roll = np.arctan2(dvy[:, 2], hypotyxy)

    return yaw, pitch, roll


def compute_rotation_matrix(rb):
    """ 'rb' being a rigid body - dictionary with:
    + qx, qy, qz, qw - rotation_type: 'xyzw' (quaternion)
    + qx, qy, qz - rotation_type: 'xyz'

    Process rotation based on Torgeir Waaga script '+optitrack/@RigidBody/RigidBody.m'
    Return a rotation matrix
    """

    assert 'qx' in rb
    assert 'qy' in rb
    assert 'qz' in rb

    rotation_type = 'xyzw' if 'qw' in rb else 'xyz'

    if rotation_type == 'xyz':
        # to radian, sign inverse
        qx = -np.deg2rad(rb['qx'])
        qy = -np.deg2rad(rb['qy'])
        qz = -np.deg2rad(rb['qz'])

        c1 = -np.cos(qx)  # cos(phi)
        c2 = np.cos(qy)  # cos(theta)
        c3 = np.cos(qz)  # cos(psi)
        s1 = -np.sin(qx)  # sin(phi)
        s2 = np.sin(qy)  # sin(theta)
        s3 = np.sin(qz)  # sin(psi)

        # Calculate rotation matrix (A = BCD)
        A = np.array([[c2*c1, c2*s1, -s2],
                      [s3*s2*c1 - c3*s1, s3*s2*s1 + c3*c1, s3*c2],
                      [c3*s2*c1 + s3*s1, c3*s2*s1 - s3*c1, c3*c2]]).transpose((2, 1, 0))
        rotation_matrix = -A[:, [2, 0, 1], :]
        return rotation_matrix

    elif rotation_type == 'xyzw':
        signc = np.array([[[-1, -1, -1], [1, 1, 1], [1, 1, 1]]])

        qi = rb['qz']
        qj = rb['qy']
        qk = rb['qx']
        qr = rb['qw']

        A = np.array([[1 - 2*qj**2 - 2*qk**2, 2*(qi*qj - qk*qr), 2*(qi*qk + qj*qr)],
                      [2*(qi*qj + qk*qr), 1 - 2*qi**2 - 2*qk**2, 2*(qj*qk - qi*qr)],
                      [2*(qi*qk - qj*qr), 2*(qi*qr + qj*qk), 1 - 2*qi**2 - 2*qj**2]]).transpose((2, 1, 0))

        rotation_matrix = A[:, [2, 0, 1], :] * signc
        return rotation_matrix


if __name__ == "__main__":
    paths = {
        "good": r"N:\bigmaze\Data\Ratvissant\20180611\1",
        "bad_123": r"N:\valentno\Data\Ratvissant\20190124\1",
        "bad_122": r"N:\valentno\Data\Ratvissant\20181012\1",
    }
    opti = {}
    for key, path in paths.items():
        opti[key] = OptiCSV(path + r"\Optitrack\Optitrack.csv")