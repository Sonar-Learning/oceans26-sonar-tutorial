import h5py
import json
from types import SimpleNamespace

import numpy as np


def ReadStaveData_Json(filename):
    with open(filename, 'r') as json_file:
        stave = json.load(json_file)

    data = stave['data']
    data = np.array(data)

    try:
        data_format = stave['signal']['data_format']
        if data_format == "real":
            # Data only has a real part - use as is
            pass
        elif data_format == "complex":
            # Data is interleaved real/imaginary doubles - convert to complex
            data = data[:, 0::2] + 1j * data[:, 1::2]
        else:
            print("Invalid data format key")
            raise KeyError
    except KeyError:
        # The data is interleaved complex
        data = data[:, 0::2] + 1j * data[:, 1::2]
        stave['signal']['data_format'] = "complex"

    return data

class StaveData():
    """
    Copied into Python OOP form from PR-MASTODON
        ReadStaveData.m matlab code and aided by
        genai.mil usage.
    """
    def __init__(self, filename:str):
        self.data_intake(filename)

    def data_intake(self, filename:str):
        with h5py.File(filename, 'r') as hf_file:
            self.n_time = hf_file["/signal/num_samples"][0]
            self.n_ele = hf_file["/sonars/receiver/nelem"][0]

            try:
                self.nElemVer = int(hf_file["/sonars/receiver/nelem_ver"][0])
            except (KeyError, ValueError):
                self.nElemVer = 1

            self.n_pings = int(hf_file["/vehicle/num_pings"][0])
            self.fc = hf_file["/sonars/projector/frequency"][0]
            self.bw = hf_file["/sonars/projector/bandwidth"][0]
            self.fs = hf_file["/signal/fs"][0]
            self.decimation = hf_file["/signal/decimation"][0]
            self.dy = hf_file["/vehicle/dping"][0]
            self.r_min = hf_file["/signal/rangemin"][0]
            self.r_max = hf_file["/signal/rangemax"][0]
            self.c = hf_file["/environment/soundspeed"][0]
            self.signal_applied = hf_file["/signal/signal_applied"][0]
            self.basebanded = hf_file["/signal/basebanded"][0]

            # Deprecated attributes, kept for compatibility
            self.proj_width = hf_file["/sonars/projector/width"][0]
            self.recv_width = hf_file["/sonars/receiver/width"][0]
            self.proj_height = hf_file["/sonars/projector/height"][0]
            self.recv_height = hf_file["/sonars/receiver/height"][0]

            # --- Projector and Receiver Nested Data ---
            self.proj = SimpleNamespace()
            self.recv = SimpleNamespace()

            try:
                self.proj.height = self.proj_height
                self.recv.height = self.recv_height
                self.proj.bearing = hf_file["/sonars/projector/bearing"][0]
                self.recv.bearing = hf_file["/sonars/receiver/bearing"][0]
                self.proj.depression = hf_file["/sonars/projector/depression"][0]
                self.recv.depression = hf_file["/sonars/receiver/depression"][0]
                self.veh_altitude = hf_file["/vehicle/altitude"][0]
                self.water_depth = hf_file["/environment/waterdepth"][0]
            except (KeyError, ValueError):
                # These fields may not exist in older files
                pass

            # --- Projector and Receiver Positions ---
            try:
                self.proj.positions = hf_file["/sonars/projector/positions"][()]
                self.recv.positions = hf_file["/sonars/receiver/positions"][()]
            except (KeyError, ValueError):
                # @todo: If positions are not specified, provide them based on other params.
                self.proj.positions = None
                self.recv.positions = None

            try:
                self.hor_spacing = float(hf_file["/sonars/receiver/hor_spacing"][()])
            except (KeyError, ValueError):
                self.hor_spacing = self.recv_width

            # --- Element Positions (Added 2022-AUG-31) ---
            try:
                self.proj.pos = SimpleNamespace()
                srcpos = hf_file["/vehicle/source_positions"][()]
                self.proj.pos.world = np.asarray(srcpos[1:, :])
                self.proj.pos.time = np.asarray(srcpos[0, :])

                self.recv.pos = SimpleNamespace()
                rcvpos = hf_file["/vehicle/receiver_positions"][()]
                self.recv.pos.world = np.asarray(rcvpos[1:, :])
                self.recv.pos.time = np.asarray(rcvpos[0, :])
            except (KeyError, ValueError):
                # These fields are optional
                pass

            # --- Optional Overlap Field ---
            try:
                self.overlap = float(hf_file["/vehicle/sas_overlap"][0])
            except (KeyError, ValueError):
                self.overlap = None

            # --- Signal Data ---
            self.signal = SimpleNamespace()
            if self.signal_applied:
                self.signal.times = np.asarray(hf_file["/signal/times"][()])
                signal_data = np.asarray(hf_file["/signal/signal"][()])
                # Convert structured numpy array (re, im) to complex
                self.signal.signal = (signal_data['re'] + 1j * signal_data['im']).T
                self.signal.pulse_length = hf_file["/signal/pulse_length"][0]

            # --- Main Data Array ---
            raw_data = np.asarray(hf_file["/data"][()])
            try:
                # h5py reads strings as bytes, so decode for comparison
                data_format = hf_file["/stave/signal/data_format"][()].decode('utf-8')
                self.signal.data_format = data_format

                if data_format == "real":
                    raise NotImplementedError("Real data in HDF5 has not been tested")
                elif data_format == "complex":
                    self.data = (raw_data['re'] + 1j * raw_data['im']).T
                else:
                    raise ValueError("Invalid data format key")
            except (KeyError, ValueError):
                # Older formats default to complex
                self.signal.data_format = "complex"
                self.data = (raw_data['re'] + 1j * raw_data['im']).T

            # --- Motion Data ---
            motion = np.asarray(hf_file["/vehicle/motion"][()])
            self.motion = [
                {
                    "time": motion["time"][n], "x": motion["x"][n], "y": motion["y"][n],
                    "z": motion["z"][n], "roll": motion["roll"][n], "pitch": motion["pitch"][n],
                    "yaw": motion["yaw"][n], "speed": motion["speed"][n]
                }
                for n in range(len(motion["time"]))
            ]
            self.motion2 = motion # Keep original structured array as well

            # --- World Motion Data ---
            motion_world = np.asarray(hf_file["/vehicle/motion_world"][()])
            self.motion_world = [
                {
                    "time": motion_world["time"][n], "latitude": motion_world["latitude"][n],
                    "longitude": motion_world["longitude"][n], "depth": motion_world["depth"][n],
                    "altitude": motion_world["altitude"][n], "roll": motion_world["roll"][n],
                    "pitch": motion_world["pitch"][n], "heading": motion_world["heading"][n],
                    "speed": motion_world["speed"][n]
                }
                for n in range(len(motion_world["time"]))
            ]
            self.motion_world2 = motion_world # Keep original structured array

            # --- Optional Navigation Data ---
            try:
                self.navigation = np.asarray(hf_file["/vehicle/navigation"][()])
            except (KeyError, ValueError):
                self.navigation = None