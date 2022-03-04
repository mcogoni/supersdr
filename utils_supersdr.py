import random
import struct
import array
import math
from collections import deque, defaultdict
import pickle
import threading, queue
import socket
import time
from datetime import datetime, timedelta
import sys
if sys.version_info > (3,):
    buffer = memoryview
    def bytearray2str(b):
        return b.decode('ascii')
else:
    def bytearray2str(b):
        return str(b)

import numpy as np
from scipy.signal import resample_poly, welch

import sounddevice as sd
import wave

import tkinter
from tkinter import *
from pygame.locals import *
import pygame, pygame.font, pygame.event, pygame.draw, string, pygame.freetype

from qrz_utils import *

from kiwi import wsclient
import mod_pywebsocket.common
from mod_pywebsocket.stream import Stream
from mod_pywebsocket.stream import StreamOptions
from mod_pywebsocket._stream_base import ConnectionTerminatedException

VERSION = "v3.11"

TENMHZ = 10000 # frequency threshold for auto mode (USB/LSB) switch
CW_PITCH = 0.5 # CW offset from carrier in kHz

# Initial KIWI receiver parameters
LOW_CUT_SSB = 30 # Bandpass low end SSB
HIGH_CUT_SSB = 3000 # Bandpass high end
LOW_CUT_CW = int(CW_PITCH*1000-200) # Bandpass for CW
HIGH_CUT_CW = int(CW_PITCH*1000+200) # High end CW
HIGHLOW_CUT_AM = 6000 # Bandpass AM
delta_low, delta_high = 0., 0. # bandpass tuning
default_kiwi_port = 8073
default_kiwi_password = ""

# predefined RGB colors
GREY = (200,200,200)
WHITE = (255,255,255)
BLACK = (0,0,0)
D_GREY = (50,50,50)
D_RED = (200,0,0)
D_BLUE = (0,0,200)
D_GREEN = (0,120,0)
RED = (255,0,0)
BLUE = (0,0,255)
GREEN = (0,255,0)
YELLOW = (200,180,0)
ORANGE = (255,140,0)

ALLOWED_KEYS = [K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, K_9]
ALLOWED_KEYS += [K_KP0, K_KP1, K_KP2, K_KP3, K_KP4, K_KP5, K_KP6, K_KP7, K_KP8, K_KP9]
ALLOWED_KEYS += [K_BACKSPACE, K_RETURN, K_ESCAPE, K_KP_ENTER]

HELP_MESSAGE_LIST = ["SuperSDR %s HELP" % VERSION,
        "",
        "- LEFT/RIGHT: move KIWI RX freq +/- 1kHz (+SHIFT: x10)",
        "- PAGE UP/DOWN: move WF freq +/- SPAN/4",
        "- UP/DOWN: zoom in/out by a factor 2X",
        "- U/L/C/A: switches to USB, LSB, CW, AM",
        "- J/K/O: tune RX low/high cut (SHIFT inverts, try CTRL!), O resets",
        "- G/H: inc/dec spectrum and WF averaging to improve SNR",
        "- ,/.(+SHIFT) change high(low) clip level for spectrum and WF",
        "- E: start/stop audio recording",
        "- F: enter frequency with keyboard",
        "- W/R: Write/Restore quick cyclic memory (up to 10)",
        "- SHIFT+W: Saves all memories to disk",
        "- SHIFT+R: Deletes all stored memories",
        "- V/B: up/down volume 10%, SHIFT+V mute/unmute",
        "- M: S-METER show/hide",
        "- Y: activate SUB RX or switch MAIN/SUB RX (+SHIFT kills it)",
        "- S: SYNC CAT and KIWI RX ON/OFF",
        "- Z: Center KIWI RX, shift WF instead",
        "- SPACE: FORCE SYNC of WF to RX if no CAT, else sync to CAT",
        "- X: AUTO MODE ON/OFF depending on amateur/broadcast band",
        "- D: connect/disconnect from DXCLUSTER server",
        "- I: show/hide EIBI database stations",
        "- Q: switch to a different KIWI server",
        "- 1/2 & 3: adjust AGC threshold, 3 switch WF autoscale",
        "- 0/9: [LOGGER] add QSO to log / open search QSO dialog",
        "- SHIFT+ESC: quits",
        "",
        "  --- 73 de marco/IS0KYB cogoni@gmail.com ---  "]

font_size_dict = {"small": 12, "medium": 16, "big": 18}

pygame.init()

nanofont = pygame.freetype.SysFont('Mono', 8)
microfont = pygame.freetype.SysFont('Mono', 10)
smallfont = pygame.freetype.SysFont('Mono', 12)
midfont = pygame.freetype.SysFont('Mono', 14)
bigfont = pygame.freetype.SysFont('Mono', 16)
hugefont = pygame.freetype.SysFont('Mono', 35)


class flags():
    # global mutable flags
    auto_mode = True
    input_freq_flag = False
    input_server_flag = False
    show_help_flag =  False
    s_meter_show_flag = False
    show_eibi_flag = False
    show_dxcluster_flag = False
    connect_dxcluster_flag = False
    input_callsign_flag = False
    input_qso_flag = False
    dualrx_flag = False
    click_drag_flag = False
    start_drag_x = None

    wf_cat_link_flag = True
    wf_snd_link_flag = False
    cat_snd_link_flag = True

            
class audio_recording():
    def __init__(self, kiwi_snd):
        self.filename = ""
        self.audio_buffer = []
        self.kiwi_snd = kiwi_snd
        self.frames = []
        self.recording_flag = False

    def start(self):
        self.filename = "supersdr_%sUTC.wav"%datetime.utcnow().isoformat().split(".")[0].replace(":", "_")
        print("start recording")
        self.audio_buffer = []
        self.recording_flag = True

    def stop(self):
        print("stop recording")
        self.recording_flag = False
        self.save()

    def save(self):
        self.wave = wave.open(self.filename, 'wb')
        self.wave.setnchannels(self.kiwi_snd.CHANNELS)
        self.wave.setsampwidth(2) # two bytes per sample (int16)
        self.wave.setframerate(self.kiwi_snd.AUDIO_RATE)
        # process audio data here
        self.wave.writeframes(b''.join(self.audio_buffer))
        self.wave.close()
        self.recording = False


class dxcluster():
    CLEANUP_TIME = 120
    UPDATE_TIME = 10
    SPOT_TTL_BASETIME = 600
    color_dict = {0: GREEN, SPOT_TTL_BASETIME: YELLOW, SPOT_TTL_BASETIME*2: ORANGE, SPOT_TTL_BASETIME*3: RED, SPOT_TTL_BASETIME*4: GREY}

    def __init__(self, mycall_):
        if mycall_ == "":
            raise
        self.mycall = mycall_
        host, port = 'dxfun.com', 8000
        self.server = (host, port)
        self.spot_dict = {}
        self.visible_stations = []
        self.terminate = False
        self.failed_counter = 0
        self.update_now = False

    def disconnect(self):
        self.terminate = True
        try:
            self.sock.shutdown(1)
            self.sock.close()
        except:
            pass
        print("DXCLuster disconnected!")
        
    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connected = False
        while not connected:
            print("Connecting to: %s:%d" % self.server)
            try:
                self.sock.connect(self.server)
            except:
                print('Impossibile to connect')
                sleep(5)     
            else:
                # self.sock.settimeout(1)
                print('Connected!!!')
                connected = True
        self.send(self.mycall)
        self.time_to_live = self.SPOT_TTL_BASETIME*5 # seconds for a spot to live
        self.last_update = datetime.utcnow()
        self.last_cleanup = datetime.utcnow()

    def send(self, msg):
        msg = msg + '\n'
        self.sock.send(msg.encode())

    def keepalive(self):
        try:
            self.send(chr(8))
            self.receive()
        except:
            print("DX cluster failed to reply to keepalive msg")

    def receive(self):
        try:
            msg = self.sock.recv(2048)
            msg = msg.decode("utf-8")
        except:
            msg = None
            #print("DX cluster msg decode failed")
        return msg

    def decode_spot(self, line):
        els = line.split("  ")
        els = [x for x in els if x]
        spotter = els[0][6:].replace(":", "")
        utc = datetime.utcnow()
        try:
            qrg = float(els[1].strip())
            callsign = els[2].strip()
            dxde_callsign = els[0][6:].split(":")[0]
            print("New SPOT:", utc.strftime('%H:%M:%SZ'), qrg, "kHz", callsign, "DX de", dxde_callsign)
        except:
            qrg, callsign, utc = None, None, None
            print("DX cluster msg decode failed: %s"%els)
        return qrg, callsign, utc, els

    def clean_old_spots(self):
        now  = datetime.utcnow()
        del_list = []
        for spot_id in self.spot_dict.keys():
            spot_utc = self.spot_dict[spot_id][2]
            duration = now - spot_utc
            duration_in_s = duration.total_seconds()
            if duration_in_s > self.time_to_live:
                del_list.append(spot_id)
        for spot_id in del_list:
            del self.spot_dict[spot_id]
        print("Number of spots in memory:", len(self.spot_dict.keys()))

    def run(self, kiwi_wf):
        self.connect()
        while not self.terminate:
            try:
                dx_cluster_msg = self.receive()
            except:
                continue

            spot_str = "%s"%dx_cluster_msg
            stored_something_flag = False
            for line in spot_str.replace("\x07", "").split("\n"):
                if "DX de " in line:
                    qrg, callsign, utc, spot_msg = self.decode_spot(line)
                    if qrg and callsign:
                        self.store_spot(qrg, callsign, utc, spot_msg)
                        stored_something_flag = True
                    else:
                        continue
            if stored_something_flag:           
                self.update_now = True

            delta_t = (datetime.utcnow() - self.last_cleanup).total_seconds()
            if delta_t > self.CLEANUP_TIME: # cleanup db and keepalive msg
                self.keepalive()
                self.clean_old_spots()
                self.last_cleanup = datetime.utcnow()
                # print("DXCLUST: cleaned old spots")
            delta_t = (datetime.utcnow() - self.last_update).total_seconds()
            if delta_t > self.UPDATE_TIME or self.update_now:
                self.get_stations(kiwi_wf.start_f_khz, kiwi_wf.end_f_khz)
                # print("DXCLUST: updated visible spots")
                self.last_update = datetime.utcnow()
                self.update_now = False
        print("Exited from DXCLUSTER loop")

    def store_spot(self, qrg_, callsign_, utc_, spot_msg_):
        spot_id = next(self.unique_id()) # create a unique hash for each spot
        self.spot_dict[spot_id] = (callsign_, qrg_, utc_, spot_msg_) # save spots as elements of a dictionary with hashes as keys

    def get_stations(self, start_f, end_f):
        count_dupes_dict = defaultdict(list)
        self.visible_stations = []
        for spot_id in self.spot_dict.keys():
            (callsign_, qrg_, utc_, spot_msg_) = self.spot_dict[spot_id]
            if start_f < qrg_ < end_f:
                count_dupes_dict[callsign_].append(spot_id)
                self.visible_stations.append(spot_id)

        self.visible_stations = sorted(self.visible_stations, key=lambda spot_id: self.spot_dict[spot_id][1])
        for call in count_dupes_dict.keys():
            same_call_list = []
            if len(count_dupes_dict[call])>1:
                same_call_list = sorted([spot_id for spot_id in count_dupes_dict[call]], key = lambda spot_id: self.spot_dict[spot_id][2])
                for spot_id in same_call_list[:-1]:
                    self.visible_stations.remove(spot_id)
                    del self.spot_dict[spot_id]

    def unique_id(self):
        seed = random.getrandbits(32)
        while True:
           yield seed
           seed += 1


class filtering():
    def __init__(self, fl, fs):
        b = fl/fs
        N = int(np.ceil((4 / b)))
        if not N % 2: N += 1  # Make sure that N is odd.
        self.n_tap = N
        self.h = np.sinc(2. * fl / fs * (np.arange(N) - (N - 1) / 2.))
        w = np.blackman(N)
        # Multiply sinc filter by window.
        self.h = self.h * w
        # Normalize to get unity gain.
        self.h = self.h / np.sum(self.h)

    def lowpass(self, signal):
        filtered_sig = np.convolve(signal, self.h, mode="valid")
        return filtered_sig


class memory():
    def __init__(self):
        self.mem_list = deque([], 10)
        self.index = 0
        try:
            self.load_from_disk()
        except:
            pass
        self.index = len(self.mem_list)

    def write_mem(self, freq, radio_mode, delta_low, delta_high):
        self.mem_list.append((round(freq, 3), radio_mode, delta_low, delta_high))
    
    def restore_mem(self):
        if len(self.mem_list)>0:
            self.index += 1
            self.index %= len(self.mem_list)
            return self.mem_list[self.index]
        else:
            return None
    
    def reset_all_mem(self):
        self.mem_list = deque([], 10)

    def save_to_disk(self):
        current_mem = self.mem_list
        self.load_from_disk()
        self.mem_list += current_mem
        self.mem_list = list(set(self.mem_list))
        try:
            with open("supersdr.memory", "wb") as fd:
                pickle.dump(self.mem_list, fd)
        except:
            print("Cannot save memory file!")

    def load_from_disk(self):
        try:
            with open("supersdr.memory", "rb") as fd:
                self.mem_list = pickle.load(fd)
        except:
            print("No memory file found!")


class kiwi_list():
    def __init__(self):
        self.mem_list = deque([], 10)
        self.index = 0
        try:
            self.load_from_disk()
        except:
            pass
        self.index = len(self.mem_list)

    def write_mem(self, host_, port_, pass_):
        self.mem_list.append((host_, port_, pass_))

    def delete_mem(self, index):
        try:
            del self.mem_list[index]
            return True
        except:
            return None

    def save_to_disk(self):
        try:
            with open("kiwi.list", "wb") as fd:
                pickle.dump(self.mem_list, fd)
        except:
            print("Cannot save kiwi list to disk!")

    def load_from_disk(self):
        try:
            with open("kiwi.list", "rb") as fd:
                self.mem_list = pickle.load(fd)
        except:
            print("No kiwi list file found!")


class kiwi_waterfall():
    MAX_FREQ = 30000
    CENTER_FREQ = int(MAX_FREQ/2)
    MAX_ZOOM = 14
    WF_BINS = 1024
    MAX_FPS = 23
    MIN_DYN_RANGE = 40. # minimum visual dynamic range in dB
    CLIP_LOWP, CLIP_HIGHP = 40., 100 # clipping percentile levels for waterfall colors
    delta_low_db, delta_high_db = 0, 0
    low_clip_db, high_clip_db = -120, -60 # tentative initial values for wf db limits
    wf_min_db, wf_max_db = low_clip_db, low_clip_db+MIN_DYN_RANGE
    kiwi_wf_timestamp = None
    
    def __init__(self, host_, port_, pass_, zoom_, freq_, eibi, disp):
        self.eibi = eibi
        # kiwi hostname and port
        self.host = host_
        self.port = port_
        self.password = pass_
        print ("KiwiSDR Server: %s:%d" % (self.host, self.port))
        self.zoom = zoom_
        self.freq = freq_
        self.averaging_n = 1
        self.wf_auto_scaling = True
        self.BINS2PIXEL_RATIO = disp.DISPLAY_WIDTH / self.WF_BINS

        self.old_averaging_n = self.averaging_n
        self.dynamic_range = self.MIN_DYN_RANGE
        
        self.wf_white_flag = False
        self.terminate = False

        if not self.freq:
            self.freq = 14200
        self.tune = self.freq
        self.radio_mode = "USB"

        print ("Zoom factor:", self.zoom)
        self.span_khz = self.zoom_to_span()
        self.start_f_khz = self.start_freq()
        self.end_f_khz = self.end_freq()
        
        self.div_list = []
        self.subdiv_list = []
        self.min_bin_spacing = 100 # minimum pixels between major ticks (/10 for minor ticks)
        self.space_khz = 10 # initial proposed spacing between major ticks in kHz

        self.counter, self.actual_freq = self.start_frequency_to_counter(self.start_f_khz)
        print ("Actual frequency:", self.actual_freq, "kHz")
        self.socket = None
        self.wf_stream = None
        self.wf_color = None

        # connect to kiwi WF server
        print ("Trying to contact %s..."%self.host)
        try:
            self.socket = socket.socket()
            self.socket.connect((self.host, self.port))
            print ("Socket open...")
        except:
            print ("Failed to connect")
            raise Exception()
        
        self.start_stream()
        
        while True:
            msg = self.wf_stream.receive_message()
            if msg:
                if bytearray2str(msg[0:3]) == "W/F":
                    break
                elif "MSG center_freq" in bytearray2str(msg):
                    els = bytearray2str(msg[4:]).split()                
                    self.MAX_FREQ = int(int(els[1].split("=")[1])/1000)
                    self.CENTER_FREQ = int(int(self.MAX_FREQ)/2)
                elif "MSG wf_fft_size" in bytearray2str(msg):
                    els = bytearray2str(msg[4:]).split()
                    self.MAX_ZOOM = int(els[3].split("=")[1])
                    self.WF_BINS = int(els[0].split("=")[1])
                    self.MAX_FPS = int(els[2].split("=")[1])
                
        self.bins_per_khz = self.WF_BINS / self.span_khz
        self.wf_data = np.zeros((disp.WF_HEIGHT, self.WF_BINS))
        self.avg_spectrum_deque = deque([], self.averaging_n)

    def gen_div(self):
        self.space_khz = 10
        self.div_list = []
        self.subdiv_list = []
        self.div_list = []
        f_s = int(self.start_f_khz)
        f_e = int(self.end_f_khz)
    
        while self.div_list == [] and self.subdiv_list == []:
            if self.bins_per_khz*self.space_khz > self.min_bin_spacing:
                for f in range(f_s, f_e+1):
                    if not f%self.space_khz:
                        fbin = int(self.offset_to_bin(f-self.start_f_khz))
                        self.div_list.append(fbin)

            if self.bins_per_khz*self.space_khz/10 > self.min_bin_spacing/10:
                for f in range(f_s, f_e+1):
                    if not f%(self.space_khz/10):
                        fbin = int(self.offset_to_bin(f-self.start_f_khz))
                        self.subdiv_list.append(fbin)
            self.space_khz *= 10                

    def start_stream(self):
        self.kiwi_wf_timestamp = int(time.time())
        print("WF timestamp", self.kiwi_wf_timestamp)
        uri = '/%d/%s' % (self.kiwi_wf_timestamp, 'W/F')

        try:
            handshake_wf = wsclient.ClientHandshakeProcessor(self.socket, self.host, self.port)
            handshake_wf.handshake(uri)
            request_wf = wsclient.ClientRequest(self.socket)
        except:
            return None
        request_wf.ws_version = mod_pywebsocket.common.VERSION_HYBI13
        stream_option_wf = StreamOptions()
        stream_option_wf.mask_send = True
        stream_option_wf.unmask_receive = False

        self.wf_stream = Stream(request_wf, stream_option_wf)
        print(self.wf_stream)
        if self.wf_stream:
            print ("Waterfall data stream active...")

        # send a sequence of messages to the server, hardcoded for now
        # max wf speed, no compression
        msg_list = ['SET auth t=kiwi p=%s ipl=%s'%(self.password, self.password), 'SET zoom=%d start=%d'%(self.zoom,self.counter),\
        'SET maxdb=-10 mindb=-110', 'SET wf_speed=4', 'SET wf_comp=0', "SET interp=13"]
        for msg in msg_list:
            self.wf_stream.send_message(msg)
        print ("Starting to retrieve waterfall data...")

    def zoom_to_span(self):
            """return frequency span in kHz for a given zoom level"""
            assert(self.zoom >= 0 and self.zoom <= self.MAX_ZOOM)
            self.span_khz = self.MAX_FREQ / 2**self.zoom
            return self.span_khz

    def start_frequency_to_counter(self, start_frequency_):
        """convert a given start frequency in kHz to the counter value used in _set_zoom_start"""
        assert(start_frequency_ >= 0 and start_frequency_ <= self.MAX_FREQ)
        self.counter = round(start_frequency_/self.MAX_FREQ * 2**self.MAX_ZOOM * self.WF_BINS)
        start_frequency_ = self.counter * self.MAX_FREQ / self.WF_BINS / 2**self.MAX_ZOOM
        return self.counter, start_frequency_

    def start_freq(self):
        self.start_f_khz = self.freq - self.span_khz/2
        return self.start_f_khz

    def end_freq(self):
        self.end_f_khz = self.freq + self.span_khz/2
        return self.end_f_khz

    def offset_to_bin(self, offset_khz_):
        bins_per_khz_ = self.WF_BINS / self.span_khz
        return bins_per_khz_ * (offset_khz_)

    def bins_to_khz(self, bins_):
        bins_per_khz_ = self.WF_BINS / self.span_khz
        return (1./bins_per_khz_) * (bins_) + self.start_f_khz

    def deltabins_to_khz(self, bins_):
        bins_per_khz_ = self.WF_BINS / self.span_khz
        return (1./bins_per_khz_) * (bins_)

    def receive_spectrum(self):
        msg = self.wf_stream.receive_message()
        if msg and bytearray2str(msg[0:3]) == "W/F": # this is one waterfall line
            msg = msg[16:] # remove some header from each msg AND THE FIRST BIN!
            self.spectrum = np.ndarray(len(msg), dtype='B', buffer=msg).astype(np.float32) # convert from binary data
            self.keepalive()

    def spectrum_db2col(self):
        wf = self.spectrum
        wf = -(255 - wf)  # dBm
        wf_db = wf - 13 + (3*self.zoom) # typical Kiwi wf cal and zoom correction
        wf_db[0] = wf_db[1] # first bin is broken
        
        if self.wf_auto_scaling:
            # compute min/max db of the power distribution at selected percentiles
            self.low_clip_db = np.percentile(wf_db, self.CLIP_LOWP)
            self.high_clip_db = np.percentile(wf_db, self.CLIP_HIGHP)
            self.dynamic_range = max(self.high_clip_db - self.low_clip_db, self.MIN_DYN_RANGE)

        # shift chosen min to zero
        wf_color_db = (wf_db - (self.low_clip_db+self.delta_low_db))
        # standardize the distribution between 0 and 1 (at least MIN_DYN_RANGE dB will be allocated in the colormap if delta=0)
        normal_factor_db = self.dynamic_range + self.delta_high_db
        self.wf_color = wf_color_db / (normal_factor_db-self.delta_low_db)

        self.wf_color = np.clip(self.wf_color, 0.0, 1.0)

        self.wf_min_db = self.low_clip_db + self.delta_low_db - (3*self.zoom)
        self.wf_max_db = self.low_clip_db + normal_factor_db - (3*self.zoom)

        # standardize again between 0 and 255
        self.wf_color *= 254
        # clip exceeding values
        self.wf_color = np.clip(self.wf_color, 0, 255)

    def set_freq_zoom(self, freq_, zoom_):
        self.freq = freq_
        self.zoom = zoom_
        self.zoom_to_span()
        self.start_freq()
        self.end_freq()
        if zoom_ == 0: # 30 MHz span, WF freq should be 15 MHz
            self.freq = self.CENTER_FREQ
            self.start_freq()
            self.end_freq()
            self.span_khz = self.MAX_FREQ
        else: # zoom level > 0
            if self.start_f_khz<0: # did we hit the left limit?
                #self.freq -= self.start_f_khz
                self.freq = self.zoom_to_span()/2
                self.start_freq()
                self.end_freq()
                self.zoom_to_span()
            elif self.end_f_khz>self.MAX_FREQ: # did we hit the right limit?
                self.freq = self.MAX_FREQ - self.zoom_to_span()/2 
                self.start_freq()
                self.end_freq()
                self.zoom_to_span()
        self.counter, actual_freq = self.start_frequency_to_counter(self.start_f_khz)
        msg = "SET zoom=%d start=%d" % (self.zoom, self.counter)
        self.wf_stream.send_message(msg)
        self.eibi.get_stations(self.start_f_khz, self.end_f_khz)
        self.bins_per_khz = self.WF_BINS / self.span_khz
        self.gen_div()

        return self.freq

    def keepalive(self):
        self.wf_stream.send_message("SET keepalive")

    def close_connection(self):
        if not self.wf_stream:
            return
        try:
            self.wf_stream.close_connection(mod_pywebsocket.common.STATUS_GOING_AWAY)
            self.socket.close()
        except Exception as e:
            print ("exception: %s" % e)

    def change_passband(self, delta_low_, delta_high_):
        if self.radio_mode == "USB":
            lc_ = LOW_CUT_SSB+delta_low_
            hc_ = HIGH_CUT_SSB+delta_high_
        elif self.radio_mode == "LSB":
            lc_ = -HIGH_CUT_SSB-delta_high_
            hc_ = -LOW_CUT_SSB-delta_low_
        elif self.radio_mode == "AM":
            lc_ = -HIGHLOW_CUT_AM-delta_low_
            hc_ = HIGHLOW_CUT_AM+delta_high_
        elif self.radio_mode == "CW":
            lc_ = LOW_CUT_CW+delta_low_
            hc_ = HIGH_CUT_CW+delta_high_
        self.lc, self.hc = lc_, hc_
        return lc_, hc_

    def set_white_flag(self):
        self.wf_color = np.ones_like(self.wf_color)*255
        self.wf_data[0,:] = self.wf_color

    def run(self):
        while not self.terminate:
            if self.averaging_n>1:
                self.avg_spectrum_deque = deque([], self.averaging_n)
                for avg_idx in range(self.averaging_n):
                    self.receive_spectrum()
                    self.avg_spectrum_deque.append(self.spectrum)
                self.spectrum = np.mean(self.avg_spectrum_deque, axis=0)
            else:
                self.receive_spectrum()

            self.spectrum_db2col()

            self.wf_data[1:,:] = self.wf_data[0:-1,:] # scroll wf array 1 line down
            self.wf_data[0,:] = self.wf_color # overwrite top line with new data
        return


class kiwi_sound():
    # Soundedevice options
    FORMAT = np.int16
    CHANNELS = 2
    AUDIO_RATE = 48000
    KIWI_RATE = 12000
    SAMPLE_RATIO = int(AUDIO_RATE/KIWI_RATE)
    CHUNKS = 1
    KIWI_SAMPLES_PER_FRAME = 512

    def __init__(self, freq_, mode_, lc_, hc_, password_, kiwi_wf, buffer_len, volume_=100, host_=None, port_=None, subrx_=False):
        self.subrx = subrx_
        # connect to kiwi server
        self.kiwi_wf = kiwi_wf
        self.host = host_ if host_ else kiwi_wf.host
        self.port = port_ if port_ else kiwi_wf.port
        self.FULL_BUFF_LEN = buffer_len
        self.audio_buffer = queue.Queue(maxsize=self.FULL_BUFF_LEN)
        self.terminate = False
        self.volume = volume_
        self.max_rssi_before_mute = -20
        self.mute_counter = 0
        self.muting_delay = 15

        self.run_index = 0
        self.delta_t = 0.0
        
        self.rssi = -127
        self.freq = freq_
        self.radio_mode = mode_
        self.lc, self.hc = lc_, hc_

        # Kiwi parameters
        self.on = True # AGC auto mode
        self.hang = False # AGC hang
        self.thresh = -80 # AGC threshold in dBm
        self.slope = 0 # AGC slope decay
        self.decay_other = 4000 # AGC decay time constant
        self.decay_cw = 1000 # AGC decay time constant
        self.gain = 50 # AGC manual gain
        self.min_agc_delay, self.max_agc_delay = 400, 8000
        self.decay = self.decay_other
        self.audio_balance = 0.0

        print ("Trying to contact server...")
        try:
            self.socket = socket.socket()
            self.socket.connect((self.host, self.port)) # future: allow different kiwiserver for audio stream
            new_timestamp = int(time.time())
            if new_timestamp - kiwi_wf.kiwi_wf_timestamp > 10:
                kiwi_wf.kiwi_wf_timestamp = new_timestamp
            uri = '/%d/%s' % (kiwi_wf.kiwi_wf_timestamp, 'SND')
            handshake_snd = wsclient.ClientHandshakeProcessor(self.socket, self.host, self.port)
            handshake_snd.handshake(uri)
            request_snd = wsclient.ClientRequest(self.socket)
            request_snd.ws_version = mod_pywebsocket.common.VERSION_HYBI13
            stream_option_snd = StreamOptions()
            stream_option_snd.mask_send = True
            stream_option_snd.unmask_receive = False
            self.stream = Stream(request_snd, stream_option_snd)
            
            print ("Audio data stream active...")
            msg_list = ["SET auth t=kiwi p=%s ipl=%s"%(password_, password_),
            "SET mod=%s low_cut=%d high_cut=%d freq=%.3f" % (self.radio_mode.lower(), self.lc, self.hc, self.freq), 
            "SET compression=0", "SET ident_user=SuperSDR","SET OVERRIDE inactivity_timeout=1000",
            "SET agc=%d hang=%d thresh=%d slope=%d decay=%d manGain=%d" % (self.on, self.hang, self.thresh, self.slope, self.decay, self.gain),
            "SET AR OK in=%d out=%d" % (self.KIWI_RATE, self.AUDIO_RATE)]
            
            for msg in msg_list:
                self.stream.send_message(msg)
            while True:
                msg = self.stream.receive_message()
                if msg and "SND" == bytearray2str(msg[:3]):
                    break
                elif msg and "MSG audio_init" in bytearray2str(msg):
                    print(msg)
                    msg = bytearray2str(msg)
                    els = msg[4:].split()                
                    self.KIWI_RATE = int(int(els[1].split("=")[1]))
                    KIWI_RATE_TRUE = float(els[2].split("=")[1])
                    self.delta_t = KIWI_RATE_TRUE - self.KIWI_RATE
                    self.SAMPLE_RATIO = self.AUDIO_RATE/self.KIWI_RATE
        except:
            print ("Failed to connect to Kiwi audio stream")
            raise
        
        self.kiwi_filter = filtering(self.KIWI_RATE/2, self.AUDIO_RATE)
        gcd = np.gcd((self.KIWI_RATE),self.AUDIO_RATE)
        self.n_low, self.n_high = int(self.KIWI_RATE/gcd), int(self.AUDIO_RATE/gcd)

        self.n_tap = self.kiwi_filter.n_tap
        self.lowpass = self.kiwi_filter.lowpass
        self.old_buffer = np.zeros((self.n_tap-1))

        self.audio_rec = audio_recording(self)

    def change_agc_delay(self, delta):
        if delta < 0:
            if self.decay > self.min_agc_delay:
                self.decay += delta
        else:
            if self.decay < self.max_agc_delay:
                self.decay += delta
        if self.radio_mode == "CW":
            self.decay_cw = self.decay
        else:
            self.decay_other = self.decay

        
    def set_agc_params(self):
        msg = "SET agc=%d hang=%d thresh=%d slope=%d decay=%d manGain=%d" % (self.on, self.hang, self.thresh, self.slope, self.decay, self.gain)
        self.stream.send_message(msg)

    def set_mode_freq_pb(self):
        self.decay = self.decay_other if self.radio_mode != "CW" else self.decay_cw
        msg = 'SET mod=%s low_cut=%d high_cut=%d freq=%.3f' % (self.radio_mode.lower(), self.lc, self.hc, self.freq)
        self.stream.send_message(msg)

    def get_audio_chunk(self):
        try:
            snd_buf = self.process_audio_stream()
        except:
            self.terminate = True
            return

        if snd_buf is not None:
            self.keepalive()
        else:
            snd_buf = None
        return snd_buf

    def process_audio_stream(self):
        try:
            data = self.stream.receive_message()
            # if not self.run_index % 100:
                # print(self.run_index, self.run_index * self.delta_t * self.KIWI_SAMPLES_PER_FRAME/self.KIWI_RATE)
            if self.run_index * self.delta_t * self.KIWI_SAMPLES_PER_FRAME/self.KIWI_RATE >= self.KIWI_SAMPLES_PER_FRAME: # self.KIWI_SAMPLES_PER_FRAME:
                # print("Double reading from server to compensate audio non integer sample rate!", self.run_index)
                data = self.stream.receive_message()
                self.run_index = 0
            if data is None:
                self.terminate = True
                self.kiwi_wf.terminate = True
                self.socket.close()
                print ('server closed the connection cleanly')
                raise
        except ConnectionTerminatedException:
                self.terminate = True
                self.kiwi_wf.terminate = True
                print('server closed the connection unexpectedly')
                raise

        #flags,seq, = struct.unpack('<BI', buffer(data[0:5]))
        if bytearray2str(data[0:3]) == "SND": # this is one waterfall line
            s_meter, = struct.unpack('>H',  buffer(data[8:10]))
            self.rssi = (0.1 * s_meter - 127)
            data = data[10:]
            count = len(data) // 2
            samples = np.ndarray(count, dtype='>h', buffer=data).astype(np.int16)
            return samples
        else:
            return None

    def change_passband(self, delta_low_, delta_high_):
        if self.radio_mode == "USB":
            lc_ = LOW_CUT_SSB+delta_low_
            hc_ = HIGH_CUT_SSB+delta_high_
        elif self.radio_mode == "LSB":
            lc_ = -HIGH_CUT_SSB-delta_high_
            hc_ = -LOW_CUT_SSB-delta_low_
        elif self.radio_mode == "AM":
            lc_ = -HIGHLOW_CUT_AM-delta_low_
            hc_ = HIGHLOW_CUT_AM+delta_high_
        elif self.radio_mode == "CW":
            lc_ = LOW_CUT_CW+delta_low_
            hc_ = HIGH_CUT_CW+delta_high_
        self.lc, self.hc = lc_, hc_
        return lc_, hc_

    def keepalive(self):
        self.stream.send_message("SET keepalive")

    def close_connection(self):
        if self.stream == None:
            return
        try:
            self.stream.close_connection(mod_pywebsocket.common.STATUS_GOING_AWAY)
            self.socket.close()
        except Exception as e:
            print ("exception: %s" % e)
    
    def play_buffer(self, outdata, frame_count, time_info, status):
        popped = []
        for _ in range(self.CHUNKS):
            popped.append( self.audio_buffer.get() )

        popped = np.array(popped).flatten()
        popped = popped.astype(np.float64) * (self.volume/100)

        n = len(popped)
        if self.SAMPLE_RATIO % 1: # high bandwidth kiwis (3ch 20kHz)
            pyaudio_buffer = resample_poly(popped, self.n_high, self.n_low, padtype="line")
        else: # normal 12kHz kiwis
            pyaudio_buffer = np.zeros(int(self.SAMPLE_RATIO*n))
            pyaudio_buffer[::int(self.SAMPLE_RATIO)] = popped
            pyaudio_buffer = np.concatenate([self.old_buffer, pyaudio_buffer])

            # low pass filter
            self.old_buffer = pyaudio_buffer[-(self.n_tap-1):]
            pyaudio_buffer = self.kiwi_filter.lowpass(pyaudio_buffer) * int(self.SAMPLE_RATIO)

        if self.audio_rec.recording_flag:
            self.audio_rec.audio_buffer.append(pyaudio_buffer.astype(np.int16))

        # mute on TX (over some rssi threshold)
        if self.rssi > self.max_rssi_before_mute:
            self.mute_counter = self.muting_delay
        elif self.mute_counter > 0:
            self.mute_counter -= 1
        if self.mute_counter > 0:
            pyaudio_buffer *= 0
        left_volume, right_volume = min(1-self.audio_balance, 1.0), min(1+self.audio_balance, 1.0)
        outdata[:,0] = (pyaudio_buffer*left_volume**2).astype(np.int16)   # LEFT  CHANNEL
        outdata[:,1] = (pyaudio_buffer*right_volume**2).astype(np.int16)     # RIGHT CHANNEL
        
    def run(self):
        while not self.terminate:
            snd_buf = self.get_audio_chunk()
            if snd_buf is not None:
                self.audio_buffer.put(snd_buf)
                self.run_index += 1
        return


class cat:
    CAT_MIN_FREQ = 100 # 100 kHz is OK for most radios
    CAT_MAX_FREQ = 30000
    def __init__(self, radiohost_, radioport_):
        self.KNOWN_MODES = {"USB", "LSB", "CW", "AM"}
        self.radiohost, self.radioport = radiohost_, radioport_
        print ("RTX rigctld server: %s:%d" % (self.radiohost, self.radioport))
        # create a socket to communicate with rigctld
        self.socket = socket.socket()
        self.socket.settimeout(3.0)
        try: # if rigctld is running but the radio is off this will seem OK... TBF!
            self.socket.connect((self.radiohost, self.radioport))
        except:
            return None
        self.freq = self.get_freq()
        if not self.freq:
            return None
        self.radio_mode = self.get_mode()
        self.reply = None
        self.cat_ok = True
        self.cat_tx = False

    def send_msg(self, msg):
        self.socket.send((msg+"\n").encode())
        try:
            out = self.socket.recv(512).decode() # tbi implement verification of reply
        except:
            out = ""
        if len(out)==0 or "RPRT -5" in out:
             self.cat_ok = False
             self.reply = None
        else:
            self.reply = out        

    def get_ptt(self):
        self.send_msg("\\get_ptt")
        if self.reply:
            self.cat_tx = True if self.reply=="1\n" else False

    def set_freq(self, freq_):
        if freq_ >= self.CAT_MIN_FREQ and freq_ <= self.CAT_MAX_FREQ:
            self.send_msg(("\\set_freq %d" % (freq_*1000)))
            self.freq = freq_

    def set_mode(self, radio_mode_):
        self.send_msg(("\\set_mode %s 2400"%radio_mode_))
        if self.reply:
            self.radio_mode = radio_mode_

    def get_freq(self):
        self.send_msg("\\get_freq")
        if self.reply:
            try:
                self.freq = int(self.reply)/1000.
            except:
                self.cat_ok = False

        return self.freq

    def get_mode(self):
        self.send_msg("\\get_mode")
        if self.reply:
            self.radio_mode = self.reply.split("\n")[0]
            if self.radio_mode not in self.KNOWN_MODES:
                self.radio_mode = "USB" # defaults to USB if radio selects RTTY, FSK, etc
            return self.radio_mode
        else:
            return "USB"


# Approximate HF band plan from https://www.itu.int/en/ITU-R/terrestrial/broadcast/Pages/Bands.aspx
# and https://www.iaru-r1.org/reference/band-plans/hf-bandplan/
def get_auto_mode(f):
    automode_dict = {"USB": ((14100,14350),(18110,18168),(21150,21450),(24930,24990),(28300,29100)),
                "LSB": ((1840,1850),(3600,3800),(7060,7200)),
                "CW": ((1810, 1840),(3500,3600),(7000,7060),(10100, 10150),(14000,14100),
                    (18068,18110),(21000,21150),(24890,24930),(28000,28190)),
                "AM": ((148,283),(520,1720),(2300,2500),(3200,3400),(3900,4000),(4750,5060),
                    (5900,6200),(7200,7450),(9400,9900),(11600,12100),(13570,13870),(15100,15800),
                    (17480,17900),(18900,19020),(21450,21850),(25670,26100))}

    f = round(f)
    for mode_ in automode_dict:
        for rng in automode_dict[mode_]:
            if f in range(rng[0], rng[1]):
                return mode_
    # if f not in bands, apply generic rule
    return "USB" if f>TENMHZ else "LSB"


def start_audio_stream(kiwi_snd):
    def _get_std_input_dev():
        devices = sd.query_devices()
        for dev_id, device in enumerate(devices):
            if device["max_input_channels"] > 0 and "pulse" in device["name"]:
                std_dev_id = dev_id
            else:
                std_dev_id = None
        return std_dev_id

    rx_t = threading.Thread(target=kiwi_snd.run, daemon=True)
    rx_t.start()

    print("Filling audio buffer...")
    while kiwi_snd.audio_buffer.qsize() < kiwi_snd.FULL_BUFF_LEN and not kiwi_snd.terminate:
        pass

    if kiwi_snd.terminate:
        print("kiwi sound not started!")
        del kiwi_snd
        return (None, None)

    std_dev_id = _get_std_input_dev()
    kiwi_audio_stream = sd.OutputStream(blocksize = int(kiwi_snd.KIWI_SAMPLES_PER_FRAME*kiwi_snd.CHUNKS*kiwi_snd.SAMPLE_RATIO),
                        device=std_dev_id, dtype=kiwi_snd.FORMAT, latency="low", samplerate=kiwi_snd.AUDIO_RATE, channels=kiwi_snd.CHANNELS, callback = kiwi_snd.play_buffer)
    kiwi_audio_stream.start()

    return True, kiwi_audio_stream


class eibi_db():
    def __init__(self):
        try:
            with open("./eibi.csv", encoding="latin") as fd:
                data = fd.readlines()
        except:
            return None
        label_list = data[0].rstrip().split(";")
        self.station_dict = defaultdict(list)
        self.int_freq_dict = defaultdict(list)
        self.station_freq_dict = {}
        self.visible_stations = []

        for el in data[1:]:
            els = el.rstrip().split(";")
            self.int_freq_dict[int(round(float(els[0])))].append(float(els[0])) # store or each integer freqeuncy key all float freq in kHz
            self.station_dict[float(els[0])].append(els[1:]) # store all stations' data using the float freq in kHz as key (multiple)
            self.station_freq_dict[float(els[0])] = els[1:]

        self.freq_set = set(self.int_freq_dict.keys())

    def get_stations(self, start_f, end_f):
        inters = set(range(int(start_f), int(end_f))) & self.freq_set
        self.visible_stations = []
        for intf in inters:
            record = self.int_freq_dict[intf]
            for f_khz in record:
                self.visible_stations.append(f_khz) #self.station_dict[f_khz])
        return self.visible_stations

    def get_names(self, f_khz):
        name_list = []
        for station_record in self.station_dict[f_khz]:
            name_list.append(station_record[3])
        return name_list


class display_stuff():
    def __init__(self, DISPLAY_WIDTH):
        # SuperSDR constants
        self.DISPLAY_WIDTH = DISPLAY_WIDTH
        DISPLAY_HEIGHT = DISPLAY_WIDTH//2
        self.WF_HEIGHT = DISPLAY_HEIGHT*60//100
        self.SPECTRUM_HEIGHT = DISPLAY_HEIGHT*40//100
        self.TOPBAR_HEIGHT = 23
        self.BOTTOMBAR_HEIGHT = 20
        self.TUNEBAR_HEIGHT = 23
        self.DISPLAY_HEIGHT = self.WF_HEIGHT + self.SPECTRUM_HEIGHT + self.TOPBAR_HEIGHT + self.BOTTOMBAR_HEIGHT + self.TUNEBAR_HEIGHT
        self.TOPBAR_Y = 0
        self.SPECTRUM_Y = self.TOPBAR_HEIGHT
        self.TUNEBAR_Y = self.SPECTRUM_Y + self.SPECTRUM_HEIGHT
        self.WF_Y = self.TUNEBAR_Y + self.TUNEBAR_HEIGHT
        self.BOTTOMBAR_Y = self.WF_Y + self.WF_HEIGHT
        self.SPECTRUM_FILLED = True
        self.V_POS_TEXT = 6

    def create_cm(self, which):
        if which == "cutesdr":
            # this colormap is taken from CuteSDR source code
            colormap = []
            for i in range(255):
                if i<43:
                    col = ( 0,0, 255*(i)/43)
                if( (i>=43) and (i<87) ):
                    col = ( 0, 255*(i-43)/43, 255 )
                if( (i>=87) and (i<120) ):
                    col = ( 0,255, 255-(255*(i-87)/32))
                if( (i>=120) and (i<154) ):
                    col = ( (255*(i-120)/33), 255, 0)
                if( (i>=154) and (i<217) ):
                    col = ( 255, 255 - (255*(i-154)/62), 0)
                if( (i>=217) ):
                    col = ( 255, 0, 128*(i-217)/38)
                colormap.append(col)
        # elif which == "jet":
        #     # setup colormap from matplotlib
        #     colormap = cm.jet(range(256))[:,:3]*255
        return colormap

    def update_textsurfaces(self, surface_, radio_mode, rssi_smooth, mouse, wf_width, kiwi_wf, kiwi_snd, kiwi_snd2, fl, cat_radio, kiwi_host2, run_index):
        mousex_pos = mouse[0]
        if mousex_pos < 25:
            mousex_pos = 25
        elif mousex_pos >= self.DISPLAY_WIDTH - 80:
            mousex_pos = self.DISPLAY_WIDTH - 80
        mouse_khz = kiwi_wf.bins_to_khz(mouse[0]/kiwi_wf.BINS2PIXEL_RATIO)
        buff_level = kiwi_snd.audio_buffer.qsize()
        main_rx_color = RED #if not kiwi_snd.subrx else GREEN
        sub_rx_color = GREEN #if not kiwi_snd.subrx else RED
        tx_on_flag = False
        audio_balance_string_list = ["LEFT", "C-LEFT", "CENTER", "C-RIGHT", "RIGHT"]
        audio_balance_string_main = audio_balance_string_list[int((kiwi_snd.audio_balance+1)*2)]

        if cat_radio:
            tx_on_flag = cat_radio.cat_tx
        #           Label   Color   Freq/Mode                       Screen position
        ts_dict = {"wf_freq": (YELLOW, "%.1f"%(kiwi_wf.freq if fl.cat_snd_link_flag else kiwi_wf.freq), (wf_width/2-48,self.TUNEBAR_Y+1), "small", False),
                "left": (GREEN, "%.1f"%(kiwi_wf.start_f_khz) ,(0,self.TUNEBAR_Y+1), "small", False),
                "right": (GREEN, "%.1f"%(kiwi_wf.end_f_khz), (wf_width-50,self.TUNEBAR_Y+1), "small", False),
                "rx_freq": (main_rx_color, "MAIN:%.3fkHz %s [%s]"%(kiwi_snd.freq+(CW_PITCH if kiwi_snd.radio_mode=="CW" else 0), kiwi_snd.radio_mode, "MUTE" if kiwi_snd.volume==0 else "%d%%"%kiwi_snd.volume), (wf_width/2-120,self.V_POS_TEXT-1), "big", False),
                "kiwi": (RED if not kiwi_snd.subrx else GREEN, kiwi_wf.host[:30]+" - BAL: %s"%audio_balance_string_main ,(95,self.BOTTOMBAR_Y+6), "small", False),
                "span": (GREEN, "SPAN:%.0fkHz"%((kiwi_wf.span_khz)), (wf_width-95,self.SPECTRUM_Y+1), "small", False),
                "filter": (GREY, "FILT:%.1fkHz"%((kiwi_snd.hc-kiwi_snd.lc)/1000.), (wf_width/2+230, self.V_POS_TEXT), "small", False),
                "p_freq": (WHITE, "%dkHz"%mouse_khz, (mousex_pos+4, self.TUNEBAR_Y-50), "small", False, "BLACK"),
                "auto": ((GREEN if fl.auto_mode else RED), "[AUTO]" if fl.auto_mode else "[MANU]", (wf_width/2+165, self.V_POS_TEXT), "small", False),
                "center": ((GREEN if fl.wf_snd_link_flag else GREY), "CENTER", (wf_width-145, self.SPECTRUM_Y+2), "small", False),
                "sync": ((GREEN if fl.cat_snd_link_flag else GREY), "SYNC", (40, self.BOTTOMBAR_Y+4), "big", False),
                "cat": (GREEN if cat_radio else GREY, "CAT", (5,self.BOTTOMBAR_Y+4), "big", False), 
                "recording": (RED if kiwi_snd.audio_rec.recording_flag and run_index%2 else D_GREY, "REC", (wf_width-90, self.BOTTOMBAR_Y+4), "big", False),
                "dxcluster": (GREEN if fl.show_dxcluster_flag else D_GREY, "DXCLUST", (wf_width-200, self.BOTTOMBAR_Y+4), "big", False),
                "utc": (ORANGE, datetime.utcnow().strftime(" %d %b %Y %H:%M:%SZ"), (wf_width-160, self.V_POS_TEXT), "small", False),
                "wf_bottom": (WHITE, "%ddB"%(kiwi_wf.wf_min_db), (0,self.TUNEBAR_Y-12), "small", False, "BLACK"),
                "wf_param": (WHITE, "%ddB AUTO %s"%(kiwi_wf.wf_max_db, "ON" if kiwi_wf.wf_auto_scaling else "OFF"), (0,self.SPECTRUM_Y+1), "small", False, "BLACK"),
                "help": (BLUE, "HELP", (wf_width-50, self.BOTTOMBAR_Y+4), "big", False)
                }

        if fl.dualrx_flag and kiwi_snd2:
            audio_balance_string_sub = audio_balance_string_list[int((kiwi_snd2.audio_balance+1)*2)]
            ts_dict["rx_freq2"] = (sub_rx_color, "SUB:%.3fkHz %s [%s]"%(kiwi_snd2.freq+(CW_PITCH if kiwi_snd2.radio_mode=="CW" else 0), kiwi_snd2.radio_mode, "MUTE" if kiwi_snd2.volume==0 else "%d%%"%kiwi_snd2.volume), (wf_width/2-410,self.V_POS_TEXT-1), "big", False)
            ts_dict["kiwi2"] = (GREEN if not kiwi_snd.subrx else RED, kiwi_host2[:30]+" - BAL: %s"%audio_balance_string_sub ,(350,self.BOTTOMBAR_Y+6), "small", False)
        if not fl.s_meter_show_flag:
            s_value = (round(rssi_smooth)+127)//6 # signal in S units of 6dB
            if s_value<=9:
                s_value = "S"+str(max(0,int(s_value)))
            else:
                s_value = "S9+"+str(int((s_value-9)*6))+"dB"
            ts_dict["smeter"] = (ORANGE if not tx_on_flag else "RED", s_value if not tx_on_flag else "TX", (20,self.V_POS_TEXT-1), "big", False)
        if fl.click_drag_flag:
            delta_khz = kiwi_wf.deltabins_to_khz(fl.start_drag_x*kiwi_wf.BINS2PIXEL_RATIO - mousex_pos)
            ts_dict["deltaf"] = (RED, ("+" if delta_khz>0 else "")+"%.1fkHz"%delta_khz, (wf_width/2,self.SPECTRUM_Y+20), "big", False)
        if kiwi_wf.averaging_n>1:
            ts_dict["avg"] = (RED, "AVG %dX"%kiwi_wf.averaging_n, (10,self.SPECTRUM_Y+13), "small", False)
        if len(kiwi_wf.div_list)>1:
            ts_dict["div"] = (YELLOW, "DIV :%.0fkHz"%(kiwi_wf.space_khz/10), (wf_width-95,self.SPECTRUM_Y+13), "small", False)
        else:
            ts_dict["div"] = (WHITE, "DIV :%.0fkHz"%(kiwi_wf.space_khz/100), (wf_width-95,self.SPECTRUM_Y+13), "small", False)

        draw_dict = {}
        for k in ts_dict:
            if k == "p_freq" and not (pygame.mouse.get_focused() and (self.WF_Y <= mouse[1] <= self.BOTTOMBAR_Y or self.TOPBAR_HEIGHT <= mouse[1] <= self.TUNEBAR_Y)):
                continue
            if "small" in ts_dict[k][3]:
                render_ = smallfont.render_to
            elif "big" in ts_dict[k][3]:
                render_ = bigfont.render_to
            try:
                bg_col = ts_dict[k][5]
            except:
                bg_col = None
            render_(surface_, ts_dict[k][2], ts_dict[k][1], ts_dict[k][0], bgcolor=bg_col)

    def draw_lines(self, surface_, wf_height, radio_mode, mouse, kiwi_wf, kiwi_snd, kiwi_snd2, fl, cat_radio):

        def _plot_bandpass(color_, kiwi_):
            snd_freq_bin = kiwi_wf.offset_to_bin(kiwi_.freq+kiwi_wf.span_khz/2-kiwi_wf.freq)
            if snd_freq_bin>0 and snd_freq_bin< kiwi_wf.WF_BINS:
                # carrier line
                pygame.draw.line(surface_, RED, (snd_freq_bin*kiwi_wf.BINS2PIXEL_RATIO, self.TUNEBAR_Y+self.TUNEBAR_HEIGHT/2), (snd_freq_bin*kiwi_wf.BINS2PIXEL_RATIO, self.TUNEBAR_Y+self.TUNEBAR_HEIGHT), 1)
            if cat_radio and not fl.cat_snd_link_flag:
                tune_freq_bin = kiwi_wf.offset_to_bin(kiwi_wf.tune+kiwi_wf.span_khz/2-kiwi_wf.freq)
                # tune wf line
                pygame.draw.line(surface_, D_RED, (tune_freq_bin*kiwi_wf.BINS2PIXEL_RATIO, self.TUNEBAR_Y+self.TUNEBAR_HEIGHT/2), (tune_freq_bin*kiwi_wf.BINS2PIXEL_RATIO, self.TUNEBAR_Y+self.TUNEBAR_HEIGHT), 3)
                
            lc_bin = kiwi_wf.offset_to_bin(kiwi_.lc/1000.)
            lc_bin = snd_freq_bin + lc_bin
            if lc_bin>0 and lc_bin< kiwi_wf.WF_BINS:
                # low cut line
                pygame.draw.line(surface_, color_, (lc_bin*kiwi_wf.BINS2PIXEL_RATIO, self.TUNEBAR_Y+self.TUNEBAR_HEIGHT/2), ((lc_bin-5)*kiwi_wf.BINS2PIXEL_RATIO, self.TUNEBAR_Y+self.TUNEBAR_HEIGHT), 1)
            
            hc_bin = kiwi_wf.offset_to_bin(kiwi_.hc/1000)
            hc_bin = snd_freq_bin + hc_bin
            if hc_bin>0 and hc_bin< kiwi_wf.WF_BINS:
                # high cut line
                pygame.draw.line(surface_, color_, (hc_bin*kiwi_wf.BINS2PIXEL_RATIO, self.TUNEBAR_Y+self.TUNEBAR_HEIGHT/2), ((hc_bin+5)*kiwi_wf.BINS2PIXEL_RATIO, self.TUNEBAR_Y+self.TUNEBAR_HEIGHT), 1)
            pygame.draw.line(surface_, color_, (lc_bin*kiwi_wf.BINS2PIXEL_RATIO, self.TUNEBAR_Y+self.TUNEBAR_HEIGHT/2), (hc_bin*kiwi_wf.BINS2PIXEL_RATIO, self.TUNEBAR_Y+self.TUNEBAR_HEIGHT/2), 2)

        center_freq_bin = kiwi_wf.offset_to_bin(kiwi_wf.span_khz/2)
        # center WF line
        pygame.draw.line(surface_, RED, (center_freq_bin*kiwi_wf.BINS2PIXEL_RATIO, self.WF_Y), (center_freq_bin*kiwi_wf.BINS2PIXEL_RATIO, self.WF_Y+6), 4)
        # mouse click_freq line
        if pygame.mouse.get_focused() and self.WF_Y <= mouse[1] <= self.BOTTOMBAR_Y:
            pygame.draw.line(surface_, RED, (mouse[0], self.TUNEBAR_Y), (mouse[0], self.BOTTOMBAR_Y), 1)
        elif pygame.mouse.get_focused() and self.TOPBAR_HEIGHT <= mouse[1] <= self.TUNEBAR_Y:
            pygame.draw.line(surface_, GREEN, (mouse[0], self.TOPBAR_HEIGHT+self.TUNEBAR_HEIGHT/2), (mouse[0], self.TUNEBAR_Y+self.TUNEBAR_HEIGHT+self.TUNEBAR_HEIGHT/2), 1)

        # SUB RX
        if fl.dualrx_flag and kiwi_snd2:
            _plot_bandpass(GREEN, kiwi_snd2)
        # MAIN RX        
        _plot_bandpass(RED, kiwi_snd)

        #### CAT RADIO bandpass
        if cat_radio and not fl.cat_snd_link_flag:
            tune_freq_bin = kiwi_wf.offset_to_bin(kiwi_wf.tune+kiwi_wf.span_khz/2-kiwi_wf.freq)
            lc_, hc_ = kiwi_wf.change_passband(delta_low, delta_high)
            lc_bin = kiwi_wf.offset_to_bin(lc_/1000.)
            lc_bin = tune_freq_bin + lc_bin + 1
            if lc_bin>0 and lc_bin< kiwi_wf.WF_BINS:
                # low cut line
                pygame.draw.line(surface_, ORANGE, (lc_bin*kiwi_wf.BINS2PIXEL_RATIO, self.TUNEBAR_Y+self.TUNEBAR_HEIGHT/2), ((lc_bin-5)*kiwi_wf.BINS2PIXEL_RATIO, self.TUNEBAR_Y+self.TUNEBAR_HEIGHT), 1)
            
            hc_bin = kiwi_wf.offset_to_bin(hc_/1000)
            hc_bin = tune_freq_bin + hc_bin
            if hc_bin>0 and hc_bin< kiwi_wf.WF_BINS:
                # high cut line
                pygame.draw.line(surface_, ORANGE, (hc_bin*kiwi_wf.BINS2PIXEL_RATIO, self.TUNEBAR_Y+self.TUNEBAR_HEIGHT/2), ((hc_bin+5)*kiwi_wf.BINS2PIXEL_RATIO, self.TUNEBAR_Y+self.TUNEBAR_HEIGHT), 1)
            pygame.draw.line(surface_, ORANGE, (lc_bin*kiwi_wf.BINS2PIXEL_RATIO, self.TUNEBAR_Y+self.TUNEBAR_HEIGHT/2), (hc_bin*kiwi_wf.BINS2PIXEL_RATIO, self.TUNEBAR_Y+self.TUNEBAR_HEIGHT/2), 2)

        # plot click and drag red horiz bar
        if fl.click_drag_flag:
            pygame.draw.line(surface_, RED, (fl.start_drag_x*kiwi_wf.BINS2PIXEL_RATIO, self.SPECTRUM_Y+10), (mouse[0], self.SPECTRUM_Y+10), 4)

        # plot tuning minor and major ticks
        for x in kiwi_wf.div_list:
            pygame.draw.line(surface_, YELLOW, (x*kiwi_wf.BINS2PIXEL_RATIO, self.TUNEBAR_Y+self.TUNEBAR_HEIGHT), (x*kiwi_wf.BINS2PIXEL_RATIO, self.TUNEBAR_Y+12), 3)
        for x in kiwi_wf.subdiv_list:
            pygame.draw.line(surface_, WHITE, (x*kiwi_wf.BINS2PIXEL_RATIO, self.TUNEBAR_Y+self.TUNEBAR_HEIGHT), (x*kiwi_wf.BINS2PIXEL_RATIO, self.TUNEBAR_Y+17), 1)


    def display_kiwi_box(self, screen, current_string_, kiwilist):
        size = 550
        y_size = size * 0.6
        rec_pos = ((screen.get_width() / 2) - size/2, (screen.get_height() / 2) - size/4)
        question = "Enter hostname [port] [password]"
        message = question + ": " + "".join(current_string_)
        pygame.draw.rect(screen, BLACK,
                       (rec_pos[0], rec_pos[1],
                        size,y_size), 0)
        pygame.draw.rect(screen, WHITE,
                       (rec_pos[0]-1, rec_pos[1]-1,
                        size+2,y_size+2), 1)
        pos = (rec_pos[0]+2, rec_pos[1]+y_size-20)
        smallfont.render_to(screen, pos, message, WHITE)
        message = "Choose stored Kiwi number or enter new one (port and password are optional)"
        smallfont.render_to(screen, (pos[0], pos[1]-20), message, WHITE)
        for i, kiwi in enumerate(kiwilist.mem_list):
            pos = (rec_pos[0]+2, rec_pos[1]+5+i*20)
            msg = "Kiwi server: %d -> %s:%d:%s"%(i, kiwi[0], kiwi[1], kiwi[2])
            smallfont.render_to(screen, pos, msg, GREY)


    def display_box(self, screen, message, size):
        pygame.draw.rect(screen, BLACK,
                       ((screen.get_width() / 2) - size/2,
                        (screen.get_height() / 2) - 12,
                        size,18), 0)
        pygame.draw.rect(screen, WHITE,
                       ((screen.get_width() / 2) - size/2+2,
                        (screen.get_height() / 2) - 14,
                        size+4,20), 1)
        if len(message) != 0:
            pos = ((screen.get_width() / 2) - size/2+5, (screen.get_height() / 2) - 10)
            smallfont.render_to(screen, pos, message, WHITE)

    def display_help_box(self, screen, message_list):
        font_size = font_size_dict["small"]

        window_size = 525
        pygame.draw.rect(screen, (0,0,0),
                       ((screen.get_width() / 2) - window_size/2,
                        (screen.get_height() / 2) - window_size/3,
                        window_size , window_size-150), 0)
        pygame.draw.rect(screen, (255,255,255),
                       ((screen.get_width() / 2) - window_size/2,
                        (screen.get_height() / 2) - window_size/3,
                        window_size,window_size-150), 1)

        if len(message_list) != 0:
            for ii, msg in enumerate(message_list):
                pos = (screen.get_width() / 2 - window_size/2 + font_size, 
                        screen.get_height() / 2-window_size/3 + ii*(font_size+1) + font_size)
                smallfont.render_to(screen, pos, msg, WHITE)

    def display_msg_box(self, screen, message, pos=None, color=WHITE):
        if not pos:
            pos = (screen.get_width() / 2 - 100, screen.get_height() / 2 - 10)
        if len(message) != 0:
            hugefont.render_to(screen, pos, message, color)


    def s_meter_draw(self, rssi_smooth, agc_threshold, agc_decay):
        s_meter_radius = 50.
        SMETER_XSIZE, SMETER_YSIZE = 2*s_meter_radius+20, s_meter_radius+20
        smeter_surface = pygame.Surface((SMETER_XSIZE, SMETER_YSIZE))

        s_meter_center = (s_meter_radius+10,s_meter_radius+8)
        alpha_rssi = rssi_smooth + 127
        alpha_rssi = -math.radians(alpha_rssi * 180/127.)-math.pi*1.02

        alpha_agc = agc_threshold + 127
        alpha_agc = -math.radians(alpha_agc * 180/127.)-math.pi*1.02

        def _coords_from_angle(angle, s_meter_radius_):
            x_ = s_meter_radius_ * math.cos(angle)
            y_ = s_meter_radius_ * math.sin(angle)
            s_meter_x = s_meter_center[0] + x_
            s_meter_y = s_meter_center[1] - y_
            return s_meter_x, s_meter_y
        
        s_meter_x, s_meter_y = _coords_from_angle(alpha_rssi, s_meter_radius * 0.95)
        agc_meter_x, agc_meter_y = _coords_from_angle(alpha_agc, s_meter_radius * 0.7)
        pygame.draw.rect(smeter_surface, YELLOW,
                       (s_meter_center[0]-60, s_meter_center[1]-58, SMETER_XSIZE, SMETER_YSIZE), 0)
        pygame.draw.rect(smeter_surface, BLACK,
                       (s_meter_center[0]-60, s_meter_center[1]-58, SMETER_XSIZE, SMETER_YSIZE), 3)
        
        angle_list = np.linspace(0.2, math.pi-0.2, 9)
        text_list = ["1", "3", "5", "7", "9", "+10", "+20", "+30", "+40"]
        for alpha_seg, msg in zip(angle_list, text_list[::-1]):
            text_x, text_y = _coords_from_angle(alpha_seg, s_meter_radius*0.8)
            nanofont.render_to(smeter_surface, (text_x-6, text_y-2), msg, D_GREY)

            seg_x, seg_y = _coords_from_angle(alpha_seg, s_meter_radius)
            color_ =  BLACK
            tick_rad = 2
            if alpha_seg < 1.4:
                color_ = RED
                tick_rad = 3
            pygame.draw.circle(smeter_surface, color_, (seg_x, seg_y), tick_rad)
        pygame.draw.circle(smeter_surface, D_GREY, s_meter_center, 4)

        pygame.draw.line(smeter_surface, BLACK, s_meter_center, (s_meter_x, s_meter_y), 2)
        pygame.draw.line(smeter_surface, BLUE, s_meter_center, (agc_meter_x, agc_meter_y), 2)
        str_rssi = "%ddBm"%rssi_smooth
        str_len = len(str_rssi)
        pos = (s_meter_center[0]+13, s_meter_center[1])
        microfont.render_to(smeter_surface, pos, str_rssi, BLACK)
        
        str_decay = "%.1fs" % (agc_decay/1000)
        str_len = len(str_decay)
        pos = (s_meter_center[0]-40, s_meter_center[1])
        microfont.render_to(smeter_surface, pos, str_decay, BLACK)
        
        return smeter_surface

    def plot_spectrum(self, sdrdisplay, kiwi_wf, t_avg=15, col=YELLOW, filled=False):
        spectrum_surf = pygame.Surface((kiwi_wf.WF_BINS, self.SPECTRUM_HEIGHT))
        pixarr = pygame.PixelArray(spectrum_surf)
        if not kiwi_wf.wf_auto_scaling:
            wf_dyn_range = kiwi_wf.wf_max_db-kiwi_wf.wf_min_db
            min_wf_10 = int(kiwi_wf.wf_min_db/10)*10
            max_wf_10 = int(kiwi_wf.wf_max_db/10)*10
            subdiv_list = [self.SPECTRUM_HEIGHT-1-int((v-kiwi_wf.wf_min_db)/wf_dyn_range * self.SPECTRUM_HEIGHT) for v in range(min_wf_10, max_wf_10, 10)]

        for x, v in enumerate(np.nanmean(kiwi_wf.wf_data.T[:,:t_avg], axis=1)):
            y = self.SPECTRUM_HEIGHT-1-int(v/255 * self.SPECTRUM_HEIGHT)
            if filled:
                pixarr[x,y:self.SPECTRUM_HEIGHT] = col
            else:
                pixarr[x,y] = col

            if not kiwi_wf.wf_auto_scaling and not x%3:
                for y_div in subdiv_list:
                    pixarr[x,y_div] = D_GREEN
        del pixarr
        if self.DISPLAY_WIDTH != kiwi_wf.WF_BINS:
            spectrum_surf = pygame.transform.smoothscale(spectrum_surf, (self.DISPLAY_WIDTH, self.SPECTRUM_HEIGHT))
        sdrdisplay.blit(spectrum_surf, (0, self.SPECTRUM_Y))

    def plot_eibi(self, surface_, eibi, kiwi_wf):
        y_offset = 0
        old_fbin = -100
        fontsize = font_size_dict["medium"]
        station_list = [ string_f_khz for f_khz in set(eibi.visible_stations) for string_f_khz in eibi.int_freq_dict[f_khz] ]
        sorted_station_list = sorted(station_list, key=float)
        shown_list = []
        for string_f_khz in sorted_station_list:
            station_record = eibi.station_freq_dict[string_f_khz]
            if station_record in shown_list:
                continue
            f_khz_float = float(string_f_khz)
            f_bin = int(kiwi_wf.offset_to_bin(f_khz_float-kiwi_wf.start_f_khz))
            shown_list.append(station_record)
            try:
                ts = (WHITE, station_record[3], (f_bin,self.WF_Y+20), "small")
            except:
                continue
            render_ = midfont.render_to
            str_len = len(ts[1])
            x, y = ts[2]
            if x>fontsize*str_len/2 and x<self.DISPLAY_WIDTH-10:
                if f_bin-old_fbin <= fontsize*str_len/2+5:
                    y_offset += fontsize
                else:
                    y_offset = 0
                old_fbin = f_bin
                try:
                    render_(surface_, ((x*kiwi_wf.BINS2PIXEL_RATIO-str_len*fontsize/2-2), y+y_offset), ts[1],  rotation=0, fgcolor=ts[0], bgcolor=(20,20,20))
                    pygame.draw.line(surface_, WHITE, (f_bin*kiwi_wf.BINS2PIXEL_RATIO, self.WF_Y), (f_bin*kiwi_wf.BINS2PIXEL_RATIO, self.WF_Y+20+y_offset), 1)
                except:
                    pass

    def plot_dxcluster(self, surface_, dxclust, kiwi_wf):
        now  = datetime.utcnow()        
        y_offset = 0
        old_fbin = -100
        fontsize = font_size_dict["medium"]

        for spot_id in dxclust.visible_stations:
            try:
                f_khz_float = float(dxclust.spot_dict[spot_id][1])
                f_bin = int(kiwi_wf.offset_to_bin(f_khz_float-kiwi_wf.start_f_khz))
                call = dxclust.spot_dict[spot_id][0]
                spot_utc = dxclust.spot_dict[spot_id][2]
                duration = now - spot_utc
                duration_in_s = duration.total_seconds()
                duration_normal = int(duration_in_s//dxclust.SPOT_TTL_BASETIME*dxclust.SPOT_TTL_BASETIME)
                color = dxclust.color_dict[duration_normal]
                ts = (color, call, (f_bin,self.WF_Y+20), "small")

                render_ = midfont.render_to
                str_len = len(ts[1])
                x, y = ts[2]
                if x>fontsize*str_len/2 and x<self.DISPLAY_WIDTH-10:
                    if f_bin-old_fbin <= fontsize*str_len/2+5:
                        y_offset += fontsize-2
                    else:
                        y_offset = 0
                    old_fbin = f_bin
                    render_(surface_, (x*kiwi_wf.BINS2PIXEL_RATIO-str_len*fontsize/2-2, y+y_offset%(self.WF_HEIGHT/2)), ts[1],  rotation=0, fgcolor=ts[0], bgcolor=(20,20,20))
                    pygame.draw.line(surface_, WHITE, (f_bin*kiwi_wf.BINS2PIXEL_RATIO, self.WF_Y), (f_bin*kiwi_wf.BINS2PIXEL_RATIO, self.WF_Y+20+y_offset%(self.WF_HEIGHT/2)), 1)
            except:
                pass

    def plot_beacons(self, surface_, beacon_project, kiwi_wf):
        y_offset = 0
        old_fbin = -100
        fontsize = font_size_dict["medium"]
        
        for band in beacon_project.freq_dict:
            if math.fabs(kiwi_wf.freq - beacon_project.freq_dict[band])<100:    
                f_khz_float = float(beacon_project.freq_dict[band])
                f_bin = int(kiwi_wf.offset_to_bin(f_khz_float-kiwi_wf.start_f_khz))
                ts = (GREEN, beacon_project.beacons_dict[band], (f_bin,(self.SPECTRUM_Y+self.TUNEBAR_Y)/2), "small")
                render_ = midfont.render_to
                str_len = len(ts[1])
                x, y = ts[2]
                if x>fontsize*str_len/2 and x<self.DISPLAY_WIDTH-10:
                    old_fbin = f_bin
                    render_(surface_, ((x*kiwi_wf.BINS2PIXEL_RATIO-str_len*fontsize/2-5), y), ts[1],  rotation=0, fgcolor=ts[0], bgcolor=(20,20,20))
                    pygame.draw.line(surface_, (0, 100, 0, 20), (f_bin*kiwi_wf.BINS2PIXEL_RATIO, self.TUNEBAR_Y-60), (f_bin*kiwi_wf.BINS2PIXEL_RATIO, y), 1)

    def splash_screen(self, sdrdisplay):
        font = pygame.font.Font(None, 50)
        sdrdisplay.fill((0, 0, 0))
        block = font.render(" - SUPERSDR - ", True, GREEN)
        rect = block.get_rect()
        rect = block.get_rect(center=(self.DISPLAY_WIDTH/2, self.DISPLAY_HEIGHT/2-90))
        sdrdisplay.blit(block, rect)
        block = font.render("...CONNECTING...", True, GREY)
        rect = block.get_rect()
        rect = block.get_rect(center=(self.DISPLAY_WIDTH/2, self.DISPLAY_HEIGHT/2))
        sdrdisplay.blit(block, rect)
        block = font.render("marco cogoni - IS0KYB", True, BLUE)
        rect = block.get_rect()
        rect = block.get_rect(center=(self.DISPLAY_WIDTH/2, self.DISPLAY_HEIGHT/2+90))
        sdrdisplay.blit(block, rect)
        pygame.display.flip()
        time.sleep(1)


class logger():
    def __init__(self, callsign):
        self.log_file = "log.sdr"
        self.qso_dict = defaultdict(set)
        try:
            self.qrz = QRZ("qrz_settings.cfg")
        except:
            pass
        self.check_qrzcom_flag = False
        self.check_previous_flag = True

    def qrzcom_lookup(self):
        try:
            result = self.qrz.callsign(self.entry_callsign.get().upper())
            comments_string = (result["fname"]+", "+result["addr2"]+", "+result["country"]).strip()
            self.entry_comments.insert(0, comments_string)
        except:
            pass
          
    def read_file(self):
        log_data = None
        try:
            with open(self.log_file, "r") as fd:
                log_data = fd.readlines()
        except:
            print("no logfile found!")
        if log_data:
            for idx, row in enumerate(log_data):
                els = row.split(";")
                if len(els)>1:
                    try:
                        qso_utc = datetime.strptime(els[0].strip(), "%d/%m/%Y %H:%M")
                        qso_callsign = els[1].strip()
                        qso_frequency = float(els[2].strip())
                        qso_mode = els[3].strip()
                        qso_power = float(els[4].strip())
                        qso_rst_his = els[5].strip()
                        qso_rst_mine = els[6].strip()
                        qso_comments = els[7].strip()
                        
                        self.qso_dict[qso_callsign].add((qso_utc, qso_frequency, qso_mode, qso_power, qso_rst_his, qso_rst_mine, qso_comments))
                    except:
                        print("QSO entry not readable at line: %d"%idx)

    def store_qso_to_file(self):
        qso_callsign = self.entry_callsign.get().upper()
        qso_utc = self.entry_utc.get()
        qso_frequency = self.entry_frequency.get()
        qso_mode = self.entry_mode.get()
        qso_power = self.entry_power.get()
        qso_rst_his = self.entry_rst_his.get()
        qso_rst_mine = self.entry_rst_mine.get()
        qso_comments = self.entry_comments.get()
        self.qso_dict[qso_callsign].add((datetime.strptime(qso_utc, "%d/%m/%Y %H:%M"), qso_frequency, qso_mode, qso_power, qso_rst_his, qso_rst_mine, qso_comments))
        with open(self.log_file, "a") as fd:
            qso_row_list = [qso_utc, qso_callsign, qso_frequency, qso_mode, qso_power, qso_rst_his, qso_rst_mine, qso_comments]
            qso_row_string = ";".join(qso_row_list)+"\n"
            fd.write(qso_row_string)

        # self.root.destroy()
        self.entry_callsign.focus()

    def log_popup(self, kiwi_snd):
        def assign_qrzcom_bool():
            self.check_qrzcom_flag = self.check_qrzcom_flag_tk.get()
        def assign_previous_bool():
            self.check_previous_flag = self.check_previous_flag_tk.get()
        def newcall_callback():
            self.entry_comments.delete(0, END)
            self.check_qrzcom_flag = self.check_qrzcom_flag_tk.get()
            self.check_previous_flag = self.check_previous_flag_tk.get()
            if self.check_qrzcom_flag:
                self.qrzcom_lookup()
            if self.check_previous_flag:
                self.previous_qso()

            self.entry_utc.delete(0, END)
            self.entry_utc.insert(END, datetime.utcnow().strftime("%d/%m/%Y %H:%M"))
            self.entry_frequency.delete(0, END)
            self.entry_frequency.insert(END, kiwi_snd.freq+(CW_PITCH if kiwi_snd.radio_mode=="CW" else 0))
            self.entry_mode.delete(0, END)
            self.entry_mode.insert(END, kiwi_snd.radio_mode)

        try:
            if 'normal' == self.root.state():
                self.root.focus_force()
                return
        except:
            try:
                if 'normal' == self.root_search.state():
                    return
            except:
                pass
        self.root = tkinter.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
        self.root.geometry("400x280+1500+400")
        self.root.resizable(False,False)
        self.root.title("Logger")
        self.root.bind('<Escape>', lambda event: self.root.destroy())

        self.main_dialog = tkinter.Frame(self.root)
        self.main_dialog.pack(side=TOP)
        l = tkinter.Label(self.main_dialog, text = "New QSO")
        l.config(font =("Mono", 14))
        l.pack(side=TOP)

        frame_top = tkinter.Frame(self.root, borderwidth=1)
        frame_top.pack(side=TOP)

        frame_left = tkinter.Frame(self.root, borderwidth=1)
        frame_left.pack(side=LEFT)
        frame_right = tkinter.Frame(self.root, borderwidth=1)
        frame_right.pack(side=RIGHT)

        self.check_qrzcom_flag_tk = tkinter.BooleanVar()
        self.check_qrzcom_flag_tk.set(self.check_qrzcom_flag)
        self.check_previous_flag_tk = tkinter.BooleanVar()
        self.check_previous_flag_tk.set(self.check_previous_flag)

        check_qrzcom = tkinter.Checkbutton(frame_top, text='QRZ.COM lookup', var=self.check_qrzcom_flag_tk, command = assign_qrzcom_bool) 
        check_previous = tkinter.Checkbutton(frame_top, text='LOG lookup', var=self.check_previous_flag_tk, command = assign_previous_bool) 

        self.main_dialog = tkinter.Frame(self.root)
        self.main_dialog.pack()
        check_qrzcom.pack(side=LEFT)
        check_previous.pack(side=RIGHT)

        label_callsign = tkinter.Label(frame_left, text="Callsign")
        self.entry_callsign = tkinter.Entry(frame_left)
        self.entry_callsign.focus()
        self.entry_callsign.bind('<FocusOut>', lambda event: newcall_callback())
        label_utc = tkinter.Label(frame_left, text="Date and Time (UTC)")
        self.entry_utc = tkinter.Entry(frame_left)
        label_frequency = tkinter.Label(frame_left, text="Frequency (kHz)")
        self.entry_frequency = tkinter.Entry(frame_left)
        label_mode = tkinter.Label(frame_left, text="Mode")
        self.entry_mode = tkinter.Entry(frame_left)
        label_rst_his = tkinter.Label(frame_right, text="RST (his)")
        self.entry_power = tkinter.Entry(frame_right)
        label_power = tkinter.Label(frame_right, text="Power (W)")
        self.entry_rst_his = tkinter.Entry(frame_right)
        label_rst_mine = tkinter.Label(frame_right, text="RST (Mine)")
        self.entry_rst_mine = tkinter.Entry(frame_right)
        label_comments = tkinter.Label(frame_right, text="Comments (Name, QTH, etc)")
        self.entry_comments = tkinter.Entry(frame_right, width=50)

        label_callsign.pack()
        self.entry_callsign.pack()
        label_utc.pack()
        self.entry_utc.pack()
        self.entry_utc.insert(END, datetime.utcnow().strftime("%d/%m/%Y %H:%M"))
        label_frequency.pack()
        self.entry_frequency.pack()
        self.entry_frequency.insert(END, kiwi_snd.freq+(CW_PITCH if kiwi_snd.radio_mode=="CW" else 0))
        label_mode.pack()
        self.entry_mode.pack()
        self.entry_mode.insert(END, kiwi_snd.radio_mode)
        label_power.pack()
        self.entry_power.pack()
        self.entry_power.insert(END, 100)
        label_rst_his.pack()
        self.entry_rst_his.pack()
        self.entry_rst_his.insert(END, 59 if kiwi_snd.radio_mode!="CW" else 599)
        label_rst_mine.pack()
        self.entry_rst_mine.pack()
        self.entry_rst_mine.insert(END, 59 if kiwi_snd.radio_mode!="CW" else 599)
        label_comments.pack()
        self.entry_comments.pack()

        self.b1 = tkinter.Button(frame_right, text = "Save QSO", command = self.store_qso_to_file)
        self.b2 = tkinter.Button(frame_left, text = "Cancel", command = self.root.destroy)

        self.b1.pack()
        self.b2.pack()

    def find_qso(self, qso_callsign):
        if len(qso_callsign)<3:
            return "call_too_short"
        call_list = list(self.qso_dict.keys())
        callsign_find_list = [True if key.find(qso_callsign)>-1 else False for i, key in enumerate(call_list)]
        index = 1
        qso_string_list = []
        if any(callsign_find_list):
            for i, qso_call_flag in enumerate(callsign_find_list):
                if qso_call_flag:
                    qso_list = self.qso_dict[call_list[i]]
                    for qso_record in sorted(qso_list, key = lambda x: x[0], reverse=True): # sort QSOs with the same call by datetime
                        qso_string = "%d. "%(index)+call_list[i]+"\n"+" - ".join([str(el.strftime("%d/%m/%Y %H:%M") if ii==0 else el)+("\n" if ii==0 else "") for ii, el in enumerate(qso_record)])+"\n"
                        qso_string_list.append(qso_string)
                        index += 1
        else:
            return "no_qso_found"
        return qso_string_list

    def prev_qso_callback(self):
        qso_string_list = self.find_qso(self.entry_callsign_search.get().upper())
        self.t.configure(state='normal')
        self.t.delete(1.0, END)
        self.t.insert(END, "Last QSO with %s:\n" % self.entry_callsign_search.get().upper())
        if qso_string_list == "no_qso_found":
            self.t.insert(END, "No QSOs found!")            
        elif qso_string_list == "call_too_short":
            self.t.insert(END, "Callsign too short!\n")
        else:
            for qso_string in qso_string_list:
                self.t.insert(END, qso_string)
        self.t.configure(state='disabled')

    def previous_qso(self):
        qso_string_list = self.find_qso(self.entry_callsign.get().upper())
        if qso_string_list == "no_qso_found" or qso_string_list == "call_too_short":
            return

        w = tkinter.Toplevel()
        w.wm_title("Previous QSOs")
        w.bind('<Escape>', lambda event: w.destroy())
        w.bind('<Return>', lambda event: w.destroy())
        w.bind('<Tab>', lambda event: w.destroy())
        w.lift()
        w.focus_force()
        w.grab_set()
        w.grab_release()
        frame_text = tkinter.Frame(w, borderwidth=1)
        frame_text.pack(fill=BOTH, expand=True)
        scrollbar = Scrollbar(frame_text)
        self.t = tkinter.Text(frame_text, height=15, width=55, yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.t.yview)         
        scrollbar.pack(side=RIGHT, fill=Y)
        self.t.pack(side="left")
        self.t.delete(1.0, END)
        self.t.insert(END, "Last QSO with %s:\n" % self.entry_callsign.get().upper())
        for qso_string in qso_string_list:
            self.t.insert(END, qso_string)
        self.t.configure(state='disabled')

    def search_popup(self, kiwi_snd):
        try:
            if 'normal' == self.root_search.state():
                self.root_search.focus_force()
                return
        except:
            try:
                if 'normal' == self.root.state():
                    return
            except:
                pass

        self.root_search = tkinter.Tk()
        self.root_search.protocol("WM_DELETE_WINDOW", self.root_search.destroy)
        self.root_search.geometry("400x400+1500+900")
        self.root_search.resizable(False,False)
        self.root_search.title("Search QSO")
        self.root_search.bind('<Escape>', lambda event: self.root_search.destroy())
        self.root_search.bind('<Return>', lambda event: self.prev_qso_callback())
        self.root_search.bind('<Tab>', lambda event: self.prev_qso_callback())
        self.main_dialog = tkinter.Frame(self.root_search)
        self.main_dialog.pack()

        l = tkinter.Label(self.root_search, text = "Search QSO")
        l.config(font =("Mono", 14))

        label_callsign_search = tkinter.Label(text="Callsign")
        self.entry_callsign_search = tkinter.Entry()
        self.entry_callsign_search.focus()

        l.pack()
        label_callsign_search.pack()
        self.entry_callsign_search.pack()

        frame_text = tkinter.Frame(self.root_search, borderwidth=1)
        frame_text.pack(fill=BOTH, expand=True)
        scrollbar = Scrollbar(frame_text)
        self.t = tkinter.Text(frame_text, height=15, width=55, yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.t.yview)         
        scrollbar.pack(side=RIGHT, fill=Y)
        self.t.pack(side="left")

        frame_bottom = tkinter.Frame(self.root_search, borderwidth=5)
        frame_bottom.pack(fill=BOTH, expand=True)
        self.b1 = tkinter.Button(master=frame_bottom, text = "Find!", command = self.prev_qso_callback)
        self.b2 = tkinter.Button(master=frame_bottom, text = "Cancel",
                    command = self.root_search.destroy)

        self.b1.pack(side=LEFT)
        self.b2.pack(side=RIGHT)
        frame_bottom.pack()


class beacons():
    beacon_calls = ["4U1UN","VE8AT","W6WX","KH6WO","ZL6B","VK6RBP","JA2IGY","RR9O",
    "VR2B","4S7B","ZS6DN","5Z4B","4X6TU","OH2B","CS3B","LU4AA","OA4B","YV5B"]
    bands = [14, 18, 21, 24, 28]
    freq_dict = {14:14100, 18:18110, 21:21150, 24:24930, 28:28200}
    beacons_dict = {14:"", 18:"", 21:"", 24:"", 28:""}

    def which_beacons(self):
        time_now = datetime.utcnow()
        delta_seconds = timedelta(minutes=time_now.minute%3, seconds=time_now.second).total_seconds()
        index = int(delta_seconds // 10)
        cycle = time_now.minute // 3
        self.beacons_dict = {}
        for i, band in enumerate(self.bands):
            self.beacons_dict[band] = self.beacon_calls[(index-i)]
        
# 4U1UN   YV5B    OA4B    LU4AA   CS3B    0:00:00 0:03:00 0:06:00 0:09:00 0:12:00 0:15:00 0:18:00 0:21:00 0:24:00 0:27:00
# VE8AT   4U1UN   YV5B    OA4B    LU4AA   0:00:10 0:03:10 0:06:10 0:09:10 0:12:10 0:15:10 0:18:10 0:21:10 0:24:10 0:27:10
# W6WX    VE8AT   4U1UN   YV5B    OA4B    0:00:20 0:03:20 0:06:20 0:09:20 0:12:20 0:15:20 0:18:20 0:21:20 0:24:20 0:27:20
# KH6WO   W6WX    VE8AT   4U1UN   YV5B    0:00:30 0:03:30 0:06:30 0:09:30 0:12:30 0:15:30 0:18:30 0:21:30 0:24:30 0:27:30
# ZL6B    KH6WO   W6WX    VE8AT   4U1UN   0:00:40 0:03:40 0:06:40 0:09:40 0:12:40 0:15:40 0:18:40 0:21:40 0:24:40 0:27:40
# VK6RBP  ZL6B    KH6WO   W6WX    VE8AT   0:00:50 0:03:50 0:06:50 0:09:50 0:12:50 0:15:50 0:18:50 0:21:50 0:24:50 0:27:50
# JA2IGY  VK6RBP  ZL6B    KH6WO   W6WX    0:01:00 0:04:00 0:07:00 0:10:00 0:13:00 0:16:00 0:19:00 0:22:00 0:25:00 0:28:00
# RR9O    JA2IGY  VK6RBP  ZL6B    KH6WO   0:01:10 0:04:10 0:07:10 0:10:10 0:13:10 0:16:10 0:19:10 0:22:10 0:25:10 0:28:10
# VR2B    RR9O    JA2IGY  VK6RBP  ZL6B    0:01:20 0:04:20 0:07:20 0:10:20 0:13:20 0:16:20 0:19:20 0:22:20 0:25:20 0:28:20
# 4S7B    VR2B    RR9O    JA2IGY  VK6RBP  0:01:30 0:04:30 0:07:30 0:10:30 0:13:30 0:16:30 0:19:30 0:22:30 0:25:30 0:28:30
# ZS6DN   4S7B    VR2B    RR9O    JA2IGY  0:01:40 0:04:40 0:07:40 0:10:40 0:13:40 0:16:40 0:19:40 0:22:40 0:25:40 0:28:40
# 5Z4B    ZS6DN   4S7B    VR2B    RR9O    0:01:50 0:04:50 0:07:50 0:10:50 0:13:50 0:16:50 0:19:50 0:22:50 0:25:50 0:28:50
# 4X6TU   5Z4B    ZS6DN   4S7B    VR2B    0:02:00 0:05:00 0:08:00 0:11:00 0:14:00 0:17:00 0:20:00 0:23:00 0:26:00 0:29:00
# OH2B    4X6TU   5Z4B    ZS6DN   4S7B    0:02:10 0:05:10 0:08:10 0:11:10 0:14:10 0:17:10 0:20:10 0:23:10 0:26:10 0:29:10
# CS3B    OH2B    4X6TU   5Z4B    ZS6DN   0:02:20 0:05:20 0:08:20 0:11:20 0:14:20 0:17:20 0:20:20 0:23:20 0:26:20 0:29:20
# LU4AA   CS3B    OH2B    4X6TU   5Z4B    0:02:30 0:05:30 0:08:30 0:11:30 0:14:30 0:17:30 0:20:30 0:23:30 0:26:30 0:29:30
# OA4B    LU4AA   CS3B    OH2B    4X6TU   0:02:40 0:05:40 0:08:40 0:11:40 0:14:40 0:17:40 0:20:40 0:23:40 0:26:40 0:29:40
# YV5B    OA4B    LU4AA   CS3B    OH2B    0:02:50 0:05:50 0:08:50 0:11:50 0:14:50 0:17:50 0:20:50 0:23:50 0:26:50 0:29:50
