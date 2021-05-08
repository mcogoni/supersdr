import pygame
import pyaudio
import wave

from pygame.locals import *
import pygame, pygame.font, pygame.event, pygame.draw, string, pygame.freetype

from matplotlib import cm
import numpy as np

import threading, queue

import socket
import time
from datetime import datetime
import sys
if sys.version_info > (3,):
    buffer = memoryview
    def bytearray2str(b):
        return b.decode('ascii')
else:
    def bytearray2str(b):
        return str(b)

import random
import struct
import array
import math
from collections import deque, defaultdict


from kiwi import wsclient
import mod_pywebsocket.common
from mod_pywebsocket.stream import Stream
from mod_pywebsocket.stream import StreamOptions
from mod_pywebsocket._stream_base import ConnectionTerminatedException


# Pyaudio options
FORMAT = pyaudio.paInt16
CHANNELS = 1
AUDIO_RATE = 48000
KIWI_RATE = 12000
SAMPLE_RATIO = int(AUDIO_RATE/KIWI_RATE)
CHUNKS = 2 # 10 or more for remote kiwis
KIWI_SAMPLES_PER_FRAME = 512
FULL_BUFF_LEN = 10 # 16 or more for remote kiwis

# Hardcoded values for most kiwis
MAX_FREQ = 30000. # 32000 # this should be dynamically set after connection
MAX_ZOOM = 14
WF_BINS  = 1024

# SuperSDR constants
WF_HEIGHT = 400
DISPLAY_WIDTH = WF_BINS
TOPBAR_HEIGHT = 20
SPECTRUM_HEIGHT = 100
BOTTOMBAR_HEIGHT = 20
TUNEBAR_HEIGHT = 20
DISPLAY_HEIGHT = WF_HEIGHT + SPECTRUM_HEIGHT + TOPBAR_HEIGHT + BOTTOMBAR_HEIGHT + TUNEBAR_HEIGHT
TOPBAR_Y = 0
SPECTRUM_Y = TOPBAR_HEIGHT
TUNEBAR_Y = SPECTRUM_Y + SPECTRUM_HEIGHT
WF_Y = TUNEBAR_Y + TUNEBAR_HEIGHT
BOTTOMBAR_Y = WF_Y + WF_HEIGHT
V_POS_TEXT = 5
MIN_DYN_RANGE = 90. # minimum visual dynamic range in dB
CLIP_LOWP, CLIP_HIGHP = 40., 100 # clipping percentile levels for waterfall colors
TENMHZ = 10000 # frequency threshold for auto mode (USB/LSB) switch
CAT_LOWEST_FREQ = 100 # 100 kHz is OK for most radio
CW_PITCH = 0.6 # CW offset from carrier in kHz

# Initial KIWI receiver parameters
on=True # AGC auto mode
hang=False # AGC hang
thresh=-75 # AGC threshold in dBm
slope=6 # AGC slope decay
decay=4000 # AGC decay time constant
gain=50 # AGC manual gain
LOW_CUT_SSB=30 # Bandpass low end SSB
HIGH_CUT_SSB=3000 # Bandpass high end
LOW_CUT_CW=300 # Bandpass for CW
HIGH_CUT_CW=800 # High end CW
HIGHLOW_CUT_AM=6000 # Bandpass AM
delta_low, delta_high = 0., 0. # bandpass tuning

# predefined RGB colors
GREY = (200,200,200)
WHITE = (255,255,255)
BLACK = (0,0,0)
D_GREY = (50,50,50)
D_RED = (200,0,0)
D_BLUE = (0,0,200)
D_GREEN = (0,200,0)
RED = (255,0,0)
BLUE = (0,0,255)
GREEN = (0,255,0)
YELLOW = (200,180,0)
ORANGE = (255,140,0)

ALLOWED_KEYS = [K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, K_9]
ALLOWED_KEYS += [K_KP0, K_KP1, K_KP2, K_KP3, K_KP4, K_KP5, K_KP6, K_KP7, K_KP8, K_KP9]
ALLOWED_KEYS += [K_BACKSPACE, K_RETURN, K_ESCAPE, K_KP_ENTER]

HELP_MESSAGE_LIST = ["COMMANDS HELP",
        "",
        "- LEFT/RIGHT: move KIWI RX freq +/- 1kHz (+SHIFT: X10)",
        "- PAGE UP/DOWN: move WF freq +/- SPAN/2",
        "- UP/DOWN: zoom in/out by a factor 2X",
        "- U/L/C/A: switches to USB, LSB, CW, AM",
        "- J/K/O: change low/high cut of RX (SHIFT inverts), O resets",
        "- E: start audio recording",
        "- F: enter frequency with keyboard",
        "- W/R: Write/Restore quick memory (up to 10)",
        "- SHIFT+W: Deletes all stored memories",
        "- V/B: up/down volume 10%",
        "- M: mute/unmute",
        "- SHIFT+S: S-METER show/hide",
        "- S: SYNC CAT and KIWI RX ON/OFF",
        "- Z: Center KIWI RX, shift WF instead",
        "- SPACE: FORCE SYNC of WF to RX if no CAT, else all to CAT",
        "- X: AUTO MODE ON/OFF depending on amateur/broadcast band",
        "- H: displays this help window",
        "- I: displays EIBI DB labels",
        "- Q: switch to a different KIWI server",
        "- SHIFT+ESC: quits",
        "",
        "  --- 73 de marco/IS0KYB cogoni@gmail.com ---  "]

font_size_dict = {"small": 12, "big": 18}

class audio_recording():
    def __init__(self, filename_):
        self.filename = filename_
        self.audio_buffer = []

        self.frames = []
        self.recording_flag = False

    def start(self):
        print("start recording")
        self.recording_flag = True

    def stop(self, p_):
        print("stop recording")
        self.recording_flag = False
        self.save(p_)

    def save(self, p_):
        self.wave = wave.open(self.filename, 'wb')
        self.wave.setnchannels(CHANNELS)
        self.wave.setsampwidth(p_.get_sample_size(FORMAT))
        self.wave.setframerate(AUDIO_RATE)

        # process audio data here

        self.wave.writeframes(b''.join(self.audio_buffer))
        self.wave.close()
        self.recording = False

class dxcluster():
    def __init__(self, mycall_):
        self.mycall = mycall_
        host, port = 'dxfun.com', 8000
        self.server = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect()
    
    def connect(self):
        self.sock.connect(self.server)
        self.send(self.mycall)

    def send(self, msg):
        msg = msg + '\n'
        self.sock.send(msg.encode())

    def receive(self):
        msg = self.sock.recv(2048)
        return msg.decode("latin-1")

    def run(self, kiwi_snd):
        while not kiwi_snd.terminate:
            dx_cluster_msg = self.receive()
            print(dx_cluster_msg)
            time.sleep(5)


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

    def write_mem(self, freq, radio_mode, lc, hc):
        self.mem_list.append((freq, radio_mode, lc, hc))
    
    def restore_mem(self):
        if len(self.mem_list)>0:
            self.index -= 1
            self.index %= len(self.mem_list)
            return self.mem_list[self.index]
        else:
            return None
    
    def reset_all_mem(self):
        self.mem_list = deque([], 10)

class kiwi_waterfall():
    def __init__(self, host_, port_, pass_, zoom_, freq_, eibi):
        self.eibi = eibi
        # kiwi hostname and port
        self.host = host_
        self.port = port_
        self.password = pass_
        print ("KiwiSDR Server: %s:%d" % (self.host, self.port))
        self.zoom = zoom_
        self.freq = freq_

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
        self.counter, self.actual_freq = self.start_frequency_to_counter(self.start_f_khz)
        print ("Actual frequency:", self.actual_freq, "kHz")
        self.socket = None
        self.wf_stream = None
        self.wf_color = None
        self.wf_buffer = queue.Queue(maxsize=200)

        self.wf_data = np.zeros((WF_HEIGHT, WF_BINS))

        # connect to kiwi WF server
        print ("Trying to contact %s..."%self.host)
        try:
            self.socket = socket.socket()
            self.socket.connect((self.host, self.port))
        except:
            print ("Failed to connect")
            exit()   
        print ("Socket open...")

        self.start_stream()

        while True:
            msg = self.wf_stream.receive_message()
            if msg and bytearray2str(msg[0:3]) == "W/F":
                break

    def start_stream(self):

        uri = '/%d/%s' % (int(time.time()), 'W/F')
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
        print ("Waterfall data stream active...")

        # send a sequence of messages to the server, hardcoded for now
        # max wf speed, no compression
        msg_list = ['SET auth t=kiwi p=%s'%self.password, 'SET zoom=%d start=%d'%(self.zoom,self.counter),\
        'SET maxdb=-10 mindb=-110', 'SET wf_speed=4', 'SET wf_comp=0']
        for msg in msg_list:
            self.wf_stream.send_message(msg)
        print ("Starting to retrieve waterfall data...")

    def zoom_to_span(self):
            """return frequency span in kHz for a given zoom level"""
            assert(self.zoom >= 0 and self.zoom <= MAX_ZOOM)
            self.span_khz = MAX_FREQ / 2**self.zoom
            return self.span_khz

    def start_frequency_to_counter(self, start_frequency_):
        """convert a given start frequency in kHz to the counter value used in _set_zoom_start"""
        assert(start_frequency_ >= 0 and start_frequency_ <= MAX_FREQ)
        self.counter = round(start_frequency_/MAX_FREQ * 2**MAX_ZOOM * WF_BINS)
        start_frequency_ = self.counter * MAX_FREQ / WF_BINS / 2**MAX_ZOOM
        return self.counter, start_frequency_

    def start_freq(self):
        self.start_f_khz = self.freq - self.span_khz/2
        return self.start_f_khz

    def end_freq(self):
        self.end_f_khz = self.freq + self.span_khz/2
        return self.end_f_khz

    def offset_to_bin(self, offset_khz_):
        bins_per_khz_ = WF_BINS / self.span_khz
        return bins_per_khz_ * (offset_khz_)

    def bins_to_khz(self, bins_):
        bins_per_khz_ = WF_BINS / self.span_khz
        return (1./bins_per_khz_) * (bins_) + self.start_f_khz

    def receive_spectrum(self):
        msg = self.wf_stream.receive_message()
        if msg and bytearray2str(msg[0:3]) == "W/F": # this is one waterfall line
            msg = msg[16:] # remove some header from each msg
            
            spectrum = np.ndarray(len(msg), dtype='B', buffer=msg).astype(np.float32) # convert from binary data to uint8
            wf = spectrum
            wf = -(255 - wf)  # dBm
            wf_db = wf - 13 # typical Kiwi wf cal
            dyn_range = (np.max(wf_db[1:-1])-np.min(wf_db[1:-1]))
            self.wf_color = (wf_db - np.min(wf_db[1:-1]))
            # standardize the distribution between 0 and 1
            self.wf_color /= np.max(self.wf_color[1:-1])
            # clip extreme values
            self.wf_color = np.clip(self.wf_color, np.percentile(self.wf_color,CLIP_LOWP), np.percentile(self.wf_color, CLIP_HIGHP))
            # standardize again between 0 and 255
            self.wf_color -= np.min(self.wf_color[1:-1])
            # expand between 0 and 255
            self.wf_color /= (np.max(self.wf_color[1:-1])/255.)
            # avoid too bright colors with no signals
            self.wf_color *= (min(dyn_range, MIN_DYN_RANGE)/MIN_DYN_RANGE)
            # insert a full signal line to see freq/zoom changes
        self.keepalive()

    def set_freq_zoom(self, freq_, zoom_):
        self.freq = freq_
        self.zoom = zoom_
        self.zoom_to_span()
        self.start_freq()
        self.end_freq()
        if zoom_ == 0:
            print("zoom 0 detected!")
            self.freq = 15000
            self.start_freq()
            self.end_freq()
            self.span_khz = MAX_FREQ
            #self.zoom_to_span()
        else:
            if self.start_f_khz<0:
                self.freq -= self.start_f_khz
                self.start_freq()
                self.end_freq()
                self.zoom_to_span()

            if self.end_f_khz>MAX_FREQ:
                self.freq -= self.end_f_khz - MAX_FREQ
                self.start_freq()
                self.end_freq()
                self.zoom_to_span()
        self.counter, actual_freq = self.start_frequency_to_counter(self.start_f_khz)
        if zoom_>0 and actual_freq<=0:
            self.freq = self.zoom_to_span()
            self.start_freq()
            self.end_freq()
            self.counter, actual_freq = self.start_frequency_to_counter(self.start_f_khz)
        msg = "SET zoom=%d start=%d" % (self.zoom, self.counter)
        self.wf_stream.send_message(msg)
        self.eibi.get_stations(self.start_f_khz, self.end_f_khz)

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
        self.wf_data[-2,:] = self.wf_color

    def run(self):
        while not self.terminate:
            self.receive_spectrum()
            self.wf_data[-1,:] = self.wf_color
            self.wf_data[0:WF_HEIGHT-1,:] = self.wf_data[1:WF_HEIGHT,:]
        return


class kiwi_sound():
    def __init__(self, freq_, mode_, lc_, hc_, password_, kiwi_wf, kiwi_filter, audio_rec_):
        # connect to kiwi server
        self.kiwi_wf = kiwi_wf
        self.n_tap = kiwi_filter.n_tap
        self.lowpass = kiwi_filter.lowpass
        self.audio_rec_ = audio_rec_
        self.audio_buffer = queue.Queue(maxsize=FULL_BUFF_LEN)
        self.terminate = False

        self.old_buffer = np.zeros((self.n_tap))
        self.volume = 100
        
        self.rssi = -127
        self.freq = freq_
        self.radio_mode = mode_
        self.lc, self.hc = lc_, hc_
        print ("Trying to contact server...")
        try:
            #self.socket = kiwi_wf.socket
            self.socket = socket.socket()
            self.socket.connect((kiwi_wf.host, kiwi_wf.port)) # future: allow different kiwiserver for audio stream

            uri = '/%d/%s' % (int(time.time()), 'SND')
            handshake_snd = wsclient.ClientHandshakeProcessor(self.socket, kiwi_wf.host, kiwi_wf.port)
            handshake_snd.handshake(uri)
            request_snd = wsclient.ClientRequest(self.socket)
            request_snd.ws_version = mod_pywebsocket.common.VERSION_HYBI13
            stream_option_snd = StreamOptions()
            stream_option_snd.mask_send = True
            stream_option_snd.unmask_receive = False
            self.stream = Stream(request_snd, stream_option_snd)
            print ("Audio data stream active...")

            msg_list = ["SET auth t=kiwi p=%s"%password_, "SET mod=%s low_cut=%d high_cut=%d freq=%.3f" %
            (self.radio_mode.lower(), self.lc, self.hc, self.freq),
            "SET compression=0", "SET ident_user=SuperSDR","SET OVERRIDE inactivity_timeout=1000",
            "SET agc=%d hang=%d thresh=%d slope=%d decay=%d manGain=%d" % (on, hang, thresh, slope, decay, gain),
            "SET AR OK in=%d out=%d" % (KIWI_RATE, AUDIO_RATE)]
            
            for msg in msg_list:
                self.stream.send_message(msg)
            data = ""
            while "MSG" in data:
                data = self.stream.receive_message()

        except:
            print ("Failed to connect to Kiwi audio stream")
            raise

    def set_mode_freq_pb(self):
        #print (self.radio_mode, self.lc, self.hc, self.freq)
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
            snd_buf = None #= np.zeros((KIWI_SAMPLES_PER_FRAME)).astype(np.int16)
        return snd_buf

    def process_audio_stream(self):
        try:
            data = self.stream.receive_message()
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
            self.rssi = 0.1 * s_meter - 127
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

    def callback(self, in_data, frame_count, time_info, status):
        popped = []
        for _ in range(CHUNKS):
            popped.append( self.audio_buffer.get() )

        popped = np.array(popped).flatten()
        popped = popped.astype(np.float64) * (self.volume/100)        

        n = len(popped)
        # oversample
        pyaudio_buffer = np.zeros((SAMPLE_RATIO*n))
        pyaudio_buffer[::SAMPLE_RATIO] = popped
        pyaudio_buffer = np.concatenate([self.old_buffer, pyaudio_buffer])
        
        # low pass filter
        self.old_buffer = pyaudio_buffer[-(self.n_tap-1):]
        pyaudio_buffer = self.lowpass(pyaudio_buffer) * SAMPLE_RATIO
        if self.audio_rec_.recording_flag:
            self.audio_rec_.audio_buffer.append(pyaudio_buffer.astype(np.int16))

        return (pyaudio_buffer.astype(np.int16), pyaudio.paContinue)

    def run(self):
        while not self.terminate:
            snd_buf = self.get_audio_chunk()
            if snd_buf is not None:
                self.audio_buffer.put(snd_buf)
        return


class cat:
    def __init__(self, radiohost_, radioport_):
        self.KNOWN_MODES = {"USB", "LSB", "CW", "AM"}
        self.radiohost, self.radioport = radiohost_, radioport_
        print ("RTX rigctld server: %s:%d" % (self.radiohost, self.radioport))
        # create a socket to communicate with rigctld
        self.socket = socket.socket()
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

    def send_msg(self, msg):
        self.socket.send((msg+"\n").encode())
        out = self.socket.recv(512).decode() # tbi implement verification of reply
        if len(out)==0 or "RPRT -5" in out:
             self.cat_ok = False
             self.reply = None
        else:
            #print(out)
            self.reply = out        

    def set_freq(self, freq_):
        if freq_ >= CAT_LOWEST_FREQ:
            self.send_msg(("\\set_freq %d" % (freq_*1000)))
            #if self.reply: # to be verified!
            #    self.freq = freq_

    def set_mode(self, radio_mode_):
        self.send_msg(("\\set_mode %s 2400"%radio_mode_))
        if self.reply:
            self.radio_mode = radio_mode_

    def get_freq(self):
        self.send_msg("\\get_freq")
        if self.reply:
            self.freq = int(self.reply)/1000.
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
    rx_t = threading.Thread(target=kiwi_snd.run, daemon=True)
    rx_t.start()

    print("Filling audio buffer...")
    while kiwi_snd.audio_buffer.qsize()<FULL_BUFF_LEN and not kiwi_snd.terminate:
        pass

    if kiwi_snd.terminate:
        print("kiwi sound not started!")
        del kiwi_snd
        return (None, None)

    play = pyaudio.PyAudio()
    for i in range(play.get_device_count()):
        #print(play.get_device_info_by_index(i))
        if play.get_device_info_by_index(i)['name'] == "pulse":
            CARD_INDEX = i
        else:
            CARD_INDEX = None

    # open stream using callback (3)
    kiwi_audio_stream = play.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=AUDIO_RATE,
                    output=True,
                    output_device_index=CARD_INDEX,
                    frames_per_buffer= int(KIWI_SAMPLES_PER_FRAME*CHUNKS*SAMPLE_RATIO),
                    stream_callback=kiwi_snd.callback)
    kiwi_audio_stream.start_stream()

    return play, kiwi_audio_stream


class eibi_db():
    def __init__(self):
        try:
            with open("./eibi.csv", encoding="latin") as fd:
                data = fd.readlines()
        except:
            return None
        label_list = data[0].rstrip().split(";")
        self.station_dict = defaultdict(list)
        self.intfreq_dict = defaultdict(list)
        self.visible_stations = []

        for el in data[1:]:
            els = el.rstrip().split(";")
            self.intfreq_dict[int(round(float(els[0])))].append(float(els[0])) # store or each integer freqeuncy key all float freq in kHz
            self.station_dict[float(els[0])].append(els[1:]) # store all stations' data using the float freq in kHz as key (multiple)

        self.freq_set = set(self.intfreq_dict.keys())

    def get_stations(self, start_f, end_f):
        inters = set(range(int(start_f), int(end_f))) & self.freq_set
        self.visible_stations = []
        for intf in inters:
            record = self.intfreq_dict[intf]
            for f_khz in record:
                self.visible_stations.append(f_khz) #self.station_dict[f_khz])
        return self.visible_stations

    def get_names(self, f_khz):
        name_list = []
        for station_record in self.station_dict[f_khz]:
            name_list.append(station_record[3])
        return name_list


def create_cm(which):
    if which == "jet":
        # setup colormap from matplotlib
        colormap = cm.jet(range(256))[:,:3]*255
    elif which == "cutesdr":
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
    return colormap
