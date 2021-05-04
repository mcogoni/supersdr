#!/usr/bin/env python3

import pygame
import pyaudio
from pygame.locals import *
import pygame, pygame.font, pygame.event, pygame.draw, string, pygame.freetype
from matplotlib import cm
import numpy as np

import sys
print (sys.version_info)
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
import socket
import time
import math
from collections import deque

from kiwi import wsclient
import mod_pywebsocket.common
from mod_pywebsocket.stream import Stream
from mod_pywebsocket.stream import StreamOptions

from optparse import OptionParser


# Pyaudio options
FORMAT = pyaudio.paInt16
CHANNELS = 1
AUDIO_RATE = 48000
KIWI_RATE = 12000
SAMPLE_RATIO = int(AUDIO_RATE/KIWI_RATE)
CHUNKS = 14
KIWI_SAMPLES_PER_FRAME = 512
FULL_BUFF_LEN = 20
VOLUME = 100

# Hardcoded values for most kiwis
MAX_FREQ = 30000. # 32000 # this should be dynamically set after connection
MAX_ZOOM = 14
WF_BINS  = 1024
WF_HEIGHT = 400

# SuperSDR constants
DISPLAY_WIDTH = WF_BINS
DISPLAY_HEIGHT = 450
V_POS_TEXT = 5
MIN_DYN_RANGE = 70. # minimum visual dynamic range in dB
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

# setup colormap from matplotlib
palRGB = cm.jet(range(256))[:,:3]*255

ALLOWED_KEYS = [K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, K_9]
ALLOWED_KEYS += [K_KP0, K_KP1, K_KP2, K_KP3, K_KP4, K_KP5, K_KP6, K_KP7, K_KP8, K_KP9]
ALLOWED_KEYS += [K_BACKSPACE, K_RETURN, K_ESCAPE, K_KP_ENTER]

HELP_MESSAGE_LIST = ["COMMANDS HELP",
        "",
        "- LEFT/RIGHT: move KIWI RX freq +/- 1kHz (+SHIFT: X10)",
        "- PAGE UP/DOWN: move WF freq +/- SPAN/2",
        "- UP/DOWN: zoom in/out by a factor 2X",
        "- U/L/C/A: switches to USB, LSB, CW, AM",
        "- J/K: change low/high cut of RX (SHIFT inverts)",
        "- O: resets passband to defaults",
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
        "- SHIFT+ESC: quits",
        "",
        "  --- 73 de marco/IS0KYB cogoni@gmail.com ---  "]

font_size_dict = {"small": 12, "big": 18}

# Approximate HF band plan from https://www.itu.int/en/ITU-R/terrestrial/broadcast/Pages/Bands.aspx
# and https://www.iaru-r1.org/reference/band-plans/hf-bandplan/

class filter():
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
    def __init__(self, host_, port_, pass_, zoom_, freq_):
        # kiwi hostname and port
        self.host = host_
        self.port = port_
        self.password = pass_
        print ("KiwiSDR Server: %s:%d" % (self.host, self.port))
        self.zoom = zoom_
        self.freq = freq_

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
        self.wf_data = np.zeros((WF_HEIGHT, WF_BINS))

        # connect to kiwi WF server
        print ("Trying to contact server...")
        try:
            self.socket = socket.socket()
            self.socket.connect((self.host, self.port))
        except:
            print ("Failed to connect")
            exit()   
        print ("Socket open...")

        self.start_stream()

    
    def start_stream(self):

        uri = '/%d/%s' % (int(time.time()), 'W/F')
        handshake_wf = wsclient.ClientHandshakeProcessor(self.socket, self.host, self.port)
        handshake_wf.handshake(uri)
        request_wf = wsclient.ClientRequest(self.socket)
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

    def receive_spectrum(self, white_flag=False):
        msg = self.wf_stream.receive_message()
        if msg and bytearray2str(msg[0:3]) == "W/F": # this is one waterfall line
            msg = msg[16:] # remove some header from each msg
            
            spectrum = np.ndarray(len(msg), dtype='B', buffer=msg).astype(np.float32) # convert from binary data to uint8
            wf = spectrum
            wf = -(255 - wf)  # dBm
            wf_db = wf - 13 # typical Kiwi wf cal
            dyn_range = (np.max(wf_db[1:-1])-np.min(wf_db[1:-1]))
            wf_color =  (wf_db - np.min(wf_db[1:-1]))
            # standardize the distribution between 0 and 1
            wf_color /= np.max(wf_color[1:-1])
            # clip extreme values
            wf_color = np.clip(wf_color, np.percentile(wf_color,CLIP_LOWP), np.percentile(wf_color, CLIP_HIGHP))
            # standardize again between 0 and 255
            wf_color -= np.min(wf_color[1:-1])
            # expand between 0 and 255
            wf_color /= (np.max(wf_color[1:-1])/255.)
            # avoid too bright colors with no signals
            wf_color *= (min(dyn_range, MIN_DYN_RANGE)/MIN_DYN_RANGE)
            # insert a full signal line to see freq/zoom changes
            if white_flag:
                wf_color = np.ones_like(wf_color)*255
            self.wf_data[-1,:] = wf_color
            self.wf_data[0:WF_HEIGHT-1,:] = self.wf_data[1:WF_HEIGHT,:]
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

        return self.freq

    def keepalive(self):
        self.wf_stream.send_message("SET keepalive")

    def close_connection(self):
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
        self.radio_mode = self.get_mode()

    def set_freq(self, freq_):
        if freq_ >= CAT_LOWEST_FREQ:
            self.socket.send(("F %d\n" % (freq_*1000)).encode())
            tmp = self.socket.recv(512).decode() # tbi implement verification of reply

    def set_mode(self, radio_mode_):
        self.socket.send(("+M %s 2400\n"%radio_mode_).encode())
        self.radio_mode = radio_mode_
        out = self.socket.recv(512) # tbi check reply

    def get_freq(self):
        self.socket.send("+f\n".encode())
        out = self.socket.recv(512) # tbi check reply
        self.freq = int(out.decode().split(" ")[1].split("\n")[0])/1000.
        return self.freq

    def get_mode(self):
        self.socket.send("m\n".encode())
        out = self.socket.recv(512) # tbi check reply
        self.radio_mode = out.decode().split("\n")[0]
        if self.radio_mode not in self.KNOWN_MODES:
            self.radio_mode = "USB" # defaults to USB if radio selects RTTY, FSK, etc
        return self.radio_mode

class kiwi_sound():
    def __init__(self, freq_, mode_, lc_, hc_, password_):
        # connect to kiwi server
        self.rssi = -127
        self.freq = freq_
        self.radio_mode = mode_
        self.lc, self.hc = lc_, hc_
        print ("Trying to contact server...")
        try:
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
        except:
            print ("Failed to connect to Kiwi audio stream")
            return None

    def set_mode_freq_pb(self):
        #print (self.radio_mode, self.lc, self.hc, self.freq)
        msg = 'SET mod=%s low_cut=%d high_cut=%d freq=%.3f' % (self.radio_mode.lower(), self.lc, self.hc, self.freq)
        self.stream.send_message(msg)

    def get_audio_chunk(self):
        #self.stream.send_message('SET keepalive')
        snd_buf = self.process_audio_stream()
        self.keepalive()
        return snd_buf

    def process_audio_stream(self):
        data = self.stream.receive_message()
        if data is None:
            return None
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
        try:
            self.stream.close_connection(mod_pywebsocket.common.STATUS_GOING_AWAY)
            self.socket.close()
        except Exception as e:
            print ("exception: %s" % e)


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
    return "USB" if f>10000 else "LSB"

def callback(in_data, frame_count, time_info, status):
    global audio_buffer, old_buffer
    samples_got = 0
    audio_buf_start_len = len(audio_buffer)
    while audio_buf_start_len+samples_got <= FULL_BUFF_LEN:
        snd_buf = kiwi_snd.get_audio_chunk()
        if snd_buf is not None:
            audio_buffer.append(snd_buf)
            samples_got += 1
        else:
            break
    # emergency buffer fillup with silence
    while len(audio_buffer) <= FULL_BUFF_LEN:
        print("!", end=' ')
        audio_buffer.append(np.zeros((KIWI_SAMPLES_PER_FRAME)))
    
    popped = np.array(audio_buffer[:CHUNKS]).flatten()
    popped = popped.astype(np.float64) * (VOLUME/100)
    audio_buffer = audio_buffer[CHUNKS:] # removed used chunks

    n = len(popped)
    # oversample
    pyaudio_buffer = np.zeros((SAMPLE_RATIO*n))
    pyaudio_buffer[::SAMPLE_RATIO] = popped
    pyaudio_buffer = np.concatenate([old_buffer, pyaudio_buffer])
    
    # low pass filter
    old_buffer = pyaudio_buffer[-(kiwi_filter.n_tap-1):]
    pyaudio_buffer = kiwi_filter.lowpass(pyaudio_buffer) * SAMPLE_RATIO

    return (pyaudio_buffer.astype(np.int16), pyaudio.paContinue)

def display_box(screen, message, size):
    smallfont = pygame.freetype.SysFont('Mono', 12)

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
        smallfont.render_to(sdrdisplay, pos, message, WHITE)

def display_help_box(screen, message_list):
    font_size = font_size_dict["small"]
    smallfont = pygame.freetype.SysFont('Mono', font_size)

    window_size = 450
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
                    screen.get_height() / 2-window_size/3 + ii*font_size + font_size)
            smallfont.render_to(sdrdisplay, pos, msg, WHITE)

def display_msg_box(screen, message, pos=None, fontsize=12, color=WHITE):
    smallfont = pygame.freetype.SysFont('Mono', fontsize)
    if not pos:
        pos = (screen.get_width() / 2 - 100, screen.get_height() / 2 - 10)
    # pygame.draw.rect(screen, BLACK,
    #                ((screen.get_width() / 2) - msg_len/2,
    #                 (screen.get_height() / 2) - 10, msg_len,20), 0)
    # pygame.draw.rect(screen, WHITE,
    #                ((screen.get_width() / 2) - msg_len/2+2,
    #                 (screen.get_height() / 2) - 12, msg_len+4,24), 1)
    if len(message) != 0:
        smallfont.render_to(sdrdisplay, pos, message, color)
    
def s_meter_draw(rssi_smooth):
    font_size = 8
    smallfont = pygame.freetype.SysFont('Mono', font_size)

    s_meter_radius = 50.
    s_meter_center = (140,s_meter_radius+8)
    alpha_rssi = rssi_smooth+127
    alpha_rssi = -math.radians(alpha_rssi* 180/127.)-math.pi

    def _coords_from_angle(angle, s_meter_radius_):
        x_ = s_meter_radius_ * math.cos(angle)
        y_ = s_meter_radius_ * math.sin(angle)
        s_meter_x = s_meter_center[0] + x_
        s_meter_y = s_meter_center[1] - y_
        return s_meter_x, s_meter_y
    
    s_meter_x, s_meter_y = _coords_from_angle(alpha_rssi, s_meter_radius* 0.95)
    pygame.draw.rect(sdrdisplay, YELLOW,
                   (s_meter_center[0]-60, s_meter_center[1]-58, 2*s_meter_radius+20,s_meter_radius+20), 0)
    pygame.draw.rect(sdrdisplay, BLACK,
                   (s_meter_center[0]-60, s_meter_center[1]-58, 2*s_meter_radius+20,s_meter_radius+20), 3)
    
    angle_list = np.linspace(0.4, math.pi-0.4, 9)
    text_list = ["1", "3", "5", "7", "9", "+10", "+20", "+30", "+40"]
    for alpha_seg, msg in zip(angle_list, text_list[::-1]):
        text_x, text_y = _coords_from_angle(alpha_seg, s_meter_radius*0.8)
        smallfont.render_to(sdrdisplay, (text_x-6, text_y-2), msg, D_GREY)

        seg_x, seg_y = _coords_from_angle(alpha_seg, s_meter_radius)
        color_ =  BLACK
        tick_rad = 2
        if alpha_seg < 1.4:
            color_ = RED
            tick_rad = 3
        pygame.draw.circle(sdrdisplay, color_, (seg_x, seg_y), tick_rad)
    pygame.draw.circle(sdrdisplay, D_GREY, s_meter_center, 4)

    pygame.draw.line(sdrdisplay, BLACK, s_meter_center, (s_meter_x, s_meter_y), 2)
    str_rssi = "%ddBm"%rssi_smooth
    smallfont = pygame.freetype.SysFont('Mono', 10)
    str_len = len(str_rssi)
    pos = (s_meter_center[0]+13, s_meter_center[1])
    smallfont.render_to(sdrdisplay, pos, str_rssi, BLACK)

def update_textsurfaces(radio_mode, rssi, mouse, wf_width):
    global sdrdisplay
    mousex_pos = mouse[0]
    if mousex_pos < 25:
        mousex_pos = 25
    elif mousex_pos >= DISPLAY_WIDTH - 80:
        mousex_pos = DISPLAY_WIDTH - 80

    #           Label   Color   Freq/Mode                       Screen position
    ts_dict = {"wf_freq": (GREEN, "%.2fkHz"%(kiwi_wf.freq if cat_snd_link_flag else kiwi_wf.freq), (wf_width/2-60,wf_height-12), "small", False),
            "left": (GREEN, "%.1f"%(kiwi_wf.start_f_khz) ,(0,wf_height-12), "small", False),
            "right": (GREEN, "%.1f"%(kiwi_wf.end_f_khz), (wf_width-50,wf_height-12), "small", False),
            "rx_freq": (GREY, "%.2fkHz %s"%(kiwi_snd.freq, kiwi_snd.radio_mode), (wf_width/2+55,V_POS_TEXT), "small", False),
            "kiwi": (GREY, ("kiwi:"+kiwi_wf.host)[:30] ,(230,V_POS_TEXT), "small", False),
            "span": (GREEN, "SPAN %.0fkHz"%(round(kiwi_wf.span_khz)), (wf_width-180,wf_height-12), "small", False),
            "filter": (GREEN, "FILT %.1fkHz"%((kiwi_snd.hc-kiwi_snd.lc)/1000.), (wf_width-270,wf_height-12), "small", False),
            "p_freq": (WHITE, "%dkHz"%mouse_khz, (mousex_pos, wf_height-25), "small", False),
            "auto": ((GREEN if auto_mode else RED), "[AUTO]", (wf_width/2+165, V_POS_TEXT), "small", False),
            "center": ((GREEN if wf_snd_link_flag else RED), "CENTER", (wf_width/2-20, V_POS_TEXT), "big", False),
            "sync": ((GREEN if cat_snd_link_flag else RED), "SYNC", (wf_width/2-75, V_POS_TEXT), "big", False)
    }
    if not s_meter_show_flag:
        ts_dict["smeter"] = (GREEN, "%.0fdBm"%rssi_smooth, (wf_width/2-370,V_POS_TEXT), "big", False)
    
    draw_dict = {}
    for k in ts_dict:
        if k == "p_freq" and not pygame.mouse.get_focused():
            continue
        if "small" in ts_dict[k][3]:
            smallfont = pygame.freetype.SysFont('Mono', 12)
            render_ = smallfont.render_to
        elif "big" in ts_dict[k][3]:
            bigfont = pygame.freetype.SysFont('Mono', 16)
            render_ = bigfont.render_to
        fontsize_ = font_size_dict[ts_dict[k][3]]

        str_len = len(ts_dict[k][1])
        x_r, y_r = ts_dict[k][2]
        if ts_dict[k][4]:
            pygame.draw.rect(sdrdisplay, D_GREY, (x_r-1, y_r-1, (str_len)*fontsize_*0.6, 14), 0)
        render_(sdrdisplay, ts_dict[k][2], ts_dict[k][1], ts_dict[k][0])

def draw_lines(surface_, wf_height, radio_mode, mouse):
    center_freq_bin = kiwi_wf.offset_to_bin(kiwi_wf.span_khz/2)
    pygame.draw.line(surface_, RED, (center_freq_bin, DISPLAY_HEIGHT-30), (center_freq_bin, DISPLAY_HEIGHT-20), 4)
    if pygame.mouse.get_focused():
        pygame.draw.line(surface_, (250,0,0), (mouse[0], DISPLAY_HEIGHT-20), (mouse[0], DISPLAY_HEIGHT), 1)

    snd_freq_bin = kiwi_wf.offset_to_bin(kiwi_snd.freq+kiwi_wf.span_khz/2-kiwi_wf.freq)
    if snd_freq_bin>0 and snd_freq_bin< WF_BINS:
        # carrier line
        pygame.draw.line(surface_, RED, (snd_freq_bin, DISPLAY_HEIGHT-20), (snd_freq_bin, DISPLAY_HEIGHT), 2)
    if cat_radio and not cat_snd_link_flag:
        tune_freq_bin = kiwi_wf.offset_to_bin(kiwi_wf.tune+kiwi_wf.span_khz/2-kiwi_wf.freq)
        # tune wf line
        pygame.draw.line(surface_, D_RED, (tune_freq_bin, DISPLAY_HEIGHT-20), (tune_freq_bin, DISPLAY_HEIGHT), 3)
        
    lc_bin = kiwi_wf.offset_to_bin(kiwi_snd.lc/1000.)
    lc_bin = snd_freq_bin + lc_bin
    if lc_bin>0 and lc_bin< WF_BINS:
        # low cut line
        pygame.draw.line(surface_, GREEN, (lc_bin, wf_height-30), (lc_bin-5, wf_height-16), 2)
    
    hc_bin = kiwi_wf.offset_to_bin(kiwi_snd.hc/1000)
    hc_bin = snd_freq_bin + hc_bin
    if hc_bin>0 and hc_bin< WF_BINS:
        # high cut line
        pygame.draw.line(surface_, GREEN, (hc_bin, wf_height-30), (hc_bin+5, wf_height-16), 2)
    
    pygame.draw.line(surface_, GREEN, (lc_bin, wf_height-30), (hc_bin, wf_height-30), 2)

    if cat_radio and not cat_snd_link_flag:
        lc_, hc_ = kiwi_wf.change_passband(delta_low, delta_high)
        lc_bin = kiwi_wf.offset_to_bin(lc_/1000.)
        lc_bin = tune_freq_bin + lc_bin + 1
        if lc_bin>0 and lc_bin< WF_BINS:
            # low cut line
            pygame.draw.line(surface_, YELLOW, (lc_bin, wf_height-30), (lc_bin-5, wf_height-16), 1)
        
        hc_bin = kiwi_wf.offset_to_bin(hc_/1000)
        hc_bin = tune_freq_bin + hc_bin
        if hc_bin>0 and hc_bin< WF_BINS:
            # high cut line
            pygame.draw.line(surface_, YELLOW, (hc_bin, wf_height-30), (hc_bin+5, wf_height-16), 1)
        pygame.draw.line(surface_, YELLOW, (lc_bin, wf_height-30), (hc_bin, wf_height-30), 2)


def start_audio_stream():
    global audio_buffer

    for k in range(FULL_BUFF_LEN*2):
       snd_buf = kiwi_snd.get_audio_chunk()
       if snd_buf is not None:
           audio_buffer.append(snd_buf)

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
                    stream_callback=callback)
    kiwi_audio_stream.start_stream()

    return play, kiwi_audio_stream


parser = OptionParser()
parser.add_option("-w", "--password", type=str,
                  help="KiwiSDR password", dest="kiwipassword", default="")
parser.add_option("-s", "--kiwiserver", type=str,
                  help="KiwiSDR server name", dest="kiwiserver", default='192.168.1.82')
parser.add_option("-p", "--kiwiport", type=int,
                  help="port number", dest="kiwiport", default=8073)
parser.add_option("-S", "--radioserver", type=str,
                  help="RTX server name", dest="radioserver", default=None)
parser.add_option("-P", "--radioport", type=int,
                  help="port number", dest="radioport", default=4532)
parser.add_option("-z", "--zoom", type=int,
                  help="zoom factor", dest="zoom", default=8)
parser.add_option("-f", "--freq", type=int,
                  help="center frequency in kHz", dest="freq", default=None)
                  
options = vars(parser.parse_args()[0])

kiwi_host = options['kiwiserver']
kiwi_port = options['kiwiport']
kiwi_password = options['kiwipassword']
freq = options['freq'] # this is the central freq in kHz
zoom = options['zoom'] 
radiohost = options['radioserver']
radioport = options['radioport']

if radiohost:
    try:
        cat_radio = cat(radiohost, radioport)
        cat_radio.get_freq()
        if cat_radio.freq > CAT_LOWEST_FREQ:
            freq = cat_radio.freq
            cat_radio.get_mode()
            radio_mode = cat_radio.radio_mode
        else:
            radio_mode = "USB"
            cat_radio = None
            print("CAT radio not detected!")
    except:
        cat_radio = None
        radio_mode = "USB"
        if not freq:
            freq = 14200
        print("CAT radio not detected!")
else:
    cat_radio = None
    if not freq:
        freq = 14200
    radio_mode = "USB"

print(freq)
kiwi_wf = kiwi_waterfall(kiwi_host, kiwi_port, kiwi_password, zoom, freq)
kiwi_snd = kiwi_sound(freq, radio_mode, 30, 3000, kiwi_password)

# init pygame basic objects
pygame.init()
sdrdisplay = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
wf_width = sdrdisplay.get_width()
wf_height = sdrdisplay.get_height()

i_icon = "icon.jpg"
icon = pygame.image.load(i_icon)
pygame.display.set_icon(icon)
pygame.display.set_caption("SuperSDR 1.1")
clock = pygame.time.Clock()
pygame.key.set_repeat(200, 50)

wf_quit = False

auto_mode = True
input_freq_flag = False
input_server_flag = False
show_help_flag =  False
s_meter_show_flag = True

input_new_server = None

rssi = -127
if kiwi_snd:
    rssi = kiwi_snd.rssi

current_string = []

kiwi_filter = filter(KIWI_RATE/2, AUDIO_RATE)
old_buffer = np.zeros((kiwi_filter.n_tap))
audio_buffer = []


play, kiwi_audio_stream = start_audio_stream()

kiwi_memory = memory()
kiwi_wf.set_freq_zoom(freq, zoom)
kiwi_snd.freq = freq
kiwi_snd.radio_mode = radio_mode
lc, hc = kiwi_snd.change_passband(delta_low, delta_high)
kiwi_snd.set_mode_freq_pb()

# Operating modes:
wf_cat_link_flag = True if cat_radio else False
wf_snd_link_flag = False
cat_snd_link_flag = True if cat_radio else False
print("SYNC OPTIONS:")
print("WF<>CAT", wf_cat_link_flag, "WF<>RX", wf_snd_link_flag, "CAT<>RX", cat_snd_link_flag)

rssi_maxlen = FULL_BUFF_LEN*2 # buffer length used to smoothen the s-meter
rssi_hist = deque(rssi_maxlen*[rssi], rssi_maxlen)
rssi_smooth = rssi
run_index = 0
run_index_automode = 0
show_bigmsg = None

while not wf_quit:
    run_index += 1

    click_freq = None
    manual_wf_freq = None
    manual_snd_freq = None
    manual_zoom = None
    manual_mode = None
    change_passband_flag = False
    wf_white_flag = False
    force_sync_flag = None

    rssi = kiwi_snd.rssi
    rssi_hist.append(rssi)
    mouse = pygame.mouse.get_pos()

    for event in pygame.event.get():
        mouse_khz = kiwi_wf.bins_to_khz(mouse[0])

        if event.type == pygame.KEYDOWN:
            before_help_flag = show_help_flag
            show_help_flag = False
            if not input_freq_flag and not input_server_flag:
                keys = pygame.key.get_pressed()
                fast_tune = True if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else False

                # Force SYNC WF to RX freq if no CAT, else WF and RX to CAT
                if keys[pygame.K_SPACE]:
                    force_sync_flag = True
                    show_bigmsg = "forcesync"
                    run_index_bigmsg = run_index

                # Center RX freq on WF
                if keys[pygame.K_z]:
                    wf_snd_link_flag = False if wf_snd_link_flag else True
                    show_bigmsg = "centertune"
                    run_index_bigmsg = run_index

                # Memory read/write, reset
                if keys[pygame.K_w]:
                    if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                        kiwi_memory.reset_all_mem()
                        show_bigmsg = "resetmemory"
                        run_index_bigmsg = run_index
                    else:
                        kiwi_memory.write_mem(kiwi_snd.freq, radio_mode, lc, hc)
                        show_bigmsg = "writememory"
                        run_index_bigmsg = run_index
                if keys[pygame.K_r]:
                    run_index_bigmsg = run_index
                    mem_tmp = kiwi_memory.restore_mem()
                    if mem_tmp:
                        click_freq, kiwi_snd.radio_mode, lc, hc = mem_tmp
                        show_bigmsg = "restorememory"
                    else:
                        show_bigmsg = "emptymemory"

                # KIWI RX passband change
                if keys[pygame.K_o]:
                    change_passband_flag = True
                    delta_low = 0
                    delta_high = 0
                if keys[pygame.K_j]:
                    change_passband_flag = True
                    delta = -100 if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else 100
                    delta_low += delta
                    if delta_low > 3000:
                        delta_low = 3000
                    elif delta_low < -3000:
                        delta_low = -3000
                if keys[pygame.K_k]:
                    change_passband_flag = True
                    delta = -100 if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else 100
                    delta_high += delta
                    if delta_high > 3000:
                        delta_high = 3000
                    elif delta_high < -3000:
                        delta_high = -3000.
                
                # KIWI RX volume UP/DOWN, Mute
                if keys[pygame.K_v]:
                    if VOLUME < 150:
                        VOLUME += 10
                    show_bigmsg = "VOLUME"
                    run_index_bigmsg = run_index
                if keys[pygame.K_b]:
                    if VOLUME > 0:
                        VOLUME -= 10
                    show_bigmsg = "VOLUME"
                    run_index_bigmsg = run_index
                if keys[pygame.K_m]:
                    if VOLUME > 0:
                        old_volume = VOLUME
                        VOLUME = 0
                    else:
                        VOLUME = old_volume
                    show_bigmsg = "VOLUME"
                    run_index_bigmsg = run_index

                # KIWI WF zoom
                if keys[pygame.K_DOWN]:
                    if kiwi_wf.zoom > 0:
                        manual_zoom = kiwi_wf.zoom - 1
                elif keys[pygame.K_UP]:
                    if kiwi_wf.zoom< MAX_ZOOM:
                        manual_zoom = kiwi_wf.zoom + 1

                # KIWI WF arrow step tune
                elif keys[pygame.K_LEFT]:
                    if not (keys[pygame.K_RCTRL] or keys[pygame.K_LCTRL]):
                        if kiwi_snd.radio_mode != "CW" and kiwi_wf.zoom < 10:
                            if not fast_tune:
                                manual_snd_freq = kiwi_snd.freq//1 if kiwi_snd.freq % 1 else kiwi_snd.freq//1 - 1
                            else:
                                manual_snd_freq = kiwi_snd.freq//1 - 10
                        else:
                            manual_snd_freq = (kiwi_snd.freq*10//1)/10 - (0.1 if not fast_tune else 1.0)
                elif keys[pygame.K_RIGHT]:
                    if not (keys[pygame.K_RCTRL] or keys[pygame.K_LCTRL]):                    
                        if kiwi_snd.radio_mode != "CW" and kiwi_wf.zoom < 10:
                            if not fast_tune:
                                manual_snd_freq = kiwi_snd.freq//1 + 1
                            else:
                                manual_snd_freq = kiwi_snd.freq//1 + 10
                        else:
                            manual_snd_freq = (kiwi_snd.freq*10//1)/10 + (0.1 if not fast_tune else 1.0)
                elif keys[pygame.K_PAGEDOWN]:
                    manual_wf_freq = kiwi_wf.freq - kiwi_wf.span_khz/2
                elif keys[pygame.K_PAGEUP]:
                    manual_wf_freq = kiwi_wf.freq + kiwi_wf.span_khz/2

                # KIWI RX mode change
                elif keys[pygame.K_u]:
                    auto_mode = False
                    manual_mode = "USB"
                elif keys[pygame.K_l]:
                    auto_mode = False
                    manual_mode = "LSB"
                elif keys[pygame.K_c]:
                    auto_mode = False
                    manual_mode = "CW"
                elif keys[pygame.K_a]:
                    auto_mode = False
                    manual_mode = "AM"

                # KIWI WF manual tuning
                elif keys[pygame.K_f]:
                    input_freq_flag = True
                    current_string = []

                # Show help
                elif keys[pygame.K_h]:
                    if not before_help_flag:
                        show_help_flag = True

                # S-meter show/hide
                elif keys[pygame.K_s] and (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]):
                    s_meter_show_flag = False if s_meter_show_flag else True
                
                elif keys[pygame.K_s]:
                    if cat_radio:
                        show_bigmsg = "cat_rx_sync"
                        run_index_bigmsg = run_index
                        cat_snd_link_flag = False if cat_snd_link_flag else True

                # Automatic mode change ON/OFF
                elif keys[pygame.K_x]:
                    show_bigmsg = "automode"
                    run_index_bigmsg = run_index
                    auto_mode = False if auto_mode else True
                    if auto_mode:
                        kiwi_snd.radio_mode = get_auto_mode(kiwi_snd.freq)
                        lc, hc = kiwi_snd.change_passband(delta_low, delta_high)
                        kiwi_snd.set_mode_freq_pb()
                        if cat_radio:
                            cat_radio.set_mode(kiwi_snd.radio_mode)

                # Quit SuperSDR
                elif keys[pygame.K_ESCAPE] and keys[pygame.K_LSHIFT]:
                    wf_quit = True

                elif keys[pygame.K_q]:# and (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]):
                    input_server_flag = True
                    current_string = []

            # manual frequency input
            else:
                pygame.key.set_repeat(0) # disabe key repeat
                inkey = event.key
                if inkey in ALLOWED_KEYS or input_server_flag:
                    if inkey == pygame.K_BACKSPACE:
                        current_string = current_string[0:-1]
                    elif inkey == pygame.K_RETURN:
                        current_string = "".join(current_string)
                        try:
                            if input_freq_flag:
                                manual_snd_freq = int(current_string)
                            elif input_server_flag:
                                input_new_server = current_string
                        except:
                            pass
                            #click_freq = kiwi_wf.freq
                        pygame.key.set_repeat(200, 50)
                    elif inkey == pygame.K_ESCAPE:
                        input_freq_flag = False if input_freq_flag else False
                        input_server_flag = False if input_server_flag else False

                        pygame.key.set_repeat(200, 50)
                        print("ESCAPE!")
                    else:
                        if len(current_string)<10 or input_server_flag:
                            current_string.append(chr(inkey))

        # Quit
        if event.type == pygame.QUIT:
            wf_quit = True
        # KIWI WF mouse zooming
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4: # mouse scroll up
                if kiwi_wf.zoom<MAX_ZOOM:
                    t_khz = kiwi_wf.bins_to_khz(mouse[0])
                    zoom_f = (t_khz+kiwi_wf.freq)/2
                    kiwi_wf.set_freq_zoom(zoom_f, kiwi_wf.zoom + 1)
            elif event.button == 5: # mouse scroll down
                if kiwi_wf.zoom>0:
                    t_khz = kiwi_wf.bins_to_khz(mouse[0])
                    zoom_f = kiwi_wf.freq + (kiwi_wf.freq-t_khz)
                    kiwi_wf.set_freq_zoom(zoom_f, kiwi_wf.zoom - 1)
            elif event.button == 1:
                kiwi_wf.zoom_to_span()
                kiwi_wf.start_freq()
                kiwi_wf.end_freq()
                click_freq = kiwi_wf.bins_to_khz(mouse[0])
                if kiwi_snd.radio_mode == "CW":
                    click_freq -= CW_PITCH # tune CW signal taking into account cw offset
    
    if input_server_flag and input_new_server:
        pygame.event.clear()
        input_text_list = input_new_server.rstrip().split(" ")
        # stop stream
        kiwi_audio_stream.stop_stream()
        kiwi_audio_stream.close()

        # close PyAudio
        play.terminate()

        kiwi_snd.close_connection()
        kiwi_wf.close_connection()

        if len(input_text_list) >= 1:
            new_host = input_text_list[0]
            new_port = int(kiwi_port)
            new_password = kiwi_password
        if len(input_text_list) >= 2:
            new_port = int(input_text_list[1])
        if len(input_text_list) == 3:
            new_password = input_text_list[2]
        
        print(input_text_list)
        try:
            kiwi_wf.__init__(new_host, new_port, new_password, zoom, freq)
            kiwi_snd.__init__(freq, radio_mode, 30, 3000, new_password)
            print("Changed server to: %s:%d" % (new_host,new_port))
            kiwi_host, kiwi_port, kiwi_password = new_host, new_port, new_password

            time.sleep(2)
            play, kiwi_audio_stream = start_audio_stream()
        except:
            kiwi_wf = kiwi_waterfall(kiwi_host, kiwi_port, kiwi_password, zoom, freq)
            kiwi_snd = kiwi_sound(freq, radio_mode, 30, 3000, kiwi_password)
            print("Reverted back to server: %s:%d" % (kiwi_host, kiwi_port))

        input_server_flag = False

    # Change KIWI RX PB: this can only affect the SND stream
    if change_passband_flag:
        lc, hc = kiwi_snd.change_passband(delta_low, delta_high)
        kiwi_snd.set_mode_freq_pb()

    if force_sync_flag:
        if cat_radio:
            kiwi_wf.set_freq_zoom(cat_radio.freq, kiwi_wf.zoom)
            kiwi_snd.radio_mode = get_auto_mode(kiwi_wf.freq)
            lc, hc = kiwi_snd.change_passband(delta_low, delta_high)
            kiwi_snd.freq = kiwi_wf.freq
            kiwi_snd.set_mode_freq_pb()
        else:
            kiwi_wf.set_freq_zoom(kiwi_snd.freq, kiwi_wf.zoom)
        force_sync_flag = False


    if input_freq_flag and manual_snd_freq:
        kiwi_wf.set_freq_zoom(manual_snd_freq, kiwi_wf.zoom)
        kiwi_snd.radio_mode = get_auto_mode(kiwi_snd.freq)
        lc, hc = kiwi_snd.change_passband(delta_low, delta_high)
        kiwi_snd.freq = kiwi_wf.freq
        kiwi_snd.set_mode_freq_pb()
        input_freq_flag = False
        wf_white_flag = True

    # Change KIWI RX mode
    if manual_mode:
        show_bigmsg = "changemode"
        run_index_bigmsg = run_index
        if kiwi_snd:
            kiwi_snd.radio_mode = manual_mode
            lc, hc = kiwi_snd.change_passband(delta_low, delta_high)
            kiwi_snd.set_mode_freq_pb()

    # Change frequency by KB or WF zoom: numbers, arrows and pgup/down
    if manual_snd_freq:
        #kiwi_wf.set_freq_zoom(manual_snd_freq, kiwi_wf.zoom) # this keeps the tuned frequency always in the middle of the WF
        if wf_snd_link_flag:
            kiwi_wf.set_freq_zoom(manual_snd_freq, kiwi_wf.zoom)
            kiwi_snd.freq = kiwi_wf.freq
            if auto_mode:
                kiwi_snd.radio_mode = get_auto_mode(kiwi_snd.freq)
                lc, hc = kiwi_snd.change_passband(delta_low, delta_high)
            kiwi_snd.set_mode_freq_pb()

        else:
            kiwi_snd.freq = manual_snd_freq
            kiwi_snd.set_mode_freq_pb()
            if auto_mode:
                kiwi_snd.radio_mode = get_auto_mode(kiwi_snd.freq)
                lc, hc = kiwi_snd.change_passband(delta_low, delta_high)

            if kiwi_snd.freq < kiwi_wf.start_f_khz:
                kiwi_wf.set_freq_zoom(kiwi_wf.start_f_khz, kiwi_wf.zoom)
                wf_white_flag = True
            elif kiwi_snd.freq > kiwi_wf.end_f_khz:
                kiwi_wf.set_freq_zoom(kiwi_wf.end_f_khz, kiwi_wf.zoom)
                wf_white_flag = True

    if manual_wf_freq:
        kiwi_wf.set_freq_zoom(manual_wf_freq, kiwi_wf.zoom)
        wf_white_flag = True

    if manual_zoom:
        kiwi_wf.set_freq_zoom(kiwi_snd.freq, manual_zoom) # for now, the arrow zoom will be centered on the SND freq
        kiwi_snd.freq = kiwi_wf.freq
        kiwi_snd.set_mode_freq_pb()
        wf_white_flag = True

    # Change KIWI SND frequency
    if click_freq:
        kiwi_snd.freq = click_freq
        if auto_mode:
            kiwi_snd.radio_mode = get_auto_mode(kiwi_snd.freq)
            lc, hc = kiwi_snd.change_passband(delta_low, delta_high)
        kiwi_snd.set_mode_freq_pb()
        if wf_snd_link_flag or show_bigmsg == "restorememory":
            kiwi_wf.set_freq_zoom(click_freq, kiwi_wf.zoom)

    if cat_radio and cat_snd_link_flag:
        if manual_mode:
            cat_radio.set_mode(kiwi_snd.radio_mode)
        elif click_freq or manual_snd_freq:
            if cat_radio.radio_mode != get_auto_mode(kiwi_snd.freq) and auto_mode:
                cat_radio.set_mode(kiwi_snd.radio_mode)
            cat_radio.set_freq(kiwi_snd.freq + (CW_PITCH if kiwi_snd.radio_mode=="CW" else 0.))
        else:
            cat_radio.get_freq()
            kiwi_snd.freq = cat_radio.freq - (CW_PITCH if kiwi_snd.radio_mode=="CW" else 0.)
            kiwi_snd.radio_mode = cat_radio.get_mode()
            lc, hc = kiwi_snd.change_passband(delta_low, delta_high)
            kiwi_snd.set_mode_freq_pb()
            if wf_cat_link_flag: # shift WF by half span when RX outside WF
                delta_f = (kiwi_snd.freq - kiwi_wf.freq)
                if abs(delta_f) < 5*kiwi_wf.span_khz:
                    if delta_f + kiwi_wf.span_khz/2 < 0:
                        kiwi_wf.set_freq_zoom(kiwi_wf.start_f_khz, kiwi_wf.zoom)
                        wf_white_flag = True
                    elif delta_f - kiwi_wf.span_khz/2 > 0:
                        kiwi_wf.set_freq_zoom(kiwi_wf.end_f_khz, kiwi_wf.zoom)
                        wf_white_flag = True
                else:
                    kiwi_wf.set_freq_zoom(cat_radio.freq, kiwi_wf.zoom)

    if cat_radio and wf_cat_link_flag and not cat_snd_link_flag: # shift WF by half span when CAT outside WF
        cat_radio.get_freq()
        cat_radio.get_mode()

        kiwi_wf.tune = cat_radio.freq - (CW_PITCH if kiwi_wf.radio_mode=="CW" else 0.)
        kiwi_wf.radio_mode = cat_radio.radio_mode

        delta_f = (cat_radio.freq - kiwi_wf.freq)
        if abs(delta_f) < 5*kiwi_wf.span_khz:
            if delta_f + kiwi_wf.span_khz/2 < 0:
                kiwi_wf.set_freq_zoom(kiwi_wf.start_f_khz, kiwi_wf.zoom)
            elif delta_f - kiwi_wf.span_khz/2 > 0:
                kiwi_wf.set_freq_zoom(kiwi_wf.end_f_khz, kiwi_wf.zoom)
        else:
            kiwi_wf.set_freq_zoom(cat_radio.freq, kiwi_wf.zoom)


#   plot horiz line to show time of freq change
    kiwi_wf.receive_spectrum(True if wf_white_flag else False)

    # clear the background with a uniform color
    pygame.draw.rect(sdrdisplay, (0,0,80), (0,0,DISPLAY_WIDTH,DISPLAY_HEIGHT), 0)
    pygame.draw.rect(sdrdisplay, (0,0,00), (0,DISPLAY_HEIGHT-16,DISPLAY_WIDTH,DISPLAY_HEIGHT), 0)


    surface = pygame.surfarray.make_surface(kiwi_wf.wf_data.T)
    surface.set_palette(palRGB)
    draw_lines(sdrdisplay, wf_height, kiwi_snd.radio_mode, mouse)
    sdrdisplay.blit(surface, (0, 20))
    update_textsurfaces(kiwi_snd.radio_mode, rssi_smooth, mouse, wf_width)

#    draw_textsurfaces(draw_dict, ts_dict, sdrdisplay)

    if input_freq_flag:
        question = "Freq (kHz)"
        display_box(sdrdisplay, question + ": " + "".join(current_string), 200)
    elif input_server_flag:
        question = "hostname[ port][ password]"
        display_box(sdrdisplay, question + ": " + "".join(current_string), 550)
    elif show_help_flag:
        display_help_box(sdrdisplay, HELP_MESSAGE_LIST)
    elif show_bigmsg:
        msg_color = WHITE
        if run_index - run_index_bigmsg > 25:
            show_bigmsg = None
        if "VOLUME" == show_bigmsg:
            msg_color = WHITE if VOLUME <= 100 else RED
            msg_text = "VOLUME: %d"%(VOLUME)+'%'
        elif "cat_rx_sync" == show_bigmsg:
            msg_text = "CAT<->RX SYNC "+("ON" if cat_snd_link_flag else "OFF")
        elif "forcesync" == show_bigmsg:
            msg_text = "Force SYNC WF<-RX" if not cat_radio else "Force SYNC WF & RX -> CAT"
        elif "automode" == show_bigmsg:
            msg_text = "AUTO MODE "+("ON" if auto_mode else "OFF")
        elif "changemode" == show_bigmsg:
            msg_text = kiwi_snd.radio_mode
        elif "writememory" == show_bigmsg:
            msg_text = "Stored Memory %d"% (len(kiwi_memory.mem_list)-1)
        elif "restorememory" == show_bigmsg:
            msg_text = "Restored Memory %d"% kiwi_memory.index
        elif "resetmemory" == show_bigmsg:
            msg_text = "Reset All Memories!"
        elif "emptymemory" == show_bigmsg:
            msg_text = "No Memories!"
        elif "centertune" == show_bigmsg:
            msg_text = "WF center tune mode " + ("ON" if wf_snd_link_flag else "OFF")

        display_msg_box(sdrdisplay, msg_text, pos=None, fontsize=35, color=msg_color)

    rssi_smooth = np.mean(list(rssi_hist)[15:20])
    if s_meter_show_flag:
        s_meter_draw(rssi_smooth)

    pygame.display.update()
    clock.tick(50)
    mouse = pygame.mouse.get_pos()

# stop stream
kiwi_audio_stream.stop_stream()
kiwi_audio_stream.close()

# close PyAudio
play.terminate()

pygame.quit()
kiwi_snd.close_connection()
kiwi_wf.close_connection()


