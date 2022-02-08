#!/usr/bin/env python3

import _thread
from optparse import OptionParser
from utils_supersdr import *

# initialize global flags class, this is not really that elegant, but gets the job done
fl = flags()

parser = OptionParser()
parser.add_option("-w", "--password", type=str,
                  help="KiwiSDR password", dest="kiwipassword", default=default_kiwi_password)
parser.add_option("-s", "--kiwiserver", type=str,
                  help="KiwiSDR server name", dest="kiwiserver", default="kiwisdr.local")
parser.add_option("-p", "--kiwiport", type=int,
                  help="port number", dest="kiwiport", default=default_kiwi_port)
parser.add_option("-S", "--radioserver", type=str,
                  help="RTX server name", dest="radioserver", default=None)
parser.add_option("-P", "--radioport", type=int,
                  help="port number", dest="radioport", default=4532)
parser.add_option("-z", "--zoom", type=int,
                  help="zoom factor", dest="zoom", default=8)
parser.add_option("-f", "--freq", type=int,
                  help="center frequency in kHz", dest="freq", default=None)
parser.add_option("-r", "--fps", type=int,
                  help="screen refresh rate", dest="refresh", default=23)
parser.add_option("-l", "--large", type=int,
                  help="screen horiz size in pixels (default 1024)", dest="winsize", default=1024)
parser.add_option("-b", "--buffer", type=int,
                  help="buffer size", dest="audio_buffer", default=10)
parser.add_option("-d", "--dual",
                  help="Activate Dual RX", action="store_true", dest="dualrx", default=False)
parser.add_option("-c", "--callsign", type=str,
                  help="DX CLUSTER Callsign", dest="callsign", default="")
parser.add_option("-m", "--colormap", type=str,
                  help="colormap for waterfall", dest="colormap", default="cutesdr")
                  

# sdrdisplay = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT), 
#     pygame.SCALED | pygame.RESIZABLE | pygame.DOUBLEBUF,vsync=1)
options = vars(parser.parse_args()[0])
disp = display_constants(options["winsize"])
if disp.DISPLAY_WIDTH == 1920:
    sdrdisplay = pygame.display.set_mode((disp.DISPLAY_WIDTH, disp.DISPLAY_HEIGHT), 
        pygame.DOUBLEBUF | pygame.FULLSCREEN,vsync=1)
else:
    sdrdisplay = pygame.display.set_mode((disp.DISPLAY_WIDTH, disp.DISPLAY_HEIGHT), 
        pygame.DOUBLEBUF,vsync=1)
wf_width = sdrdisplay.get_width()
wf_height = sdrdisplay.get_height()
i_icon = "icon.jpg"
icon = pygame.image.load(i_icon)
pygame.display.set_icon(icon)
pygame.display.set_caption("SuperSDR %s"%VERSION)
clock = pygame.time.Clock()
pygame.key.set_repeat(200, 50)

splash_screen(sdrdisplay, disp)
font = pygame.font.Font(None, 50)

FPS = options['refresh']
fl.dualrx_flag = options['dualrx']

CALLSIGN = options['callsign']
try:
    dxclust = dxcluster(CALLSIGN)
except:
    dxclust = None
eibi = eibi_db()

palRGB = create_cm(options["colormap"])

kiwi_host = options['kiwiserver']
kiwi_port = options['kiwiport']
kiwi_password = options['kiwipassword']
freq = options['freq'] # this is the central freq in kHz
zoom = options['zoom'] 
radiohost = options['radioserver']
radioport = options['radioport']

input_new_server = ""
if not kiwi_host:
    input_new_server = input("***\nNo KIWI specified!\nPlease enter: hostname [port] [password]\n")
    input_text_list = input_new_server.rstrip().split(" ")
    if len(input_text_list) >= 1:
        kiwi_host = input_text_list[0]
    if len(input_text_list) >= 2:
        kiwi_port = int(input_text_list[1])
    if len(input_text_list) == 3:
        kiwi_password = input_text_list[2]

if radiohost:
    try:
        cat_radio = cat(radiohost, radioport)
        cat_radio.get_freq()
        if cat_radio.freq > cat_radio.CAT_MIN_FREQ and cat_radio.freq < cat_radio.CAT_MAX_FREQ:
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

print(kiwi_host, kiwi_port, kiwi_password, zoom, freq)
kiwi_host2, kiwi_port2, kiwi_password2 = kiwi_host, kiwi_port, kiwi_password

#init KIWI WF and RX audio
kiwi_wf = None
while not kiwi_wf:
    print(kiwi_wf)
    try:
        kiwi_wf = kiwi_waterfall(kiwi_host, kiwi_port, kiwi_password, zoom, freq, eibi, disp)
    except:
        kiwi_address = ""
        complete = False
        while True:
            for evt in pygame.event.get():
                if evt.type == KEYDOWN:
                    if evt.unicode.isprintable():
                        kiwi_address += evt.unicode
                    elif evt.key == K_BACKSPACE:
                        kiwi_address = kiwi_address[:-1]
                    elif evt.key == K_RETURN:
                        complete = True
                        break
                elif evt.type == QUIT:
                    break
            sdrdisplay.fill((0, 0, 0))
            block = font.render("Enter KIWI address:port ->" + kiwi_address, True, (255, 255, 255))
            rect = block.get_rect()
            rect = block.get_rect(center=(disp.DISPLAY_WIDTH/2, disp.DISPLAY_HEIGHT/2))
            sdrdisplay.blit(block, rect)
            pygame.display.flip()
            if complete:
                print(kiwi_address)
                if ":" in kiwi_address:
                    kiwi_host, kiwi_port = kiwi_address.split(":")
                    kiwi_port = int(kiwi_port)
                else:
                    if len(kiwi_address)>0:
                        kiwi_host = kiwi_address

                print(kiwi_host, kiwi_port)
                break

wf_t = threading.Thread(target=kiwi_wf.run, daemon=True)
wf_t.start()

kiwi_snd = kiwi_sound(freq, radio_mode, 30, 3000, kiwi_password, kiwi_wf, options["audio_buffer"])
if not kiwi_snd:
    print("Server not ready")
    sys.exit()

kiwi_snd2 = None
if fl.dualrx_flag:
    time.sleep(2)
    try:
        kiwi_snd2 = kiwi_sound(freq, radio_mode, 30, 3000, kiwi_password2, kiwi_wf, kiwi_snd.FULL_BUFF_LEN, host_ = kiwi_host2, port_ = kiwi_port2, subrx_ = True)
    # kiwi_snd2.FULL_BUFF_LEN = options["audio_buffer"]
    except:
        fl.dualrx_flag = False
        print("Server not ready")

play, kiwi_audio_stream = start_audio_stream(kiwi_snd)
if not play:
    del kiwi_snd
    sys.exit("Chosen KIWI receiver is not ready!")

if fl.dualrx_flag:
    play2, kiwi_audio_stream2 = start_audio_stream(kiwi_snd2)
    if not play2:
        kiwi_snd2 = None

old_volume = kiwi_snd.volume

wf_quit = False

current_string = []

if dxclust:
    print(dxclust)
    dxclust.connect()
    dx_t = threading.Thread(target=dxclust.run, args=(kiwi_wf,), daemon=True)
    dx_t.start()
    dx_cluster_msg = True
else:
    dx_cluster_msg = False

kiwi_memory = memory()
kiwilist = kiwi_list()
kiwilist.load_from_disk()

kiwi_wf.set_freq_zoom(freq, zoom)
kiwi_snd.freq = freq
kiwi_snd.radio_mode = radio_mode
delta_low, delta_high = 0., 0. # bandpass tuning
lc, hc = kiwi_snd.change_passband(delta_low, delta_high)
kiwi_snd.set_mode_freq_pb()

beacon_project = beacons()

# Operating modes:
fl.wf_cat_link_flag = True if cat_radio else False
fl.wf_snd_link_flag = False
fl.cat_snd_link_flag = True if cat_radio else False
print("SYNC OPTIONS:")
print("WF<>CAT", fl.wf_cat_link_flag, "WF<>RX", fl.wf_snd_link_flag, "CAT<>RX", fl.cat_snd_link_flag)

rssi_maxlen = 10 # buffer length used to smoothen the s-meter
rssi_hist = deque(rssi_maxlen*[kiwi_snd.rssi], rssi_maxlen)
rssi_smooth = kiwi_snd.rssi
run_index = 0
run_index_bigmsg = 0

run_index_automode = 0
show_bigmsg = None
msg_text = ""
fl.click_drag_flag = False

check_time = datetime.utcnow()

while not wf_quit:
    run_index += 1

    click_freq = None
    manual_wf_freq = None
    manual_snd_freq = None
    manual_mode = None
    change_passband_flag = False
    force_sync_flag = None

    rssi = kiwi_snd.rssi
    rssi_hist.append(rssi)
    mouse = pygame.mouse.get_pos()

    for event in pygame.event.get():
        mouse_khz = kiwi_wf.bins_to_khz(mouse[0])

        if event.type == pygame.KEYDOWN:
            before_help_flag = fl.show_help_flag
            fl.show_help_flag = False
            if not fl.input_freq_flag and not fl.input_server_flag and not fl.input_callsign_flag:
                keys = pygame.key.get_pressed()
                mods = pygame.key.get_mods()

                # Force SYNC WF to RX freq if no CAT, else WF and RX to CAT
                if keys[pygame.K_SPACE]:
                    force_sync_flag = True
                    show_bigmsg = "forcesync"
                    run_index_bigmsg = run_index

                # Show EIBI labels
                if keys[pygame.K_i]:
                    fl.show_eibi_flag = False if fl.show_eibi_flag else True

                # Show realtime DX-CLUSTER labels
                if keys[pygame.K_d]:
                    if dxclust:
                        fl.show_dxcluster_flag = False if fl.show_dxcluster_flag else True
                        if fl.show_dxcluster_flag:
                            dxclust.terminate = False
                        else:
                            dxclust.terminate = True
                    else:
                        fl.input_callsign_flag = True
                        current_string = []

                # Center RX freq on WF
                if keys[pygame.K_z]:
                    fl.wf_snd_link_flag = False if fl.wf_snd_link_flag else True
                    force_sync_flag = True
                    show_bigmsg = "centertune"
                    run_index_bigmsg = run_index

                # Memory read/write, reset, save to/load from disk
                if keys[pygame.K_t]:
                        kiwi_memory.load_from_disk()
                        show_bigmsg = "loadmemorydisk"
                        run_index_bigmsg = run_index
                if keys[pygame.K_w]:
                    if event.mod & pygame.KMOD_SHIFT:
                        kiwi_memory.save_to_disk()
                        show_bigmsg = "savememorydisk"
                        run_index_bigmsg = run_index
                    else:
                        kiwi_memory.write_mem(kiwi_snd.freq, kiwi_snd.radio_mode, delta_low, delta_high)
                        show_bigmsg = "writememory"
                        run_index_bigmsg = run_index
                if keys[pygame.K_r]:
                    if event.mod & pygame.KMOD_SHIFT:
                        kiwi_memory.reset_all_mem()
                        show_bigmsg = "resetmemory"
                        run_index_bigmsg = run_index
                    else:
                        run_index_bigmsg = run_index
                        mem_tmp = kiwi_memory.restore_mem()
                        if mem_tmp:
                            click_freq, kiwi_snd.radio_mode, delta_low, delta_high = mem_tmp
                            print(click_freq, kiwi_snd.radio_mode, delta_low, delta_high)
                            show_bigmsg = "restorememory"
                        else:
                            show_bigmsg = "emptymemory"

                # KIWI RX passband change
                if keys[pygame.K_o]:
                    change_passband_flag = True
                    delta_low = 0
                    delta_high = 0

                elif keys[pygame.K_j]:
                    old_delta_low, old_delta_high = delta_low, delta_high
                    min_pb_flag = False
                    max_pb_flag = False
                    delta = 100 if (event.mod & pygame.KMOD_SHIFT) else -100
                    if kiwi_snd.radio_mode == "CW":
                        delta = int(delta/5)
                    if (event.mod & pygame.KMOD_CTRL):
                        delta_low += delta
                        delta_high -= delta if kiwi_snd.radio_mode != "AM" else -delta
                    else:
                        delta_low += delta

                    if kiwi_snd.radio_mode == "CW":
                        old_passband = (HIGH_CUT_CW+old_delta_high) - (LOW_CUT_CW + old_delta_low)
                        passband = (HIGH_CUT_CW+delta_high) - (LOW_CUT_CW + delta_low)
                        if passband < 50 and passband<old_passband:
                            min_pb_flag = True
                            delta_low, delta_high = old_delta_low, old_delta_high
                        elif passband > 1600 and passband>old_passband:
                            delta_low, delta_high = old_delta_low, old_delta_high
                            max_pb_flag = True
                    else:
                        old_passband = (HIGH_CUT_SSB+old_delta_high) - (LOW_CUT_SSB + old_delta_low)
                        passband = (HIGH_CUT_SSB+delta_high) - (LOW_CUT_SSB + delta_low)
                        if passband < 50 and passband<old_passband:
                            delta_low, delta_high = old_delta_low, old_delta_high
                            min_pb_flag = True
                        elif passband > 6000 and passband>old_passband:
                            delta_low, delta_high = old_delta_low, old_delta_high
                            max_pb_flag = True
                    if not min_pb_flag and not max_pb_flag:
                        change_passband_flag = True

                elif keys[pygame.K_k]:
                    old_delta_low, old_delta_high = delta_low, delta_high
                    min_pb_flag = False
                    max_pb_flag = False
                    delta = -100 if (event.mod & pygame.KMOD_SHIFT) else 100
                    if kiwi_snd.radio_mode == "CW":
                        delta = int(delta/5)
                    if (event.mod & pygame.KMOD_CTRL):
                        delta_low += delta
                        delta_high -= delta if kiwi_snd.radio_mode != "AM" else -delta
                    else:
                        delta_high += delta

                    if kiwi_snd.radio_mode == "CW":
                        old_passband = (HIGH_CUT_CW+old_delta_high) - (LOW_CUT_CW + old_delta_low)
                        passband = (HIGH_CUT_CW+delta_high) - (LOW_CUT_CW + delta_low)
                        if passband < 50 and passband<old_passband:
                            min_pb_flag = True
                            delta_low, delta_high = old_delta_low, old_delta_high
                        elif passband > 1600 and passband>old_passband:
                            delta_low, delta_high = old_delta_low, old_delta_high
                            max_pb_flag = True
                    else:
                        old_passband = (HIGH_CUT_SSB+old_delta_high) - (LOW_CUT_SSB + old_delta_low)
                        passband = (HIGH_CUT_SSB+delta_high) - (LOW_CUT_SSB + delta_low)
                        if passband < 50 and passband<old_passband:
                            delta_low, delta_high = old_delta_low, old_delta_high
                            min_pb_flag = True
                        elif passband > 6000 and passband>old_passband:
                            delta_low, delta_high = old_delta_low, old_delta_high
                            max_pb_flag = True
                    if not min_pb_flag and not max_pb_flag:
                        change_passband_flag = True

                # KIWI WF averaging INC/DEC
                if keys[pygame.K_g]:
                    if kiwi_wf.averaging_n < 100:
                        kiwi_wf.averaging_n += 1
                    show_bigmsg = "WFAVG"
                    run_index_bigmsg = run_index
                elif keys[pygame.K_h]:
                    if kiwi_wf.averaging_n > 1:
                        kiwi_wf.averaging_n -= 1
                    show_bigmsg = "WFAVG"
                    run_index_bigmsg = run_index

                # KIWI RX volume UP/DOWN, Mute
                if keys[pygame.K_v] and (mods & pygame.KMOD_SHIFT):
                    if run_index_bigmsg < run_index:
                        if kiwi_snd.volume > 0:
                            old_volume = kiwi_snd.volume
                            kiwi_snd.volume = 0
                        else:
                            kiwi_snd.volume = old_volume
                        show_bigmsg = "VOLUME"
                        run_index_bigmsg = run_index
                elif keys[pygame.K_v]:
                    if kiwi_snd.volume < 150:
                        kiwi_snd.volume += 10
                    show_bigmsg = "VOLUME"
                    run_index_bigmsg = run_index
                elif keys[pygame.K_b]:
                    if kiwi_snd.volume > 0:
                        kiwi_snd.volume -= 10
                    show_bigmsg = "VOLUME"
                    run_index_bigmsg = run_index
                
                if keys[pygame.K_3]:
                    kiwi_wf.wf_auto_scaling = False if kiwi_wf.wf_auto_scaling else True
                    kiwi_wf.delta_low_db, kiwi_wf.delta_high_db = 0, 0
                    
                # KIWI WF colormap dynamic range (lower limit)
                if keys[pygame.K_PERIOD] and (mods & pygame.KMOD_SHIFT):
                    if kiwi_wf.delta_low_db < 30:
                        kiwi_wf.delta_low_db += 1
                elif keys[pygame.K_COMMA] and (mods & pygame.KMOD_SHIFT):
                    if kiwi_wf.delta_low_db > -30:
                        kiwi_wf.delta_low_db -= 1
                # KIWI WF colormap dynamic range (upper limit)
                elif keys[pygame.K_PERIOD]:
                    if kiwi_wf.delta_high_db < 30:
                        kiwi_wf.delta_high_db += 1
                elif keys[pygame.K_COMMA]:
                    if kiwi_wf.delta_high_db > -30:
                        kiwi_wf.delta_high_db -= 1

                # KIWI WF zoom
                if keys[pygame.K_DOWN]:
                    if kiwi_wf.zoom > 0:
                        kiwi_wf.set_freq_zoom(kiwi_snd.freq, kiwi_wf.zoom - 1)
                        kiwi_wf.set_white_flag()
                elif keys[pygame.K_UP]:
                    if kiwi_wf.zoom < kiwi_wf.MAX_ZOOM:
                        kiwi_wf.set_freq_zoom(kiwi_snd.freq, kiwi_wf.zoom + 1)
                        kiwi_wf.set_white_flag()

                # KIWI WF arrow step tune
                if keys[pygame.K_LEFT]:
                    fast_tune = True if mods & pygame.KMOD_SHIFT else False
                    if not (mods & pygame.KMOD_CTRL):
                        if kiwi_snd.radio_mode != "CW" and kiwi_wf.zoom < 10:
                            if fast_tune:
                                manual_snd_freq = kiwi_snd.freq//1 - 10
                            else:
                                manual_snd_freq = kiwi_snd.freq//1 if kiwi_snd.freq % 1 else kiwi_snd.freq//1 - 1
                        else:
                            manual_snd_freq = ((kiwi_snd.freq)*10//1)/10 - (0.1 if not fast_tune else 1.0)
                elif keys[pygame.K_RIGHT]:
                    fast_tune = True if mods & pygame.KMOD_SHIFT else False
                    if not (mods & pygame.KMOD_CTRL):                    
                        if kiwi_snd.radio_mode != "CW" and kiwi_wf.zoom < 10:
                            if fast_tune:
                                manual_snd_freq = kiwi_snd.freq//1 + 10
                            else:
                                manual_snd_freq = kiwi_snd.freq//1 + 1
                        else:
                            manual_snd_freq = ((kiwi_snd.freq)*10//1)/10 + (0.1001 if not fast_tune else 1.0)
                
                if keys[pygame.K_PAGEDOWN]:
                    manual_wf_freq = kiwi_wf.freq - kiwi_wf.span_khz/4
                elif keys[pygame.K_PAGEUP]:
                    manual_wf_freq = kiwi_wf.freq + kiwi_wf.span_khz/4

                # KIWI RX mode change
                if keys[pygame.K_u]:
                    fl.auto_mode = False
                    manual_mode = "USB"
                elif keys[pygame.K_l]:
                    fl.auto_mode = False
                    manual_mode = "LSB"
                elif keys[pygame.K_c]:
                    fl.auto_mode = False
                    manual_mode = "CW"
                elif keys[pygame.K_a]:
                    fl.auto_mode = False
                    manual_mode = "AM"

                # KIWI WF manual tuning
                if keys[pygame.K_f]:
                    fl.input_freq_flag = True
                    current_string = []

                # WF fill spectrum ON/OFF
                if keys[pygame.K_4]:
                    disp.SPECTRUM_FILLED = False if disp.SPECTRUM_FILLED else True

                # Start/stop audio recording to file
                if keys[pygame.K_e]:
                    if not kiwi_snd.audio_rec.recording_flag:
                        kiwi_snd.audio_rec.start()
                        show_bigmsg = "start_rec"
                        run_index_bigmsg = run_index
                    else:
                        kiwi_snd.audio_rec.stop()
                        show_bigmsg = "stop_rec"
                        run_index_bigmsg = run_index

                # S-meter show/hide
                if keys[pygame.K_m]:
                    fl.s_meter_show_flag = False if fl.s_meter_show_flag else True
                
                if keys[pygame.K_s] and cat_radio:
                    show_bigmsg = "cat_rx_sync"
                    run_index_bigmsg = run_index
                    fl.cat_snd_link_flag = False if fl.cat_snd_link_flag else True
                    force_sync_flag = True

                # Automatic mode change ON/OFF
                if keys[pygame.K_x]:
                    show_bigmsg = "automode"
                    run_index_bigmsg = run_index
                    fl.auto_mode = False if fl.auto_mode else True
                    if fl.auto_mode:
                        kiwi_snd.radio_mode = get_auto_mode(kiwi_snd.freq)
                        lc, hc = kiwi_snd.change_passband(delta_low, delta_high)
                        kiwi_snd.set_mode_freq_pb()
                        if cat_radio:
                            cat_radio.set_mode(kiwi_snd.radio_mode)

                # Change AGC threshold for the current KIWI receiver
                if keys[pygame.K_1]:
                    if kiwi_snd.thresh>-135:
                        kiwi_snd.thresh -= 1
                        show_bigmsg = "agc"
                        run_index_bigmsg = run_index
                    if kiwi_snd:
                        kiwi_snd.set_agc_params()
                if keys[pygame.K_2]:
                    if kiwi_snd.thresh<-20:
                        kiwi_snd.thresh += 1
                        show_bigmsg = "agc"
                        run_index_bigmsg = run_index
                if kiwi_snd:
                        kiwi_snd.set_agc_params()

                # Tune SUB RX on same freq on WF center
                if keys[pygame.K_n]:
                    if kiwi_snd2:
                        kiwi_snd2.freq = kiwi_wf.freq
                        kiwi_snd2.set_mode_freq_pb()

                # Disable SUB RX
                if keys[pygame.K_y] and (mods & pygame.KMOD_SHIFT):
                    if kiwi_snd2:
                        if kiwi_snd.subrx:
                            kiwi_snd, kiwi_snd2 = kiwi_snd2, kiwi_snd
                        kiwi_snd2.terminate = True
                        kiwi_audio_stream2.stop()
                        time.sleep(1)
                        kiwi_snd2.close_connection()
                        kiwi_snd2.terminate = False
                        print("Second RX disabled!")

                        fl.dualrx_flag = False
                        show_bigmsg = "disable2rx"
                        run_index_bigmsg = run_index
                        kiwi_snd2 = None
                # Switch audio MAIN/SUB VFOs
                elif keys[pygame.K_y]:
                    if kiwi_snd2:
                        kiwi_snd, kiwi_snd2 = kiwi_snd2, kiwi_snd
                        if not cat_radio:
                            force_sync_flag = True
                        show_bigmsg = "switchab"
                        run_index_bigmsg = run_index
                    else:
                        try:
                            kiwi_snd2 = kiwi_sound(kiwi_snd.freq, kiwi_snd.radio_mode, 30, 3000, kiwi_password2, kiwi_wf, kiwi_snd.FULL_BUFF_LEN, host_ = kiwi_host2, port_ = kiwi_port2, subrx_ = True)
                            play2, kiwi_audio_stream2 = start_audio_stream(kiwi_snd2)
                            print("Second RX active!")
                            show_bigmsg = "enable2rx"
                            run_index_bigmsg = run_index
                            fl.dualrx_flag = True
                        except:
                            print("Server not ready")
                            kiwi_snd2 = None
                            fl.dualrx_flag = False
                            show_bigmsg = "disable2rx"
                            run_index_bigmsg = run_index

                # Quit SuperSDR
                if keys[pygame.K_ESCAPE] and keys[pygame.K_LSHIFT]:
                    wf_quit = True

                elif keys[pygame.K_q]:
                    fl.input_server_flag = True
                    current_string = []

            # manual frequency input
            else:
                pygame.key.set_repeat(0) # disabe key repeat
                inkey = event.key
                if inkey in ALLOWED_KEYS or fl.input_server_flag or fl.input_callsign_flag:
                    if inkey == pygame.K_BACKSPACE:
                        current_string = current_string[0:-1]
                    elif inkey == pygame.K_RETURN:
                        current_string = "".join(current_string)
                        try:
                            if fl.input_freq_flag:
                                manual_snd_freq = int(current_string)
                                if kiwi_snd.radio_mode == "CW":
                                    manual_snd_freq -= CW_PITCH # tune CW signal taking into account cw offset
                            elif fl.input_server_flag:
                                input_new_server = current_string
                            elif fl.input_callsign_flag:
                                CALLSIGN = current_string
                        except:
                            pass
                            #click_freq = kiwi_wf.freq
                        pygame.key.set_repeat(200, 50)
                    elif inkey == pygame.K_ESCAPE:
                        fl.input_freq_flag = False if fl.input_freq_flag else False
                        fl.input_server_flag = False if fl.input_server_flag else False
                        fl.input_callsign_flag = False if fl.input_callsign_flag else False

                        pygame.key.set_repeat(200, 50)
                        print("ESCAPE!")
                    else:
                        if len(current_string)<10 or fl.input_server_flag:
                            try:
                                current_string.append(chr(inkey).upper())
                            except:
                                pass

        # Quit
        if event.type == pygame.QUIT:
            wf_quit = True
        # KIWI WF mouse zooming
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4: # mouse scroll up
                if kiwi_wf.zoom<kiwi_wf.MAX_ZOOM:
                    t_khz = kiwi_wf.bins_to_khz(mouse[0])
                    zoom_f = (t_khz+kiwi_wf.freq)/2
                    kiwi_wf.set_freq_zoom(zoom_f, kiwi_wf.zoom + 1)
                    kiwi_wf.set_white_flag()
            elif event.button == 5: # mouse scroll down
                if kiwi_wf.zoom>0:
                    t_khz = kiwi_wf.bins_to_khz(mouse[0])
                    zoom_f = kiwi_wf.freq + (kiwi_wf.freq-t_khz)
                    kiwi_wf.set_freq_zoom(zoom_f, kiwi_wf.zoom - 1)
                    kiwi_wf.set_white_flag()
            elif event.button == 1:
                if disp.WF_Y <= mouse[1] <= disp.BOTTOMBAR_Y:
                    kiwi_wf.zoom_to_span()
                    kiwi_wf.start_freq()
                    kiwi_wf.end_freq()
                    click_freq = kiwi_wf.bins_to_khz(mouse[0]/kiwi_wf.BINS2PIXEL_RATIO)
                    if kiwi_snd.radio_mode == "CW":
                        click_freq -= CW_PITCH # tune CW signal taking into account cw offset
                if disp.SPECTRUM_Y <= mouse[1] <= disp.TUNEBAR_Y:
                    pygame.mouse.get_rel()
                    fl.start_drag_x = mouse[0]/kiwi_wf.BINS2PIXEL_RATIO
                    fl.click_drag_flag = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and fl.click_drag_flag:
                delta_x = pygame.mouse.get_rel()[0]/kiwi_wf.BINS2PIXEL_RATIO
                delta_freq = kiwi_wf.deltabins_to_khz(delta_x)
                manual_wf_freq = kiwi_wf.freq - delta_freq
                fl.click_drag_flag = False
    
    if mouse[0] > wf_width-50 and mouse[1] > disp.BOTTOMBAR_Y+4 and pygame.mouse.get_focused():
        fl.show_help_flag = True
    else:
        fl.show_help_flag = False

    if fl.input_callsign_flag and CALLSIGN != "":
        try:
            dxclust = dxcluster(CALLSIGN)
            if dxclust:
                dxclust.connect()
                dx_t = threading.Thread(target=dxclust.run, args=(kiwi_wf,), daemon=True)
                dx_t.start()
                dx_cluster_msg = True
                fl.show_dxcluster_flag = True
            else:
                dx_cluster_msg = False

        except:
            dxclust = None
        fl.input_callsign_flag = False

    if fl.input_server_flag and input_new_server:
        pygame.event.clear()
        if len(input_new_server) == 1:
            try:
                input_text_list = list(kiwilist.mem_list[int(input_new_server)])
            except:
                fl.input_server_flag = False
                input_new_server = None
                continue
        elif len(input_new_server) == 2:
            if input_new_server[0] == "d":
                try:
                    mem_index = int(input_new_server[1])
                    kiwilist.delete_mem(mem_index)
                    kiwilist.save_to_disk()
                except:
                    pass
            elif input_new_server[0] == "r":
                if not kiwi_snd2:
                    kiwi_host2, kiwi_port2, kiwi_password2 = list(kiwilist.mem_list[int(input_new_server[1])])

            fl.input_server_flag = False
            input_new_server = None
            continue
        else:
            input_text_list = input_new_server.rstrip().split(" ")

        # close audio stream
        kiwi_audio_stream.stop()
        if kiwi_snd2:
            kiwi_audio_stream2.stop()

        old_volume = kiwi_snd.volume
        kiwi_snd.terminate = True
        if kiwi_snd2:
            kiwi_snd2.terminate = True

        kiwi_wf.terminate = True
        time.sleep(1)

        kiwi_wf.close_connection()
        kiwi_snd.close_connection()
        if kiwi_snd2:
            kiwi_snd2.close_connection()

        if len(input_text_list) >= 1:
            new_host = input_text_list[0]
            new_port = default_kiwi_port
            new_password = default_kiwi_password
        if len(input_text_list) >= 2:
            new_port = int(input_text_list[1])
        if len(input_text_list) == 3:
            new_password = input_text_list[2]
        
        kiwi_snd.terminate = False
        if kiwi_snd2:
            kiwi_snd2.terminate = False
            fl.dualrx_flag = False
        kiwi_wf.terminate = False

        try:
            kiwi_wf.__init__(new_host, new_port, new_password, zoom, freq, eibi, disp)
            kiwi_snd.__init__(freq, radio_mode, 30, 3000, new_password, kiwi_wf, kiwi_snd.FULL_BUFF_LEN, volume_ = old_volume)
            print("Changed server to: %s:%d" % (new_host,new_port))
            play, kiwi_audio_stream = start_audio_stream(kiwi_snd)
        except:
            print ("something went wrong...")
            play = None
        if not play:
            kiwi_wf = kiwi_waterfall(kiwi_host, kiwi_port, kiwi_password, zoom, freq, eibi, disp)
            kiwi_snd = kiwi_sound(freq, radio_mode, 30, 3000, kiwi_password, kiwi_wf, kiwi_snd.FULL_BUFF_LEN, volume_ = old_volume)
            print("Reverted back to server: %s:%d" % (kiwi_host, kiwi_port))
            play, kiwi_audio_stream = start_audio_stream(kiwi_snd)
            if not play:
                exit("Old kiwi server not available anymore!")
        else:
            kiwi_host, kiwi_port, kiwi_password = new_host, new_port, new_password
            kiwilist.write_mem(kiwi_host, kiwi_port, kiwi_password)
            kiwilist.save_to_disk()


        kiwi_snd2 = None

        wf_t = threading.Thread(target=kiwi_wf.run, daemon=True)
        wf_t.start()
            
        fl.input_server_flag = False
        input_new_server = None

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
        kiwi_wf.set_white_flag()

    if fl.input_freq_flag and manual_snd_freq:
        kiwi_wf.set_freq_zoom(manual_snd_freq, kiwi_wf.zoom)
        if fl.auto_mode:
            kiwi_snd.radio_mode = get_auto_mode(manual_snd_freq)
            lc, hc = kiwi_snd.change_passband(delta_low, delta_high)
        kiwi_snd.freq = kiwi_wf.freq
        kiwi_snd.set_mode_freq_pb()
        fl.input_freq_flag = False
        kiwi_wf.set_white_flag()
        if dxclust:
            dxclust.update_now = True

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
        if fl.wf_snd_link_flag:
            kiwi_wf.set_freq_zoom(manual_snd_freq, kiwi_wf.zoom)
            kiwi_snd.freq = kiwi_wf.freq
            if fl.auto_mode:
                kiwi_snd.radio_mode = get_auto_mode(kiwi_snd.freq)
                lc, hc = kiwi_snd.change_passband(delta_low, delta_high)
            kiwi_snd.set_mode_freq_pb()
        else:
            kiwi_snd.freq = manual_snd_freq
            kiwi_snd.set_mode_freq_pb()
            if fl.auto_mode:
                kiwi_snd.radio_mode = get_auto_mode(kiwi_snd.freq)
                lc, hc = kiwi_snd.change_passband(delta_low, delta_high)

            if kiwi_snd.freq < kiwi_wf.start_f_khz:
                kiwi_wf.set_freq_zoom(kiwi_wf.start_f_khz, kiwi_wf.zoom)
                kiwi_wf.set_white_flag()
            elif kiwi_snd.freq > kiwi_wf.end_f_khz:
                kiwi_wf.set_freq_zoom(kiwi_wf.end_f_khz, kiwi_wf.zoom)
                kiwi_wf.set_white_flag()

    if manual_wf_freq:
        kiwi_wf.set_freq_zoom(manual_wf_freq, kiwi_wf.zoom)
        kiwi_wf.set_white_flag()

    # Change KIWI SND frequency
    if click_freq:
        kiwi_snd.freq = click_freq
        if fl.wf_snd_link_flag or show_bigmsg == "restorememory":
            kiwi_wf.set_freq_zoom(click_freq, kiwi_wf.zoom)
            lc, hc = kiwi_snd.change_passband(delta_low, delta_high)
            kiwi_wf.set_white_flag()
        elif fl.auto_mode:
            kiwi_snd.radio_mode = get_auto_mode(kiwi_snd.freq)
            lc, hc = kiwi_snd.change_passband(delta_low, delta_high)
        kiwi_snd.set_mode_freq_pb()

    if cat_radio and fl.cat_snd_link_flag:
        if manual_mode:
            cat_radio.set_mode(kiwi_snd.radio_mode)
        elif click_freq or manual_snd_freq:
            cat_radio.set_freq(kiwi_snd.freq + (CW_PITCH if kiwi_snd.radio_mode=="CW" else 0.))
            if (cat_radio.radio_mode != get_auto_mode(kiwi_snd.freq) and fl.auto_mode) or show_bigmsg == "restorememory":
                cat_radio.set_mode(kiwi_snd.radio_mode)
        else:
            kiwi_snd.radio_mode = cat_radio.get_mode()
            lc, hc = kiwi_snd.change_passband(delta_low, delta_high)
            kiwi_snd.set_mode_freq_pb()

            old_cat_freq = cat_radio.freq
            cat_radio.get_freq()
            if cat_radio.freq != old_cat_freq:
                kiwi_snd.freq = cat_radio.freq - (CW_PITCH if kiwi_snd.radio_mode=="CW" else 0.)
                if fl.wf_cat_link_flag: # shift WF by half span when RX outside WF
                    delta_f = (kiwi_snd.freq - kiwi_wf.freq)
                    if abs(delta_f) < 5*kiwi_wf.span_khz:
                        if delta_f + kiwi_wf.span_khz/2 < 0:
                            kiwi_wf.set_freq_zoom(kiwi_wf.start_f_khz, kiwi_wf.zoom)
                            kiwi_wf.set_white_flag()
                        elif delta_f - kiwi_wf.span_khz/2 > 0:
                            kiwi_wf.set_freq_zoom(kiwi_wf.end_f_khz, kiwi_wf.zoom)
                            kiwi_wf.set_white_flag()
                    else:
                        kiwi_wf.set_freq_zoom(cat_radio.freq, kiwi_wf.zoom)

    if cat_radio and fl.wf_cat_link_flag and not fl.cat_snd_link_flag: # shift WF by half span when CAT outside WF
        cat_radio.get_mode()
        kiwi_wf.radio_mode = cat_radio.radio_mode

        old_cat_freq = cat_radio.freq
        cat_radio.get_freq()
        if cat_radio.freq != old_cat_freq:
            kiwi_wf.tune = cat_radio.freq - (CW_PITCH if kiwi_wf.radio_mode=="CW" else 0.)

            delta_f = (cat_radio.freq - kiwi_wf.freq)
            if abs(delta_f) < 5*kiwi_wf.span_khz:
                if delta_f + kiwi_wf.span_khz/2 < 0:
                    kiwi_wf.set_freq_zoom(kiwi_wf.start_f_khz, kiwi_wf.zoom)
                elif delta_f - kiwi_wf.span_khz/2 > 0:
                    kiwi_wf.set_freq_zoom(kiwi_wf.end_f_khz, kiwi_wf.zoom)
            else:
                kiwi_wf.set_freq_zoom(cat_radio.freq, kiwi_wf.zoom)

    if True or not run_index%kiwi_wf.averaging_n:
        plot_spectrum(sdrdisplay, kiwi_wf, disp,filled=disp.SPECTRUM_FILLED, col=YELLOW)
        wf_surface = pygame.surfarray.make_surface(kiwi_wf.wf_data.T)
        wf_surface.set_palette(palRGB)
        if disp.DISPLAY_WIDTH != kiwi_wf.WF_BINS:
            wf_surface = pygame.Surface.convert(wf_surface)
            wf_surface = pygame.transform.smoothscale(wf_surface, (disp.DISPLAY_WIDTH, disp.WF_HEIGHT))
        sdrdisplay.blit(wf_surface, (0, disp.WF_Y))

    pygame.draw.rect(sdrdisplay, (0,0,80), (0,0,disp.DISPLAY_WIDTH,disp.TOPBAR_HEIGHT), 0)
    pygame.draw.rect(sdrdisplay, (0,0,80), (0,disp.TUNEBAR_Y,disp.DISPLAY_WIDTH,disp.TUNEBAR_HEIGHT), 0)
    pygame.draw.rect(sdrdisplay, (0,0,0), (0,disp.BOTTOMBAR_Y,disp.DISPLAY_WIDTH,disp.DISPLAY_HEIGHT), 0)
    draw_lines(sdrdisplay, wf_height, kiwi_snd.radio_mode, mouse, kiwi_wf, disp, kiwi_snd, kiwi_snd2, fl, cat_radio)
    update_textsurfaces(sdrdisplay, kiwi_snd.radio_mode, rssi_smooth, mouse, wf_width, kiwi_wf, disp, kiwi_snd, kiwi_snd2, fl, cat_radio, kiwi_host2, run_index)

    if fl.show_eibi_flag and kiwi_wf.zoom > 6:
        plot_eibi(sdrdisplay, eibi, kiwi_wf, disp)
    elif fl.show_dxcluster_flag and kiwi_wf.zoom > 3:
        plot_dxcluster(sdrdisplay, dxclust, kiwi_wf, disp)

    time_now = datetime.utcnow()
    if check_time.second != time_now.second and not time_now.second % 10:
        check_time = time_now
        beacon_project.which_beacons()
        # print(beacon_project.beacons_dict)
    if kiwi_wf.zoom > 8:
        plot_beacons(sdrdisplay, beacon_project, kiwi_wf, disp)

    if fl.input_freq_flag:
        question = "Freq (kHz)"
        display_box(sdrdisplay, question + ": " + "".join(current_string), 200)
    elif fl.input_callsign_flag:
        question = "DXCLuster CALLSIGN"
        display_box(sdrdisplay, question + ": " + "".join(current_string), 300)
    elif fl.input_server_flag:
        display_kiwi_box(sdrdisplay, current_string, kiwilist)
    elif fl.show_help_flag:
        display_help_box(sdrdisplay, HELP_MESSAGE_LIST)
    elif show_bigmsg:
        pos = None
        msg_color = WHITE
        if run_index - run_index_bigmsg > 65:
            show_bigmsg = None
        if "VOLUME" == show_bigmsg:
            msg_color = WHITE if kiwi_snd.volume <= 100 else RED
            msg_text = "VOLUME: %d"%(kiwi_snd.volume)+'%'
        if "WFAVG" == show_bigmsg:
            msg_color = WHITE if kiwi_wf.averaging_n == 1 else RED
            msg_text = "WF AVG %dX -> %.2fs"%(kiwi_wf.averaging_n, kiwi_wf.averaging_n/FPS)
        elif "cat_rx_sync" == show_bigmsg:
            msg_text = "CAT<->RX SYNC "+("ON" if fl.cat_snd_link_flag else "OFF")
        elif "forcesync" == show_bigmsg:
            msg_text = "Center RX passband" if not cat_radio else "Force SYNC WF & RX -> CAT"
        elif "switchab" == show_bigmsg:
            msg_text = "Switch MAIN/SUB RXs"
        elif "enable2rx" == show_bigmsg:
            msg_text = "SUB RX enabled"
        elif "disable2rx" == show_bigmsg:
            msg_text = "SUB RX disabled"
        elif "automode" == show_bigmsg:
            msg_text = "AUTO MODE "+("ON" if fl.auto_mode else "OFF")
        elif "changemode" == show_bigmsg:
            msg_text = kiwi_snd.radio_mode
        elif "writememory" == show_bigmsg:
            msg_text = "Stored Memory %d"% (len(kiwi_memory.mem_list)-1)
        elif "restorememory" == show_bigmsg:
            msg_text = "Recall memory:%d -> %s"% (kiwi_memory.index, 
                str(kiwi_memory.mem_list[kiwi_memory.index][0])+" kHz "+kiwi_memory.mem_list[kiwi_memory.index][1]) 
            pos = (disp.DISPLAY_WIDTH / 2 - 300, disp.DISPLAY_HEIGHT / 2 - 10)
        elif "resetmemory" == show_bigmsg:
            msg_text = "Reset All Memories!"
        elif "loadmemorydisk" == show_bigmsg:
            msg_text = "Load Memories from Disk"
            pos = (disp.DISPLAY_WIDTH / 2 - 300, disp.DISPLAY_HEIGHT / 2 - 10)
        elif "savememorydisk" == show_bigmsg:
            msg_text = "Save All Memories to Disk"
            pos = (disp.DISPLAY_WIDTH / 2 - 300, disp.DISPLAY_HEIGHT / 2 - 10)
        elif "emptymemory" == show_bigmsg:
            msg_text = "No Memories!"
        elif "start_rec" == show_bigmsg:
            msg_text = "Start recording"
        elif "stop_rec" == show_bigmsg:
            msg_text = "Save recording"
        elif "centertune" == show_bigmsg:
            msg_text = "WF center tune mode " + ("ON" if fl.wf_snd_link_flag else "OFF")
        elif "agc" == show_bigmsg:
            msg_text = "AGC threshold: %d dBm" % kiwi_snd.thresh

        display_msg_box(sdrdisplay, msg_text, pos=pos, color=msg_color)

    # rssi_smooth = np.mean(list(rssi_hist)[:])+10 # +10 is to approximately recalibrate the S-meter after averaging over time

    rssi_last = rssi_hist[-1]
    if math.fabs(rssi_last)>math.fabs(rssi_smooth):
        rssi_smooth -= 3 if kiwi_snd.radio_mode=="CW" else 0.5 # s-meter decay rate
    else:
        rssi_smooth = (rssi_last+rssi_smooth)/2 # attack rate

    if fl.s_meter_show_flag:
        smeter_surface = s_meter_draw(rssi_smooth, kiwi_snd.thresh, disp)
        sdrdisplay.blit(smeter_surface, (0, disp.BOTTOMBAR_Y-80))

    mouse = pygame.mouse.get_pos()
    pygame.display.flip()
    clock.tick(FPS)

    if cat_radio and not cat_radio.cat_ok:
        cat_radio = None

# close audio stream
kiwi_audio_stream.stop()

if kiwi_snd2:
    kiwi_audio_stream2.stop()

kiwi_snd.terminate = True
if kiwi_snd2:
    kiwi_snd2.terminate = True

kiwi_wf.terminate = True
time.sleep(0.5)

kiwi_wf.close_connection()
kiwi_snd.close_connection()
if kiwi_snd2:
     kiwi_snd2.close_connection()

pygame.quit()
