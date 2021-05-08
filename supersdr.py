#!/usr/bin/env python3

import _thread
from optparse import OptionParser
from utils_supersdr import *

# Approximate HF band plan from https://www.itu.int/en/ITU-R/terrestrial/broadcast/Pages/Bands.aspx
# and https://www.iaru-r1.org/reference/band-plans/hf-bandplan/

def update_textsurfaces(radio_mode, rssi, mouse, wf_width):
    global sdrdisplay
    mousex_pos = mouse[0]
    if mousex_pos < 25:
        mousex_pos = 25
    elif mousex_pos >= DISPLAY_WIDTH - 80:
        mousex_pos = DISPLAY_WIDTH - 80
    buff_level = kiwi_snd.audio_buffer.qsize()
    #           Label   Color   Freq/Mode                       Screen position
    ts_dict = {"wf_freq": (YELLOW, "%.1f"%(kiwi_wf.freq if cat_snd_link_flag else kiwi_wf.freq), (wf_width/2+3,TUNEBAR_Y+6), "small", False),
            "left": (GREEN, "%.1f"%(kiwi_wf.start_f_khz) ,(0,TUNEBAR_Y+6), "small", False),
            "right": (GREEN, "%.1f"%(kiwi_wf.end_f_khz), (wf_width-50,TUNEBAR_Y+6), "small", False),
            "rx_freq": (GREY, "%.2fkHz %s"%(kiwi_snd.freq, kiwi_snd.radio_mode), (wf_width/2+55,V_POS_TEXT), "small", False),
            "kiwi": (RED if buff_level<FULL_BUFF_LEN/2 else GREEN, ("kiwi:"+kiwi_wf.host)[:30] ,(95,BOTTOMBAR_Y+6), "small", False),
            "span": (ORANGE, "SPAN:%.0fkHz"%(round(kiwi_wf.span_khz)), (wf_width-80,SPECTRUM_Y+1), "small", False),
            "filter": (GREY, "FILT:%.1fkHz"%((kiwi_snd.hc-kiwi_snd.lc)/1000.), (wf_width/2+210, V_POS_TEXT), "small", False),
            "p_freq": (WHITE, "%dkHz"%mouse_khz, (mousex_pos+4, TUNEBAR_Y+1), "small", False),
            "auto": ((GREEN if auto_mode else RED), "[AUTO]", (wf_width/2+165, V_POS_TEXT), "small", False),
            "center": ((GREEN if wf_snd_link_flag else GREY), "CENTER", (wf_width-130, SPECTRUM_Y+2), "small", False),
            "sync": ((GREEN if cat_snd_link_flag else GREY), "SYNC", (40, BOTTOMBAR_Y+4), "big", False),
            "cat": (GREEN if cat_radio else GREY, "CAT", (5,BOTTOMBAR_Y+4), "big", False), 
            "recording": (RED if audio_rec.recording_flag and run_index%2 else D_GREY, "REC", (wf_width-90, BOTTOMBAR_Y+4), "big", False),
            "help": (BLUE, "HELP", (wf_width-50, BOTTOMBAR_Y+4), "big", False)
            }
    if not s_meter_show_flag:
        ts_dict["smeter"] = (GREEN, "%.0fdBm"%rssi_smooth, (20,V_POS_TEXT), "big", False)
    
    draw_dict = {}
    for k in ts_dict:
        if k == "p_freq" and not (pygame.mouse.get_focused() and WF_Y <= mouse[1] <= BOTTOMBAR_Y):
            continue
        if "small" in ts_dict[k][3]:
            smallfont = pygame.freetype.SysFont('Mono', 12)
            render_ = smallfont.render_to
        elif "big" in ts_dict[k][3]:
            bigfont = pygame.freetype.SysFont('Mono', 16)
            render_ = bigfont.render_to
        fontsize_ = font_size_dict[ts_dict[k][3]]
        render_(sdrdisplay, ts_dict[k][2], ts_dict[k][1], ts_dict[k][0])

def draw_lines(surface_, wf_height, radio_mode, mouse):
    center_freq_bin = kiwi_wf.offset_to_bin(kiwi_wf.span_khz/2)
    # center WF line
    pygame.draw.line(surface_, RED, (center_freq_bin, WF_Y), (center_freq_bin, WF_Y+6), 4)
    # mouse click_freq line
    if pygame.mouse.get_focused() and WF_Y <= mouse[1] <= BOTTOMBAR_Y:
        pygame.draw.line(surface_, (250,0,0), (mouse[0], TUNEBAR_Y), (mouse[0], TUNEBAR_Y+TUNEBAR_HEIGHT), 1)

    snd_freq_bin = kiwi_wf.offset_to_bin(kiwi_snd.freq+kiwi_wf.span_khz/2-kiwi_wf.freq)
    if snd_freq_bin>0 and snd_freq_bin< WF_BINS:
        # carrier line
        pygame.draw.line(surface_, RED, (snd_freq_bin, TUNEBAR_Y), (snd_freq_bin, TUNEBAR_Y+TUNEBAR_HEIGHT), 1)
    if cat_radio and not cat_snd_link_flag:
        tune_freq_bin = kiwi_wf.offset_to_bin(kiwi_wf.tune+kiwi_wf.span_khz/2-kiwi_wf.freq)
        # tune wf line
        pygame.draw.line(surface_, D_RED, (tune_freq_bin, TUNEBAR_Y), (tune_freq_bin, TUNEBAR_Y+TUNEBAR_HEIGHT), 3)
        
    lc_bin = kiwi_wf.offset_to_bin(kiwi_snd.lc/1000.)
    lc_bin = snd_freq_bin + lc_bin
    if lc_bin>0 and lc_bin< WF_BINS:
        # low cut line
        pygame.draw.line(surface_, GREEN, (lc_bin, TUNEBAR_Y), (lc_bin-5, TUNEBAR_Y+TUNEBAR_HEIGHT), 1)
    
    hc_bin = kiwi_wf.offset_to_bin(kiwi_snd.hc/1000)
    hc_bin = snd_freq_bin + hc_bin
    if hc_bin>0 and hc_bin< WF_BINS:
        # high cut line
        pygame.draw.line(surface_, GREEN, (hc_bin, TUNEBAR_Y), (hc_bin+5, TUNEBAR_Y+TUNEBAR_HEIGHT), 1)
    
    pygame.draw.line(surface_, GREEN, (lc_bin, TUNEBAR_Y), (hc_bin, TUNEBAR_Y), 2)

    if cat_radio and not cat_snd_link_flag:
        lc_, hc_ = kiwi_wf.change_passband(delta_low, delta_high)
        lc_bin = kiwi_wf.offset_to_bin(lc_/1000.)
        lc_bin = tune_freq_bin + lc_bin + 1
        if lc_bin>0 and lc_bin< WF_BINS:
            # low cut line
            pygame.draw.line(surface_, YELLOW, (lc_bin, TUNEBAR_Y), (lc_bin-5, TUNEBAR_Y+TUNEBAR_HEIGHT), 1)
        
        hc_bin = kiwi_wf.offset_to_bin(hc_/1000)
        hc_bin = tune_freq_bin + hc_bin
        if hc_bin>0 and hc_bin< WF_BINS:
            # high cut line
            pygame.draw.line(surface_, YELLOW, (hc_bin, TUNEBAR_Y), (hc_bin+5, TUNEBAR_Y+TUNEBAR_HEIGHT), 1)
        pygame.draw.line(surface_, YELLOW, (lc_bin, TUNEBAR_Y), (hc_bin, TUNEBAR_Y), 2)

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
    s_meter_center = (s_meter_radius+10,s_meter_radius+8)
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

def plot_spectrum(t_avg=15, col=GREEN):
    global sdrdisplay
    spectrum_surf = pygame.Surface((1024, SPECTRUM_HEIGHT))
    pixarr = pygame.PixelArray (spectrum_surf)
    for x, v in enumerate(np.average(kiwi_wf.wf_data.T[:,-t_avg:], axis=1)):
        y = SPECTRUM_HEIGHT-1-int(v/255 *SPECTRUM_HEIGHT)
        pixarr[x,y] = col
    del pixarr
    sdrdisplay.blit(spectrum_surf, (0, SPECTRUM_Y))

def plot_eibi(surface_):
    for f_khz in set(eibi.visible_stations):
        f_bin = int(kiwi_wf.offset_to_bin(f_khz-kiwi_wf.start_f_khz))
        ts = (ORANGE, eibi.get_names(f_khz)[0], (f_bin,WF_Y+20), "small")
        smallfont = pygame.freetype.SysFont('Mono', 12)
        render_ = smallfont.render_to
        try:
            if ts[2][0]>10 and ts[2][0]<DISPLAY_WIDTH-10:
                render_(surface_, ts[2], ts[1],  rotation=90, fgcolor=ts[0], bgcolor=(20,20,20))
                pygame.draw.line(surface_, ORANGE, (f_bin, TUNEBAR_Y+TUNEBAR_HEIGHT), (f_bin, TUNEBAR_Y+15), 1)
        except:
            pass


parser = OptionParser()
parser.add_option("-w", "--password", type=str,
                  help="KiwiSDR password", dest="kiwipassword", default="")
parser.add_option("-s", "--kiwiserver", type=str,
                  help="KiwiSDR server name", dest="kiwiserver", default="")
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

palRGB = create_cm("cutesdr")

eibi = eibi_db()
dxclust = dxcluster("IS0KYB")

kiwi_host = options['kiwiserver']
kiwi_port = options['kiwiport']
kiwi_password = options['kiwipassword']
freq = options['freq'] # this is the central freq in kHz
zoom = options['zoom'] 
radiohost = options['radioserver']
radioport = options['radioport']

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

kiwi_filter = filtering(KIWI_RATE/2, AUDIO_RATE)
audio_rec = audio_recording("supersdr_%s.wav"%datetime.now())

print(kiwi_host, kiwi_port, kiwi_password, zoom, freq)
#init KIWI WF and RX audio
kiwi_wf = kiwi_waterfall(kiwi_host, kiwi_port, kiwi_password, zoom, freq, eibi)
wf_t = threading.Thread(target=kiwi_wf.run, daemon=True)
wf_t.start()


print(freq, radio_mode, 30, 3000, kiwi_password)
kiwi_snd = kiwi_sound(freq, radio_mode, 30, 3000, kiwi_password, kiwi_wf, kiwi_filter, audio_rec)
if not kiwi_snd:
    print("Server not ready")
    exit()
play, kiwi_audio_stream = start_audio_stream(kiwi_snd)
if not play:
    del kiwi_snd
    exit("Chosen KIWI receiver is not ready!")

# keep receiving dx cluster announces every 5s
dx_t = threading.Thread(target=dxclust.run, args=(kiwi_snd,), daemon=True)
dx_t.start()

# init Pygame
pygame.init()
sdrdisplay = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
wf_width = sdrdisplay.get_width()
wf_height = sdrdisplay.get_height()
i_icon = "icon.jpg"
icon = pygame.image.load(i_icon)
pygame.display.set_icon(icon)
pygame.display.set_caption("SuperSDR 2.0")
clock = pygame.time.Clock()
pygame.key.set_repeat(200, 50)

wf_quit = False

auto_mode = True
input_freq_flag = False
input_server_flag = False
show_help_flag =  False
s_meter_show_flag = False
show_eibi_flag = False

input_new_server = None
current_string = []

dx_cluster_msg = True

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
rssi_hist = deque(rssi_maxlen*[kiwi_snd.rssi], rssi_maxlen)
rssi_smooth = kiwi_snd.rssi
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
                if keys[pygame.K_i]:
                    show_eibi_flag = False if show_eibi_flag else True

                # Center RX freq on WF
                if keys[pygame.K_z]:
                    wf_snd_link_flag = False if wf_snd_link_flag else True
                    force_sync_flag = True
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
                    delta = 100 if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else -100
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
                    if kiwi_snd.volume < 150:
                        kiwi_snd.volume += 10
                    show_bigmsg = "VOLUME"
                    run_index_bigmsg = run_index
                if keys[pygame.K_b]:
                    if kiwi_snd.volume > 0:
                        kiwi_snd.volume -= 10
                    show_bigmsg = "VOLUME"
                    run_index_bigmsg = run_index
                if keys[pygame.K_m]:
                    if kiwi_snd.volume > 0:
                        old_volume = kiwi_snd.volume
                        kiwi_snd.volume = 0
                    else:
                        kiwi_snd.volume = old_volume
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
                            manual_snd_freq = ((kiwi_snd.freq)*10//1)/10 - (0.1 if not fast_tune else 1.0)
                elif keys[pygame.K_RIGHT]:
                    if not (keys[pygame.K_RCTRL] or keys[pygame.K_LCTRL]):                    
                        if kiwi_snd.radio_mode != "CW" and kiwi_wf.zoom < 10:
                            if not fast_tune:
                                manual_snd_freq = kiwi_snd.freq//1 + 1
                            else:
                                manual_snd_freq = kiwi_snd.freq//1 + 10
                        else:
                            manual_snd_freq = ((kiwi_snd.freq)*10//1)/10 + (0.1001 if not fast_tune else 1.0)
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

                # Start/stop audio recording to file
                elif keys[pygame.K_e]:
                    if not audio_rec.recording_flag:
                        audio_rec.start()
                        show_bigmsg = "start_rec"
                        run_index_bigmsg = run_index
                    else:
                        audio_rec.stop(play)
                        show_bigmsg = "stop_rec"
                        run_index_bigmsg = run_index

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
                            try:
                                current_string.append(chr(inkey))
                            except:
                                pass

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
                    kiwi_wf.set_white_flag()
            elif event.button == 5: # mouse scroll down
                if kiwi_wf.zoom>0:
                    t_khz = kiwi_wf.bins_to_khz(mouse[0])
                    zoom_f = kiwi_wf.freq + (kiwi_wf.freq-t_khz)
                    kiwi_wf.set_freq_zoom(zoom_f, kiwi_wf.zoom - 1)
                    kiwi_wf.set_white_flag()
            elif event.button == 1:
                if WF_Y <= mouse[1] <= BOTTOMBAR_Y:
                    kiwi_wf.zoom_to_span()
                    kiwi_wf.start_freq()
                    kiwi_wf.end_freq()
                    click_freq = kiwi_wf.bins_to_khz(mouse[0])
                    if kiwi_snd.radio_mode == "CW":
                        click_freq -= CW_PITCH # tune CW signal taking into account cw offset
    
    if mouse[0] > wf_width-50 and mouse[1] > BOTTOMBAR_Y+4:
        show_help_flag = True
    else:
        show_help_flag = False

    if input_server_flag and input_new_server:
        pygame.event.clear()
        input_text_list = input_new_server.rstrip().split(" ")

        # close PyAudio
        play.terminate()

        kiwi_snd.terminate = True
        kiwi_wf.terminate = True
        time.sleep(1)

        # stop stream
        kiwi_audio_stream.stop_stream()
        kiwi_audio_stream.close()

        kiwi_wf.close_connection()
        kiwi_snd.close_connection()

        if len(input_text_list) >= 1:
            new_host = input_text_list[0]
            new_port = int(kiwi_port)
            new_password = kiwi_password
        if len(input_text_list) >= 2:
            new_port = int(input_text_list[1])
        if len(input_text_list) == 3:
            new_password = input_text_list[2]
        
        kiwi_snd.terminate = False
        kiwi_wf.terminate = False

        try:
            kiwi_wf.__init__(new_host, new_port, new_password, zoom, freq, eibi)
            kiwi_snd.__init__(freq, radio_mode, 30, 3000, new_password, kiwi_wf, kiwi_filter, audio_rec)
            print("Changed server to: %s:%d" % (new_host,new_port))
            play, kiwi_audio_stream = start_audio_stream(kiwi_snd)
        except:
            print ("something went wrong...")
            play = None
        if not play:
            kiwi_wf = kiwi_waterfall(kiwi_host, kiwi_port, kiwi_password, zoom, freq, eibi)
            kiwi_snd = kiwi_sound(freq, radio_mode, 30, 3000, kiwi_password, kiwi_wf, kiwi_filter, audio_rec)
            print("Reverted back to server: %s:%d" % (kiwi_host, kiwi_port))
            play, kiwi_audio_stream = start_audio_stream(kiwi_snd)
            if not play:
                exit("Old kiwi server not available anymore!")
        else:
            kiwi_host, kiwi_port, kiwi_password = new_host, new_port, new_password


        wf_t = threading.Thread(target=kiwi_wf.run, daemon=True)
        wf_t.start()
            
        input_server_flag = False
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

    if input_freq_flag and manual_snd_freq:
        kiwi_wf.set_freq_zoom(manual_snd_freq, kiwi_wf.zoom)
        if auto_mode:
            kiwi_snd.radio_mode = get_auto_mode(manual_snd_freq)
            lc, hc = kiwi_snd.change_passband(delta_low, delta_high)
        kiwi_snd.freq = kiwi_wf.freq
        kiwi_snd.set_mode_freq_pb()
        input_freq_flag = False
        kiwi_wf.set_white_flag()

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
                kiwi_wf.set_white_flag()
            elif kiwi_snd.freq > kiwi_wf.end_f_khz:
                kiwi_wf.set_freq_zoom(kiwi_wf.end_f_khz, kiwi_wf.zoom)
                kiwi_wf.set_white_flag()

    if manual_wf_freq:
        kiwi_wf.set_freq_zoom(manual_wf_freq, kiwi_wf.zoom)
        kiwi_wf.set_white_flag()


    if manual_zoom:
        kiwi_wf.set_freq_zoom(kiwi_snd.freq, manual_zoom) # for now, the arrow zoom will be centered on the SND freq
        #kiwi_snd.freq = kiwi_wf.freq
        #kiwi_snd.set_mode_freq_pb()
        kiwi_wf.set_white_flag()


    # Change KIWI SND frequency
    if click_freq:

        kiwi_snd.freq = click_freq
        if auto_mode:
            kiwi_snd.radio_mode = get_auto_mode(kiwi_snd.freq)
            lc, hc = kiwi_snd.change_passband(delta_low, delta_high)
        kiwi_snd.set_mode_freq_pb()
        if wf_snd_link_flag or show_bigmsg == "restorememory":
            kiwi_wf.set_freq_zoom(click_freq, kiwi_wf.zoom)
            kiwi_wf.set_white_flag()

    if cat_radio and cat_snd_link_flag:
        if manual_mode:
            cat_radio.set_mode(kiwi_snd.radio_mode)
        elif click_freq or manual_snd_freq:
            if cat_radio.radio_mode != get_auto_mode(kiwi_snd.freq) and auto_mode:
                cat_radio.set_mode(kiwi_snd.radio_mode)
            cat_radio.set_freq(kiwi_snd.freq + (CW_PITCH if kiwi_snd.radio_mode=="CW" else 0.))
        else:
            kiwi_snd.radio_mode = cat_radio.get_mode()
            lc, hc = kiwi_snd.change_passband(delta_low, delta_high)
            kiwi_snd.set_mode_freq_pb()

            old_cat_freq = cat_radio.freq
            cat_radio.get_freq()
            if cat_radio.freq != old_cat_freq:
                kiwi_snd.freq = cat_radio.freq - (CW_PITCH if kiwi_snd.radio_mode=="CW" else 0.)
                if wf_cat_link_flag: # shift WF by half span when RX outside WF
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

    if cat_radio and wf_cat_link_flag and not cat_snd_link_flag: # shift WF by half span when CAT outside WF
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


    # clear the background with a uniform color
    pygame.draw.rect(sdrdisplay, (0,0,80), (0,0,DISPLAY_WIDTH,DISPLAY_HEIGHT), 0)
    pygame.draw.rect(sdrdisplay, (0,0,00), (0,BOTTOMBAR_Y,DISPLAY_WIDTH,DISPLAY_HEIGHT), 0)

    plot_spectrum()
    surface = pygame.surfarray.make_surface(np.flip(kiwi_wf.wf_data.T, axis=1))
    surface.set_palette(palRGB)
    sdrdisplay.blit(surface, (0, WF_Y))
    draw_lines(sdrdisplay, wf_height, kiwi_snd.radio_mode, mouse)
    update_textsurfaces(kiwi_snd.radio_mode, rssi_smooth, mouse, wf_width)

#    draw_textsurfaces(draw_dict, ts_dict, sdrdisplay)
    if show_eibi_flag and kiwi_wf.zoom > 6:
        plot_eibi(sdrdisplay)
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
            msg_color = WHITE if kiwi_snd.volume <= 100 else RED
            msg_text = "VOLUME: %d"%(kiwi_snd.volume)+'%'
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
        elif "start_rec" == show_bigmsg:
            msg_text = "Start recording"
        elif "stop_rec" == show_bigmsg:
            msg_text = "Save recording"
        elif "centertune" == show_bigmsg:
            msg_text = "WF center tune mode " + ("ON" if wf_snd_link_flag else "OFF")

        display_msg_box(sdrdisplay, msg_text, pos=None, fontsize=35, color=msg_color)

    rssi_smooth = np.mean(list(rssi_hist)[:])
    if s_meter_show_flag:
        s_meter_draw(rssi_smooth)

    pygame.display.update()
    clock.tick(20)
    mouse = pygame.mouse.get_pos()

    if cat_radio and not cat_radio.cat_ok:
        cat_radio = None

# close PyAudio
play.terminate()

kiwi_snd.terminate = True
kiwi_wf.terminate = True
time.sleep(0.5)

# stop stream
kiwi_audio_stream.stop_stream()
kiwi_audio_stream.close()

kiwi_wf.close_connection()
kiwi_snd.close_connection()

pygame.quit()
