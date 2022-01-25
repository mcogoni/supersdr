#!/usr/bin/env python3

import _thread
from optparse import OptionParser
from utils_supersdr import *

def update_textsurfaces(surface_, radio_mode, rssi, mouse, wf_width):
    mousex_pos = mouse[0]
    if mousex_pos < 25:
        mousex_pos = 25
    elif mousex_pos >= DISPLAY_WIDTH - 80:
        mousex_pos = DISPLAY_WIDTH - 80
    buff_level = kiwi_snd.audio_buffer.qsize()
    main_rx_color = RED if not kiwi_snd.subrx else GREEN
    sub_rx_color = GREEN if not kiwi_snd.subrx else RED
    #           Label   Color   Freq/Mode                       Screen position
    ts_dict = {"wf_freq": (YELLOW, "%.1f"%(kiwi_wf.freq if cat_snd_link_flag else kiwi_wf.freq), (wf_width/2-68,TUNEBAR_Y+2), "small", False),
            "left": (GREEN, "%.1f"%(kiwi_wf.start_f_khz) ,(0,TUNEBAR_Y+2), "small", False),
            "right": (GREEN, "%.1f"%(kiwi_wf.end_f_khz), (wf_width-50,TUNEBAR_Y+2), "small", False),
            "rx_freq": (main_rx_color, "%sMAIN:%.3fkHz %s"%("[MUTE]" if kiwi_snd.volume==0 else "[ENBL]", kiwi_snd.freq+(CW_PITCH if kiwi_snd.radio_mode=="CW" else 0), kiwi_snd.radio_mode), (wf_width/2-50,V_POS_TEXT), "small", False),
            "kiwi": (D_RED if buff_level<kiwi_snd.FULL_BUFF_LEN/3 else RED, ("kiwi1:"+kiwi_wf.host)[:30] ,(95,BOTTOMBAR_Y+6), "small", False),
            "span": (GREEN, "SPAN:%.0fkHz"%(round(kiwi_wf.span_khz)), (wf_width-95,SPECTRUM_Y+1), "small", False),
            "filter": (GREY, "FILT:%.1fkHz"%((kiwi_snd.hc-kiwi_snd.lc)/1000.), (wf_width/2+230, V_POS_TEXT), "small", False),
            "p_freq": (WHITE, "%dkHz"%mouse_khz, (mousex_pos+4, TUNEBAR_Y-20), "small", False),
            "auto": ((GREEN if auto_mode else RED), "[AUTO]" if auto_mode else "[MANU]", (wf_width/2+165, V_POS_TEXT), "small", False),
            "center": ((GREEN if wf_snd_link_flag else GREY), "CENTER", (wf_width-145, SPECTRUM_Y+2), "small", False),
            "sync": ((GREEN if cat_snd_link_flag else GREY), "SYNC", (40, BOTTOMBAR_Y+4), "big", False),
            "cat": (GREEN if cat_radio else GREY, "CAT", (5,BOTTOMBAR_Y+4), "big", False), 
            "recording": (RED if kiwi_snd.audio_rec.recording_flag and run_index%2 else D_GREY, "REC", (wf_width-90, BOTTOMBAR_Y+4), "big", False),
            "dxcluster": (GREEN if show_dxcluster_flag else D_GREY, "DXCLUST", (wf_width-200, BOTTOMBAR_Y+4), "big", False),
            "utc": (WHITE, datetime.utcnow().strftime(" %d %b %Y %H:%M:%SZ"), (wf_width-155, 4), "small", False),
            "wf_param": (GREEN, "WF MIN:%ddB MAX:%ddB"%(kiwi_wf.delta_low_db, kiwi_wf.delta_high_db), (10,SPECTRUM_Y+1), "small", False),
            "help": (BLUE, "HELP", (wf_width-50, BOTTOMBAR_Y+4), "big", False)
            }

    if dualrx_flag and kiwi_snd2:
        ts_dict["rx_freq2"] = (sub_rx_color, "%sSUB:%.3fkHz %s"%("[MUTE]" if kiwi_snd2.volume==0 else "[ENBL]", kiwi_snd2.freq+(CW_PITCH if kiwi_snd2.radio_mode=="CW" else 0), kiwi_snd2.radio_mode), (wf_width/2-240,V_POS_TEXT), "small", False)
        ts_dict["kiwi2"] = (D_GREEN if buff_level<kiwi_snd2.FULL_BUFF_LEN/3 else GREEN, ("[kiwi2:%s]"%kiwi_host2)[:30] ,(280,BOTTOMBAR_Y+6), "small", False)
    if not s_meter_show_flag:
        s_value = (rssi_smooth+120)//6 # signal in S units of 6dB
        if s_value<=9:
            s_value = "S"+str(int(s_value))
        else:
            s_value = "S9+"+str(int((s_value-9)*6))+"dB"
        #ts_dict["smeter"] = (GREEN, "%.0fdBm"%rssi_smooth, (20,V_POS_TEXT), "big", False)
        ts_dict["smeter"] = (GREEN, s_value, (20,V_POS_TEXT), "big", False)
    if click_drag_flag:
        delta_khz = kiwi_wf.deltabins_to_khz(start_drag_x-mousex_pos)
        ts_dict["deltaf"] = (RED, ("+" if delta_khz>0 else "")+"%.1fkHz"%delta_khz, (wf_width/2,SPECTRUM_Y+20), "big", False)
    if kiwi_wf.averaging_n>1:
        ts_dict["avg"] = (RED, "AVG %dX"%kiwi_wf.averaging_n, (10,SPECTRUM_Y+13), "small", False)
    if len(kiwi_wf.div_list)>1:
        ts_dict["div"] = (YELLOW, "DIV :%.0fkHz"%(kiwi_wf.space_khz/10), (wf_width-95,SPECTRUM_Y+13), "small", False)
    else:
        ts_dict["div"] = (WHITE, "DIV :%.0fkHz"%(kiwi_wf.space_khz/100), (wf_width-95,SPECTRUM_Y+13), "small", False)

    draw_dict = {}
    for k in ts_dict:
        if k == "p_freq" and not (pygame.mouse.get_focused() and (WF_Y <= mouse[1] <= BOTTOMBAR_Y or TOPBAR_HEIGHT <= mouse[1] <= TUNEBAR_Y)):
            continue
        if "small" in ts_dict[k][3]:
            render_ = smallfont.render_to
        elif "big" in ts_dict[k][3]:
            render_ = bigfont.render_to
        render_(surface_, ts_dict[k][2], ts_dict[k][1], ts_dict[k][0])

def draw_lines(surface_, wf_height, radio_mode, mouse):

    def _plot_bandpass(color_, kiwi_):
        snd_freq_bin = kiwi_wf.offset_to_bin(kiwi_.freq+kiwi_wf.span_khz/2-kiwi_wf.freq)
        if snd_freq_bin>0 and snd_freq_bin< kiwi_wf.WF_BINS:
            # carrier line
            pygame.draw.line(surface_, RED, (snd_freq_bin, TUNEBAR_Y), (snd_freq_bin, TUNEBAR_Y+TUNEBAR_HEIGHT), 1)
        if cat_radio and not cat_snd_link_flag:
            tune_freq_bin = kiwi_wf.offset_to_bin(kiwi_wf.tune+kiwi_wf.span_khz/2-kiwi_wf.freq)
            # tune wf line
            pygame.draw.line(surface_, D_RED, (tune_freq_bin, TUNEBAR_Y), (tune_freq_bin, TUNEBAR_Y+TUNEBAR_HEIGHT), 3)
            
        lc_bin = kiwi_wf.offset_to_bin(kiwi_.lc/1000.)
        lc_bin = snd_freq_bin + lc_bin
        if lc_bin>0 and lc_bin< kiwi_wf.WF_BINS:
            # low cut line
            pygame.draw.line(surface_, color_, (lc_bin, TUNEBAR_Y), (lc_bin-5, TUNEBAR_Y+TUNEBAR_HEIGHT), 1)
        
        hc_bin = kiwi_wf.offset_to_bin(kiwi_.hc/1000)
        hc_bin = snd_freq_bin + hc_bin
        if hc_bin>0 and hc_bin< kiwi_wf.WF_BINS:
            # high cut line
            pygame.draw.line(surface_, color_, (hc_bin, TUNEBAR_Y), (hc_bin+5, TUNEBAR_Y+TUNEBAR_HEIGHT), 1)
        
        pygame.draw.line(surface_, color_, (lc_bin, TUNEBAR_Y), (hc_bin, TUNEBAR_Y), 2)



    center_freq_bin = kiwi_wf.offset_to_bin(kiwi_wf.span_khz/2)
    # center WF line
    pygame.draw.line(surface_, RED, (center_freq_bin, WF_Y), (center_freq_bin, WF_Y+6), 4)
    # mouse click_freq line
    if pygame.mouse.get_focused() and WF_Y <= mouse[1] <= BOTTOMBAR_Y:
        pygame.draw.line(surface_, RED, (mouse[0], TUNEBAR_Y), (mouse[0], BOTTOMBAR_Y), 1)
    elif pygame.mouse.get_focused() and TOPBAR_HEIGHT <= mouse[1] <= TUNEBAR_Y:
        pygame.draw.line(surface_, GREEN, (mouse[0], TOPBAR_HEIGHT), (mouse[0], TUNEBAR_Y+TUNEBAR_HEIGHT), 1)

    # SUB RX
    if dualrx_flag and kiwi_snd2:
        _plot_bandpass(GREEN, kiwi_snd2)
    # MAIN RX        
    _plot_bandpass(RED, kiwi_snd)

    #### CAT RADIO bandpass

    if cat_radio and not cat_snd_link_flag:
        tune_freq_bin = kiwi_wf.offset_to_bin(kiwi_wf.tune+kiwi_wf.span_khz/2-kiwi_wf.freq)
        lc_, hc_ = kiwi_wf.change_passband(delta_low, delta_high)
        lc_bin = kiwi_wf.offset_to_bin(lc_/1000.)
        lc_bin = tune_freq_bin + lc_bin + 1
        if lc_bin>0 and lc_bin< kiwi_wf.WF_BINS:
            # low cut line
            pygame.draw.line(surface_, ORANGE, (lc_bin, TUNEBAR_Y), (lc_bin-5, TUNEBAR_Y+TUNEBAR_HEIGHT), 1)
        
        hc_bin = kiwi_wf.offset_to_bin(hc_/1000)
        hc_bin = tune_freq_bin + hc_bin
        if hc_bin>0 and hc_bin< kiwi_wf.WF_BINS:
            # high cut line
            pygame.draw.line(surface_, ORANGE, (hc_bin, TUNEBAR_Y), (hc_bin+5, TUNEBAR_Y+TUNEBAR_HEIGHT), 1)
        pygame.draw.line(surface_, ORANGE, (lc_bin, TUNEBAR_Y), (hc_bin, TUNEBAR_Y), 2)

    # plot click and drag red horiz bar
    if click_drag_flag:
        pygame.draw.line(surface_, RED, (start_drag_x, SPECTRUM_Y+10), (mouse[0], SPECTRUM_Y+10), 4)

    # plot tuning minor and major ticks
    for x in kiwi_wf.div_list:
        pygame.draw.line(surface_, YELLOW, (x, TUNEBAR_Y+TUNEBAR_HEIGHT), (x, TUNEBAR_Y+5), 3)
    for x in kiwi_wf.subdiv_list:
        pygame.draw.line(surface_, WHITE, (x, TUNEBAR_Y+TUNEBAR_HEIGHT), (x, TUNEBAR_Y+15), 1)

def display_kiwi_box(screen, current_string_):
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
    smallfont.render_to(sdrdisplay, pos, message, WHITE)
    message = "Choose stored Kiwi number or enter new one (port and password are optional)"
    smallfont.render_to(sdrdisplay, (pos[0], pos[1]-20), message, WHITE)
    for i, kiwi in enumerate(kiwilist.mem_list):
        pos = (rec_pos[0]+2, rec_pos[1]+5+i*20)
        msg = "Kiwi server: %d -> %s:%d:%s"%(i, kiwi[0], kiwi[1], kiwi[2])
        smallfont.render_to(sdrdisplay, pos, msg, GREY)


def display_box(screen, message, size):
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

    window_size = 495
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
            smallfont.render_to(sdrdisplay, pos, msg, WHITE)

def display_msg_box(screen, message, pos=None, color=WHITE):
    if not pos:
        pos = (screen.get_width() / 2 - 100, screen.get_height() / 2 - 10)
    if len(message) != 0:
        hugefont.render_to(sdrdisplay, pos, message, color)
    
def s_meter_draw(rssi_smooth, agc_threshold):
    s_meter_radius = 50.
    SMETER_XSIZE, SMETER_YSIZE = 2*s_meter_radius+20, s_meter_radius+20
    smeter_surface = pygame.Surface((SMETER_XSIZE, SMETER_YSIZE))

    s_meter_center = (s_meter_radius+10,s_meter_radius+8)
    alpha_rssi = rssi_smooth+127
    alpha_rssi = -math.radians(alpha_rssi* 180/127.)-math.pi

    alpha_agc = agc_threshold+127
    alpha_agc = -math.radians(alpha_agc* 180/127.)-math.pi

    def _coords_from_angle(angle, s_meter_radius_):
        x_ = s_meter_radius_ * math.cos(angle)
        y_ = s_meter_radius_ * math.sin(angle)
        s_meter_x = s_meter_center[0] + x_
        s_meter_y = s_meter_center[1] - y_
        return s_meter_x, s_meter_y
    
    s_meter_x, s_meter_y = _coords_from_angle(alpha_rssi, s_meter_radius* 0.95)
    agc_meter_x, agc_meter_y = _coords_from_angle(alpha_agc, s_meter_radius* 0.7)
    pygame.draw.rect(smeter_surface, YELLOW,
                   (s_meter_center[0]-60, s_meter_center[1]-58, SMETER_XSIZE, SMETER_YSIZE), 0)
    pygame.draw.rect(smeter_surface, BLACK,
                   (s_meter_center[0]-60, s_meter_center[1]-58, SMETER_XSIZE, SMETER_YSIZE), 3)
    
    angle_list = np.linspace(0.4, math.pi-0.4, 9)
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
    return smeter_surface

def plot_spectrum(t_avg=15, col=GREEN):
    global sdrdisplay
    spectrum_surf = pygame.Surface((kiwi_wf.WF_BINS, SPECTRUM_HEIGHT))
    pixarr = pygame.PixelArray (spectrum_surf)
    for x, v in enumerate(np.nanmean(kiwi_wf.wf_data.T[:,:t_avg], axis=1)):
        y = SPECTRUM_HEIGHT-1-int(v/255 *SPECTRUM_HEIGHT)
        pixarr[x,y] = col
    del pixarr
    sdrdisplay.blit(spectrum_surf, (0, SPECTRUM_Y))

def plot_eibi(surface_):
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
            ts = (WHITE, station_record[3], (f_bin,WF_Y+20), "small")
        except:
            continue
        render_ = midfont.render_to
        str_len = len(ts[1])
        x, y = ts[2]
        if x>fontsize*str_len/2 and x<DISPLAY_WIDTH-10:
            if f_bin-old_fbin <= fontsize*str_len/2+5:
                y_offset += fontsize
            else:
                y_offset = 0
            old_fbin = f_bin
            try:
                render_(surface_, (x-str_len*fontsize/2-2, y+y_offset), ts[1],  rotation=0, fgcolor=ts[0], bgcolor=(20,20,20))
                pygame.draw.line(surface_, WHITE, (f_bin, WF_Y), (f_bin, WF_Y+20+y_offset), 1)
            except:
                pass
def plot_dxcluster(surface_):
    y_offset = 0
    old_fbin = -100
    fontsize = font_size_dict["medium"]
    station_list = [string_f_khz for f_khz in set(dxclust.visible_stations) for string_f_khz in dxclust.int_freq_dict[f_khz] ]
    sorted_station_list = sorted(station_list, key=float)
    for string_f_khz in sorted_station_list:
        f_khz_float = float(string_f_khz)
        f_bin = int(kiwi_wf.offset_to_bin(f_khz_float-kiwi_wf.start_f_khz))
        try:
            ts = (WHITE, dxclust.callsign_freq_dict[string_f_khz], (f_bin,WF_Y+20), "small")
        except:
            continue
        render_ = midfont.render_to
        str_len = len(ts[1])
        x, y = ts[2]
        if x>fontsize*str_len/2 and x<DISPLAY_WIDTH-10:
            if f_bin-old_fbin <= fontsize*str_len/2+5:
                y_offset += fontsize
            else:
                y_offset = 0
            old_fbin = f_bin
            try:
                render_(surface_, (x-str_len*fontsize/2-2, y+y_offset), ts[1],  rotation=0, fgcolor=ts[0], bgcolor=(20,20,20))
                pygame.draw.line(surface_, WHITE, (f_bin, WF_Y), (f_bin, WF_Y+20+y_offset), 1)
            except:
                pass

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
                  help="screen refresh rate", dest="refresh", default=20)
parser.add_option("-d", "--dual",
                  help="Activate Dual RX", action="store_true", dest="dualrx", default=False)
parser.add_option("-c", "--callsign", type=str,
                  help="DX CLUSTER Callsign", dest="callsign", default="")
parser.add_option("-m", "--colormap", type=str,
                  help="colormap for waterfall", dest="colormap", default="cutesdr")
                  

options = vars(parser.parse_args()[0])
FPS = options['refresh']
dualrx_flag = options['dualrx']

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
kiwi_wf = kiwi_waterfall(kiwi_host, kiwi_port, kiwi_password, zoom, freq, eibi)
wf_t = threading.Thread(target=kiwi_wf.run, daemon=True)
wf_t.start()

kiwi_snd = kiwi_sound(freq, radio_mode, 30, 3000, kiwi_password, kiwi_wf)
if not kiwi_snd:
    print("Server not ready")
    exit()

kiwi_snd2 = None
if dualrx_flag:
    time.sleep(2)
    kiwi_snd2 = kiwi_sound(freq, radio_mode, 30, 3000, kiwi_password2, kiwi_wf, 0, kiwi_host2, kiwi_port2, True)
    if not kiwi_snd2:
        print("Server not ready")

play, kiwi_audio_stream = start_audio_stream(kiwi_snd)
if not play:
    del kiwi_snd
    exit("Chosen KIWI receiver is not ready!")

if dualrx_flag:
    play2, kiwi_audio_stream2 = start_audio_stream(kiwi_snd2)
    if not play2:
        kiwi_snd2 = None


old_volume = kiwi_snd.volume

# init Pygame
pygame.init()
sdrdisplay = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT), 
    pygame.SCALED | pygame.RESIZABLE | pygame.DOUBLEBUF)
wf_width = sdrdisplay.get_width()
wf_height = sdrdisplay.get_height()
i_icon = "icon.jpg"
icon = pygame.image.load(i_icon)
pygame.display.set_icon(icon)
pygame.display.set_caption("SuperSDR %s"%VERSION)
clock = pygame.time.Clock()
pygame.key.set_repeat(200, 50)
nanofont = pygame.freetype.SysFont('Mono', 8)
microfont = pygame.freetype.SysFont('Mono', 10)
smallfont = pygame.freetype.SysFont('Mono', 12)
midfont = pygame.freetype.SysFont('Mono', 14)
bigfont = pygame.freetype.SysFont('Mono', 16)
hugefont = pygame.freetype.SysFont('Mono', 35)

wf_quit = False

auto_mode = True
input_freq_flag = False
input_server_flag = False
show_help_flag =  False
s_meter_show_flag = False
show_eibi_flag = False
show_dxcluster_flag = False

input_new_server = None
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
lc, hc = kiwi_snd.change_passband(delta_low, delta_high)
kiwi_snd.set_mode_freq_pb()

# Operating modes:
wf_cat_link_flag = True if cat_radio else False
wf_snd_link_flag = False
cat_snd_link_flag = True if cat_radio else False
print("SYNC OPTIONS:")
print("WF<>CAT", wf_cat_link_flag, "WF<>RX", wf_snd_link_flag, "CAT<>RX", cat_snd_link_flag)

rssi_maxlen = kiwi_snd.FULL_BUFF_LEN*2 # buffer length used to smoothen the s-meter
rssi_hist = deque(rssi_maxlen*[kiwi_snd.rssi], rssi_maxlen)
rssi_smooth = kiwi_snd.rssi
run_index = 0
run_index_automode = 0
show_bigmsg = None
msg_text = ""
click_drag_flag = False

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
            before_help_flag = show_help_flag
            show_help_flag = False
            if not input_freq_flag and not input_server_flag:
                keys = pygame.key.get_pressed()
                mods = pygame.key.get_mods()

                #event.unicode== (pygame.K_Z | pygame.KMOD_SHIFT)
                # Force SYNC WF to RX freq if no CAT, else WF and RX to CAT
                if keys[pygame.K_SPACE]:
                    force_sync_flag = True
                    show_bigmsg = "forcesync"
                    run_index_bigmsg = run_index

                # Show EIBI labels
                if keys[pygame.K_i]:
                    show_eibi_flag = False if show_eibi_flag else True

                # Show realtime DX-CLUSTER labels
                if keys[pygame.K_d]:
                    if dxclust:
                        show_dxcluster_flag = False if show_dxcluster_flag else True
                        if show_dxcluster_flag:
                            dxclust.terminate = False
                        else:
                            dxclust.terminate = True
                    else:
                        if not CALLSIGN:
                            print("*"*20)
                            CALLSIGN = input("Please enter your CALLSIGN to access DXCLUSTER: ")
                            try:
                                dxclust = dxcluster(CALLSIGN)
                                if dxclust:
                                    print(dxclust)
                                    dxclust.connect()
                                    dx_t = threading.Thread(target=dxclust.run, args=(kiwi_wf,), daemon=True)
                                    dx_t.start()
                                    dx_cluster_msg = True
                                    show_dxcluster_flag = True
                                else:
                                    dx_cluster_msg = False

                            except:
                                dxclust = None


                # Center RX freq on WF
                if keys[pygame.K_z]:
                    wf_snd_link_flag = False if wf_snd_link_flag else True
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
                        kiwi_memory.write_mem(kiwi_snd.freq, radio_mode, lc, hc)
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
                    delta = 100 if (event.mod & pygame.KMOD_SHIFT) else -100
                    delta_low += delta
                    if delta_low > 3000:
                        delta_low = 3000
                    elif delta_low < -3000:
                        delta_low = -3000
                if keys[pygame.K_k]:
                    change_passband_flag = True
                    delta = -100 if (event.mod & pygame.KMOD_SHIFT) else 100
                    delta_high += delta
                    if delta_high > 3000:
                        delta_high = 3000
                    elif delta_high < -3000:
                        delta_high = -3000.
                
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
                if keys[pygame.K_v]:
                    if kiwi_snd.volume < 150:
                        kiwi_snd.volume += 10
                    show_bigmsg = "VOLUME"
                    run_index_bigmsg = run_index
                elif keys[pygame.K_b]:
                    if kiwi_snd.volume > 0:
                        kiwi_snd.volume -= 10
                    show_bigmsg = "VOLUME"
                    run_index_bigmsg = run_index
                elif keys[pygame.K_m]:
                    if kiwi_snd.volume > 0:
                        old_volume = kiwi_snd.volume
                        kiwi_snd.volume = 0
                    else:
                        kiwi_snd.volume = old_volume
                    show_bigmsg = "VOLUME"
                    run_index_bigmsg = run_index

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
                if keys[pygame.K_f]:
                    input_freq_flag = True
                    current_string = []

                # Start/stop audio recording to file
                if keys[pygame.K_e]:
                    if not kiwi_snd.audio_rec.recording_flag:
                        kiwi_snd.audio_rec.start()
                        show_bigmsg = "start_rec"
                        run_index_bigmsg = run_index
                    else:
                        kiwi_snd.audio_rec.stop(play)
                        show_bigmsg = "stop_rec"
                        run_index_bigmsg = run_index

                # S-meter show/hide
                if keys[pygame.K_s]:
                    if (mods & pygame.KMOD_SHIFT):
                        s_meter_show_flag = False if s_meter_show_flag else True
                    elif cat_radio:
                        show_bigmsg = "cat_rx_sync"
                        run_index_bigmsg = run_index
                        cat_snd_link_flag = False if cat_snd_link_flag else True
                        force_sync_flag = True

                # Automatic mode change ON/OFF
                if keys[pygame.K_x]:
                    show_bigmsg = "automode"
                    run_index_bigmsg = run_index
                    auto_mode = False if auto_mode else True
                    if auto_mode:
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
                        kiwi_audio_stream2.stop()
                        kiwi_snd2.terminate = True
                        time.sleep(1)
                        kiwi_audio_stream2.stop_stream()
                        kiwi_audio_stream2.close()
                        kiwi_snd2.close_connection()
                        kiwi_snd2.terminate = False

                        dualrx_flag = False
                        show_bigmsg = "disable2rx"
                        run_index_bigmsg = run_index
                        kiwi_snd2 = None
                # Switch audio MAIN/SUB VFOs
                elif keys[pygame.K_y]:
                    if kiwi_snd2:
                        kiwi_snd, kiwi_snd2 = kiwi_snd2, kiwi_snd
                        #force_sync_flag = True
                        show_bigmsg = "switchab"
                        run_index_bigmsg = run_index
                    else:
                        kiwi_snd2 = kiwi_sound(kiwi_snd.freq, kiwi_snd.radio_mode, 30, 3000,  kiwi_password2, kiwi_wf, 0, kiwi_host2, kiwi_port2, True)
                        if not kiwi_snd2:
                            print("Server not ready")
                        play2, kiwi_audio_stream2 = start_audio_stream(kiwi_snd2)
                        if not play2:
                            kiwi_snd2 = None
                        else:
                            print("Second RX active!")
                            show_bigmsg = "enable2rx"
                            run_index_bigmsg = run_index
                            dualrx_flag = True


                # Quit SuperSDR
                if keys[pygame.K_ESCAPE] and keys[pygame.K_LSHIFT]:
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
                if WF_Y <= mouse[1] <= BOTTOMBAR_Y:
                    kiwi_wf.zoom_to_span()
                    kiwi_wf.start_freq()
                    kiwi_wf.end_freq()
                    click_freq = kiwi_wf.bins_to_khz(mouse[0])
                    if kiwi_snd.radio_mode == "CW":
                        click_freq -= CW_PITCH # tune CW signal taking into account cw offset
                if SPECTRUM_Y <= mouse[1] <= TUNEBAR_Y:
                    pygame.mouse.get_rel()
                    start_drag_x = mouse[0]
                    click_drag_flag = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and click_drag_flag:
                delta_x = pygame.mouse.get_rel()[0]
                delta_freq = kiwi_wf.deltabins_to_khz(delta_x)
                manual_wf_freq = kiwi_wf.freq - delta_freq
                click_drag_flag = False
    
    if mouse[0] > wf_width-50 and mouse[1] > BOTTOMBAR_Y+4 and pygame.mouse.get_focused():
        show_help_flag = True
    else:
        show_help_flag = False

    if input_server_flag and input_new_server:
        pygame.event.clear()
        if len(input_new_server) == 1:
            try:
                input_text_list = list(kiwilist.mem_list[int(input_new_server)])
            except:
                input_server_flag = False
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

            input_server_flag = False
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
            dualrx_flag = False
        kiwi_wf.terminate = False

        try:
            kiwi_wf.__init__(new_host, new_port, new_password, zoom, freq, eibi)
            kiwi_snd.__init__(freq, radio_mode, 30, 3000, new_password, kiwi_wf, old_volume)
            print("Changed server to: %s:%d" % (new_host,new_port))
            play, kiwi_audio_stream = start_audio_stream(kiwi_snd)
        except:
            print ("something went wrong...")
            play = None
        if not play:
            kiwi_wf = kiwi_waterfall(kiwi_host, kiwi_port, kiwi_password, zoom, freq, eibi)
            kiwi_snd = kiwi_sound(freq, radio_mode, 30, 3000, kiwi_password, kiwi_wf, old_volume)
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

    plot_spectrum()    
    surface = pygame.surfarray.make_surface(kiwi_wf.wf_data.T)
    surface.set_palette(palRGB)
    sdrdisplay.blit(surface, (0, WF_Y))

    pygame.draw.rect(sdrdisplay, (0,0,80), (0,0,DISPLAY_WIDTH,TOPBAR_HEIGHT), 0)
    pygame.draw.rect(sdrdisplay, (0,0,80), (0,TUNEBAR_Y,DISPLAY_WIDTH,TUNEBAR_HEIGHT), 0)
    pygame.draw.rect(sdrdisplay, (0,0,0), (0,BOTTOMBAR_Y,DISPLAY_WIDTH,DISPLAY_HEIGHT), 0)
    draw_lines(sdrdisplay, wf_height, kiwi_snd.radio_mode, mouse)
    update_textsurfaces(sdrdisplay, kiwi_snd.radio_mode, rssi_smooth, mouse, wf_width)

    if show_eibi_flag and kiwi_wf.zoom > 6:
        plot_eibi(sdrdisplay)
    elif show_dxcluster_flag and kiwi_wf.zoom > 3:
        plot_dxcluster(sdrdisplay)

    if input_freq_flag:
        question = "Freq (kHz)"
        display_box(sdrdisplay, question + ": " + "".join(current_string), 200)
    elif input_server_flag:
        display_kiwi_box(sdrdisplay, current_string)
    elif show_help_flag:
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
            msg_text = "WF AVG %dX"%(kiwi_wf.averaging_n)
        elif "cat_rx_sync" == show_bigmsg:
            msg_text = "CAT<->RX SYNC "+("ON" if cat_snd_link_flag else "OFF")
        elif "forcesync" == show_bigmsg:
            msg_text = "Center RX passband" if not cat_radio else "Force SYNC WF & RX -> CAT"
        elif "switchab" == show_bigmsg:
            msg_text = "Switch MAIN/SUB RXs"
        elif "enable2rx" == show_bigmsg:
            msg_text = "SUB RX enabled"
        elif "disable2rx" == show_bigmsg:
            msg_text = "SUB RX disabled"
        elif "automode" == show_bigmsg:
            msg_text = "AUTO MODE "+("ON" if auto_mode else "OFF")
        elif "changemode" == show_bigmsg:
            msg_text = kiwi_snd.radio_mode
        elif "writememory" == show_bigmsg:
            msg_text = "Stored Memory %d"% (len(kiwi_memory.mem_list)-1)
        elif "restorememory" == show_bigmsg:
            msg_text = "Recall memory:%d -> %s"% (kiwi_memory.index, 
                str(kiwi_memory.mem_list[kiwi_memory.index][0])+" kHz "+kiwi_memory.mem_list[kiwi_memory.index][1]) 
            pos = (DISPLAY_WIDTH / 2 - 300, DISPLAY_HEIGHT / 2 - 10)
        elif "resetmemory" == show_bigmsg:
            msg_text = "Reset All Memories!"
        elif "loadmemorydisk" == show_bigmsg:
            msg_text = "Load Memories from Disk"
            pos = (DISPLAY_WIDTH / 2 - 300, DISPLAY_HEIGHT / 2 - 10)
        elif "savememorydisk" == show_bigmsg:
            msg_text = "Save All Memories to Disk"
            pos = (DISPLAY_WIDTH / 2 - 300, DISPLAY_HEIGHT / 2 - 10)
        elif "emptymemory" == show_bigmsg:
            msg_text = "No Memories!"
        elif "start_rec" == show_bigmsg:
            msg_text = "Start recording"
        elif "stop_rec" == show_bigmsg:
            msg_text = "Save recording"
        elif "centertune" == show_bigmsg:
            msg_text = "WF center tune mode " + ("ON" if wf_snd_link_flag else "OFF")
        elif "agc" == show_bigmsg:
            msg_text = "AGC threshold: %d dBm" % kiwi_snd.thresh

        display_msg_box(sdrdisplay, msg_text, pos=pos, color=msg_color)

    rssi_smooth = np.mean(list(rssi_hist)[:])+10 # +10 is to approximately recalibrate the S-meter after averaging over time
    if s_meter_show_flag:
        smeter_surface = s_meter_draw(rssi_smooth, kiwi_snd.thresh)
        sdrdisplay.blit(smeter_surface, (0, BOTTOMBAR_Y-80))

    mouse = pygame.mouse.get_pos()
    pygame.display.flip()
    clock.tick(FPS)

    if cat_radio and not cat_radio.cat_ok:
        cat_radio = None


# close PyAudio
kiwi_audio_stream.stop()

if kiwi_snd2:
    kiwi_audio_stream2.stop()

kiwi_snd.terminate = True
if kiwi_snd2:
    kiwi_snd2.terminate = True

kiwi_wf.terminate = True
time.sleep(0.5)
exit()

kiwi_wf.close_connection()
kiwi_snd.close_connection()
if kiwi_snd2:
     kiwi_snd2.close_connection()

pygame.quit()
