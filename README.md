# SuperSDR

SuperSDR combines a realtime spectrum waterfall and audio receive of any KiwiSDR around the world together with a local (or remote) controlled CAT transceiver.
There are three main items that may be independently controlled:
 - the WATERFALL, that may be linked to the CAT radio or to the mouse/keyboard tuning;
 - the RECEIVER, that may be tuned everywhere on the HF bands;
 - the CAT radio that serves as the main tuning method.
 
 All three may be strictly intelinked, or they may be set up separately depending on the operating mode:
 - the WF is usually tuned by the VFO on the CAT radio to serve as a panadapter and the KIWI RX may be turned off if not necessary;
 - the KIWI RX may be tuned on a secondary frequency with keyboard and mouse or with the VFO, then the CAT radio is unlinked from the RX and is free to explore other frequencies;
 - the RX and the CAT radio are both active on different frequencies while you can explore the bands moving around the WF window and its span.
 In the future I plan to support multiple KIWI channels both for the WF and the RX. 
 
![SuperSDR in action](https://github.com/mcogoni/supersdr/blob/main/supersdr_screenshot.png)

## Prerequisites:
Install Python 3 along with PYGAME, PYAUDIO, MATPLOTLIB and NUMPY/SCIPY, or whatever it asks when breaking apart upon launch ;)
### On Windows:
There is now a preliminary Windows [executable](https://github.com/mcogoni/supersdr/releases/download/1.0/WinSuperSDR.zip) thanks to [Giovanni Busonera](https://github.com/Strato75).
This version doesn't need any library nor Python environment. The Windows executable will always lag a bit during the development.

I don't personally use Windows, but a beta tester let me know that you can run it there:
- Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) and the proposed version of Python (3.8)
- Now open the miniconda powershell to install the packages as below:
  * conda config --add channels conda-forge 
  * conda install pyaudio
- then use pip on a powershell:
  * pip install numpy
  * pip install scipy
  * pip install matplotlib
  * pip install pygame


## Use:
[![SuperSDR tutorial](https://studio.youtube.com/video/q27zInnop8g/0.jpg)](https://studio.youtube.com/video/q27zInnop8g "SuperSDR tutorial")
To launch the program:
```
./supersdr.py --kiwiserver 192.168.1.82 --kiwiport 8073 -z 9 -f 198 -w password
```
to just explore your local kiwisdr, or:

```
./supersdr.py --kiwiserver sibamanna.duckdns.org --kiwiport 8073 -S 192.168.1.89 -P 4532 -z 9 -f 198
```
to connect to a remote kiwi AND to a local CAT radio for which ```rigctld``` is running on the server located at ```192.168.1.89```.
Add the ```-a``` switch to activate the audio connection with the kiwi server.

Just use ```--help``` to show all available command line options.

Main key commands during use are:

- LEFT/RIGHT: move freq +/- 1kHz (+SHIFT: X10)
- UP/DOWN or mouse scroll: zoom in/out by a factor 2X centered on the current active RX or on the mouse pointer (mouse)
- PAGE UP/DOWN: shift WF freq by half span
- U/L/C/A: switches to USB, LSB, CW, AM
- J/K/O: increase low(J)/high(K) cut passband (SHIFT decreases), O resets defaults
- V/B/M: volume up/down 10% or Mute
- X: AUTO MODE Switch for amateur and broadcasting bands
- Z: Sync between CAT and RX or WF
- W/R: Write/Restore fast memory up to 10 frequencies
- SHIFT+W: Deletes all stored memories 
- F: enter frequency with keyboard
- H: displays this help window
- SHIFT+ESC: quits

When connected to both a kiwisdr and to a CAT radio any click on the waterfall synchronizes the radio and, vice versa, moving the VFO on the radio, changes the tuning on the waterfall causing the WF window to follow when outside the span.


Have fun!

73,
marco / IS0KYB
