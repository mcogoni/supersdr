# SuperSDR

SuperSDR allows a realtime view of the spectrum waterfall and audio playback of any KiwiSDR around the world along with a local or remotely controlled CAT transceiver.
![SuperSDR in action](https://github.com/mcogoni/supersdr/blob/main/supersdr_screenshot.png)

## Prerequisites:
Install Python 3 along with PYGAME, PYAUDIO, MATPLOTLIB and NUMPY/SCIPY, or whatever it asks when breaking apart upon launch ;)
### On Windows:
I don't personally use Windows, but a beta tester let me know that you can run it there:
- Install miniconda (https://docs.conda.io/en/latest/miniconda.html) and the proposed version of Python (3.8)
- Now open the miniconda powershell to install the packages as below:
  * conda config --add channels conda-forge 
  * conda install pyaudio
- then use pip on a powershell:
  * pip install numpy
  * pip install scipy
  * pip install matplotlib
  * pip install pygame


## Use:
To launch the program:
```
./supersdr.py --kiwiserver 192.168.1.82 --kiwiport 8073 -z 9 -f 198 -w password
```
to just explore a local kiwisdr, or:

```
./supersdr.py --kiwiserver sibamanna.duckdns.org --kiwiport 8073 -S 192.168.1.89 -P 4532 -z 9 -f 198
```
to connect to a remote kiwi AND to a local CAT radio for which ```rigctld``` is running on the server located at ```192.168.1.89```.

Just use ```--help``` to show all available command line options.

Main key commands during use are:

- LEFT/RIGHT: move freq +/- 1kHz (+SHIFT: X10)
- UP/DOWN or mouse scroll: zoom in/out by a factor 2X
- PAGE UP/DOWN: shift freq by 1MHz
- U/L/C/A: switches to USB, LSB, CW, AM
- J/K/O: increase low(J)/high(K) cut passband (SHIFT decreases), O resets defaults
- V/B/M: volume up/down 10% or Mute
- X: AUTO MODE Switch above/below 10 MHz
- F: enter frequency with keyboard
- H: displays this help window
- SHIFT+ESC: quits

When connected to both a kiwisdr and to a CAT radio any click on the waterfall synchronizes the radio and, vice versa, moving the VFO on the radio, changes the center of the waterfall.

Have fun!

73,
marco / IS0KYB
