# SuperSDR

SuperSDR allows a realtime view of the spectrum waterfall and audio playback of any KiwiSDR around the world along with a local or remotely controlled CAT transceiver.
![SuperSDR in action](https://github.com/mcogoni/supersdr/blob/main/supersdr_screenshot.png)

## Prerequisites:
Install Python 3 along with PYGAME, PYAUDIO, MATPLOTLIB and NUMPY/SCIPY, or whatever it asks when breaking apart upon launch ;)

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
- F: enter frequency with keyboard
- H: displays this help window
- SHIFT+ESC: quits

When connected to both a kiwisdr and to a CAT radio any click on the waterfall synchronizes the radio and, vice versa, moving the VFO on the radio, changes the center of the waterfall.

Have fun!

73,
marco / IS0KYB
