# SuperSDR

SuperSDR allows a realtime view of the spectrum waterfall of any KiwiSDR around the world along with a local or remotely controlled CAT transceiver.
![SuperSDR in action](https://raw.githubusercontent.com/mcogoni/supersdr/main/supersdr_screenshot.png)

## Prerequisites:
Install Python 2.7 along with PYGAME, MATPLOTLIB and NUMPY, or whatever it asks when breaking apart upon launch ;)

## Use:
To launch the program:
```
./supersdr.py --kiwiserver 192.168.1.82 --kiwiport 8073 -z 9 -f 14060
```
to just explore a local kiwisdr, or:

```
./supersdr.py --kiwiserver http://on5kq.ddns.net --kiwiport 8073 -S 192.168.1.89 -P 4532 -z 9 -f 14060
```
to connect to a remote kiwi AND to a local CAT radio for which ```rigctld``` is running on the server located at ```192.168.1.89```.

Just use ```--help``` to show all available command line options.

Main key commands during use are:

- LEFT/RIGHT: move freq +/- 1kHz (+SHIFT: X10)
- UP/DOWN: zoom in/out by a factor 2X
- U/L/C: switches to USB, LSB, CW
- F: enter frequency with keyboard
- H: displays this help window
- SHIFT+ESC: quits

When connected to both a kiwisdr and to a CAT radio any click on the waterfall synchronizes the radio and, vice versa, moving the VFO on the radio, changes the center of the waterfall.

Have fun!

73,
marco / IS0KYB
