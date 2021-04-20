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
./supersdr.py --kiwiserver http://on5kq.ddns.net --kiwiport 8075 -S 192.168.1.89 -P 4532 -z 9 -f 14060
```
to connect to a remote kiwi AND to a local CAT radio for which ```rigctld``` is running on the server located at ```192.168.1.89```.

Just use ```--help``` to show all available command line options.

When connected to both a kiwisdr and to a CAT radio any click on the waterfall synchronizes the radio and, vice versa, moving the VFO on the radio, changes the center of the waterfall.

Have fun!

73,
marco / IS0KYB
