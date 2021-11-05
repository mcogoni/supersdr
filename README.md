# SuperSDR

There are two main scenarios in which you'd like to use SuperSDR:
 - you've got a *radio without a panadapter* or you're simply *not happy with it* (in my case I have a Kenwood TS-590SG), in this case this application can help you to use your (or someone else's) KiwiSDR as a powerful panadapter and multiple receiver and to have all seamlessly synchronized: you may control your Kiwi from the radio or vice versa; you may also transmit with the radio and have your Kiwi RX muted automatically;
 - you've got a KiwiSDR and you're sick of using its stock web interface, you'd like to have near zero latency, waterfall immunity from thunderstorms, simple keyboard shortcuts, VERY low CPU usage, much improved waterfall averaging to detect even the faintest signals (à la LINRAD), ability to receive from multiple remote KiwiSDRs, you want a low resources app able to run from a Raspberry Pi to a Windows PC to a Macbook Pro, you want a very compact code that you can tweak even not being a Python programmer. 

SuperSDR integrates a realtime spectrum waterfall and (dual) audio receive from any KiwiSDR around the world, together with a local (or remote) controlled CAT transceiver.
There are three main items that may be independently controlled:
 - the KIWI WATERFALL, that may be linked to the CAT radio or to the mouse/keyboard tuning;
 - the (dual) KIWI RECEIVER, that may be tuned everywhere on the HF bands;
 - the CAT radio that serves as the main tuning method.
 
 All three may be strictly intelinked, or set up separately depending on the operating convenience:
 - the WF is usually tuned by the VFO on the CAT radio to serve as a panadapter and the KIWI RX may be turned off if not necessary;
 - the KIWI RX may be tuned on a secondary frequency with keyboard and mouse or with the VFO, then the CAT radio is unlinked from the RX and is free to explore other frequencies;
 - the RX and the CAT radio are both active on different frequencies while you can explore the bands moving around the WF window and its span.

In the screenshot you can see both KiwiSDR receivers active at the same time (green and red passbands), while the CAT Radio is the orange one on the right. Synchronization between Kiwi RXs and CAT VFO is disabled (press S to enable SYNC) in this case (CAT SYNC grayed out) so the three are working independently. You can switch MAIN/SUB Kiwi RX frequencies by pressing Y. Pressing M (un)mutes the current MAIN RX.

Notice how the lower waterfall half is noisier than the upper half thanks to 10X sample averaging (time binning). You can change averaging by pressing G/H to increase/decrease it.
 
![SuperSDR in action](https://github.com/mcogoni/supersdr/blob/main/SuperSDR_screenshot.png)

## Prerequisites:
Install Python 3 along with PYGAME, PYAUDIO, MATPLOTLIB and NUMPY/SCIPY, or whatever it asks when breaking apart upon launch ;) If you use Linux I hope I don't have to tell you how to install librearies and Python components: I presonally use a Arch based distro and pip to keep everything updated.
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

There is now the possibility to launch the program with no command line options and select the kiwi server at the start or change it at runtime (still buggy, but it mostly works). To use the CAT, at least for now, you have to specify it from the command line.

To launch the program:
```
./supersdr.py --kiwiserver 192.168.1.82 --kiwiport 8073 -z 9 -f 198 -w password
```
to just explore your local kiwisdr, or:

```
./supersdr.py --kiwiserver sibamanna.duckdns.org --kiwiport 8073 -S 192.168.1.89 -P 4532 -z 9 -f 198
```
to connect to a remote kiwi AND to a local CAT radio for which ```rigctld``` is running on the server located at ```192.168.1.89```.

Just use ```--help``` to show all available command line options.

Main key commands during use are shown by pressing H or moving the mouse to the HELP label on the bottom right.

When connected to both a kiwisdr and to a CAT radio any click on the waterfall synchronizes the radio and, vice versa, moving the VFO on the radio, changes the tuning on the waterfall causing the WF window to follow when outside the span.


Have fun!

73,
marco / IS0KYB
