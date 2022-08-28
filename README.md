# SuperSDR

![SuperSDR in action](https://github.com/mcogoni/supersdr/blob/main/SuperSDR_screenshot.png)

There are three main scenarios in which you'd like to use SuperSDR:
 - you've got a *radio without a panadapter* (I have a Kenwood TS-590SG) or you're simply *not happy with it*, in this case this application can help you to use your (or someone else's!) KiwiSDR as a powerful panadapter and multiple receiver and to have all seamlessly synchronized: you may control your Kiwi by tuning the radio or vice versa; you may also transmit with the radio and have your Kiwi RX muted automatically;
 - you've got a KiwiSDR and you're sick of using its stock web interface, you'd like to have near-zero latency, waterfall immunity from thunderstorms, powerful keyboard shortcuts, VERY low CPU usage (you may run dozens of instances on a powerful PC), much improved waterfall averaging to detect even the faintest signals (à la LINRAD), ability to receive from multiple remote KiwiSDRs, you want a low resources app able to run from a Raspberry Pi to a Windows PC to a Macbook Pro, you want a very compact code that you can tweak to your needs even not being a Python programmer; 
 - your HF radio is connected to a very directional antenna and you don't want to lose those faint signals from every direction: in this case you may connect your Kiwi RX to a good omnidirectional wide band receive antenna and explore the bands with it, then, when you find something interesting, you turn to your directional antenna on the CAT radio and you make the contact.

SuperSDR integrates a realtime spectrum waterfall and (dual) audio receive from any KiwiSDR around the world, together with a local (or remote) controlled CAT transceiver.
There are three main items that may be independently controlled:
 - the KIWI WATERFALL, that may be linked to the CAT radio or to the mouse/keyboard tuning;
 - the (dual) KIWI RECEIVER, that may be tuned everywhere on the HF bands;
 - the CAT radio that serves as the main tuning method.
 
 All three may be strictly interlinked, or set up separately depending on the operating convenience:
 - the WF is usually tuned by the VFO on the CAT radio to serve as a panadapter and the KIWI RX may be turned off if not necessary;
 - the KIWI RX may be tuned on a secondary frequency with keyboard and mouse or with the VFO, then the CAT radio is unlinked from the RX and is free to explore other frequencies;
 - the RX and the CAT radio are both active on different frequencies while you can explore the bands moving around the WF window (left/right keys) and its span (up/down).

In the screenshot you can see both KiwiSDR receivers active at the same time (green and red passbands), while the CAT Radio is the orange one on the right. Synchronization between Kiwi RXs and CAT VFO is disabled (press S to enable SYNC) in this case (CAT SYNC grayed out) so the three are working independently. You can switch MAIN/SUB Kiwi RX frequencies by pressing Y. Pressing M (un)mutes the current MAIN RX.

Notice how the lower waterfall half is noisier than the upper half thanks to 10X sample averaging (time binning). You can change averaging by pressing G/H to increase/decrease it. In this way you may also monitor a wide frequency band for several hours (e.g. you could monitor the whole MW band through the night).

Mouse over the HELP label on the bottom right corner will show you the main keyboard commands.

## Main keyboard shortcuts:

 - **TUNING**: **LEFT/RIGHT** keys tune by -/+1 khz, when very zoomed in, they tune by 100 Hz, when you also press **SHIFT** the tuning step is 10X. You may also **point and click** on any signal with the mouse. Just **use the VFO of your radio** instead. The spectrum scope will also move by half span when you reach the borders. Pressing **Z** enables the center tuning and shifts the waterfall/scope while you tune keeping the RX bandpass centered;
 - **SPECTRUM TUNING**: **UP/DOWN** change zoom level (**SPAN**) by **2X** (equivalent: mouse wheel scroll). **PAGE UP/PAGE DOWN** move the spectrum center by half the span, but do not affect the RX frequency, you may return to your (main) RX frequency by pressing the **SPACE** bar. You may **DRAG** the spectrum scope (not the waterfall) with your mouse and also see dynamically by how much you're dragging it;
 - **FILTER BANDPASS**: **CONTROL+J/K** grows/shrinks by **100Hz** the whole filter, **J/K** changes the **LOW/HIGH CUT** and **SHIFT** inverts the change. Pressing **O (not zero)** resets it to default. Bandpass width is not supported by Hamlib (at least for my radio) so you don't affect the CAT controlled radio;
 - **RECEIVERS**: you start with one active RX passband whose rx mode changes automatically with frequency, you may decativate **AUTO RX MODE** by pressing **X** or manually changing mode. Second kiwi RX is activated by pressing **Y** that uses the same KIWISDR as the first. You may change the KIWISDR by pressing **Q** and entering the address:port of any KIWISDR. You may decouple the tuning of the Kiwi and that of the CAT radio by pressing **S**: you'll now have a separate orange bandpass representing your real VFO, press **S** again to relink. The currently being tuned RX is the **RED** one, the **GREEN** one has no controls and monitors the frequency you left it into. Pressing **Y** again you just switch the RX bandpasses and you can control the other. Each KIWI RX has a separate volume/mute control but the same **AGC threshold** that you can change via **1/2** keys and that level is visible on the s-meter with the short blue needle;
 - **S-METER**: activate it with the **M** key: it has an analog simulation that changes between CW and SSB/AM. The values over S9 are not accurate (it is a known problem I'm working on). The numbers in dB are correct though;
 - **SPECTRUM SCOPE LEVELS**: by default you have a fully automatic, dynamically set minimum and maximum values that are shown on the left. This is the most useful mode especially because it's almost insensitive to thunderstorms and you don't get those pesky horizontal lines on the low bands! But if you need to perform some measurement, you may turn **AUTO mode** off with the **3** key that will also get you horizontal lines every 10dB vertical division. The **TOP/BOTTOM dB levels** may be changed by pressing **./, keys** (**SHIFT** to change bottom level);
 - **ACCUMULATION AVERAGE**:  if you're familiar with LINRAD by Leif Asbrink, you'll surely be fond of the possibility to perform multiple samplings for each shown spectrum/waterfall line. This leads to large visual enhancements of the SNR both in the scope and the waterfall. This is performed by pressing **G** several times and it will slow down the update. Pressing **H** will lower the number of samples until you get back to normal;

OK, these are the main points you should be familiar with to use the software. There are several other functions but you'll discover them with the HELP menu... :)

## Prerequisites:
Install Python 3 along with PYGAME, SOUNDDEVICE, MATPLOTLIB and NUMPY/SCIPY, or whatever it asks when breaking apart upon launch ;) If you use Linux I hope I don't have to tell you how to install librearies and Python components: I presonally use a Arch based distro and pip to keep everything updated.
### On Windows:
There is now a preliminary Windows [executable](https://github.com/mcogoni/supersdr/releases/download/v3.0beta/WinSuperSDR.zip) thanks to [Giovanni Busonera](https://github.com/Strato75).
This version doesn't need any library nor Python environment. The Windows executable will always lag a bit during the development.

## Use:
[![SuperSDR tutorial](https://studio.youtube.com/video/q27zInnop8g/0.jpg)](https://youtu.be/Q4H7ZYqxGA0 "SuperSDR tutorial")

There is now the possibility to launch the program *with no command line options* and select the kiwi server at the start or change it at runtime (still buggy, but it mostly works). To use the CAT, at least for now, you have to specify it from the command line.

To launch the program under Linux:
```
./supersdr.py --kiwiserver 192.168.1.82 --kiwiport 8073 -z 9 -f 198 -w password
```
to just explore your local kiwisdr (defaults to kiwisdr.local if not specified), or:

```
./supersdr.py --kiwiserver sibamanna.duckdns.org --kiwiport 8073 -S 192.168.1.89 -P 4532 -z 9 -f 198
```
to connect to a remote kiwi AND to a local CAT radio for which ```rigctld``` is running on the server located at ```192.168.1.89```.

To start rigctld use somthing like: ```rigctld -m 237 -r /dev/ttyUSB0``` if you use a Kenwood TS-590SG (model nr. 237) and it is connected to the USB of your local computer. Run ```rigctld -l``` to show a list of supported radios. 

Just use ```--help``` to show all available command line options.

When connected to both a kiwisdr and to a CAT radio any click on the waterfall synchronizes the radio and, vice versa, moving the VFO on the radio, changes the tuning on the waterfall causing the WF window to follow when outside the span.


Have fun!

73,
marco / IS0KYB
