# pypixelcolor

![pypixelcolor logo](assets/pngs/banner.png)

**pypixelcolor** (also known as `iPixel-CLI`) is a Python library and CLI tool for controlling iPixel Color LED matrix devices via Bluetooth Low Energy (BLE). It allows you to send commands to the device to manipulate the LED display, retrieve device information, and more.

## Features

- 📝 **Send text**: Display custom messages with various fonts and animations.
- 🖼️ **Send images**: Display images and GIFs on the matrix.
- ⚙️ **Control settings**: Adjust brightness, orientation, and power.
- 🕒 **Modes**: Switch between Clock, Rhythm, and Fun modes.
- 🐍 **Scriptable**: Full Python library support for automation.
- 🖥️ **CLI**: Easy to use command-line interface.

## Installation

You can install `pypixelcolor` via pip:

```bash
pip install pypixelcolor
```

## Quick Start

### Command Line Interface (CLI)

Scan for devices:

```bash
pypixelcolor --scan
```

Send text to a device:

```bash
pypixelcolor -a <MAC_ADDRESS> -c send_text "Hello World"
```

[Learn more about the CLI](getting_started/cli.md){ .md-button .md-button--primary }

### Python Library

```python
import pypixelcolor

client = pypixelcolor.Client("XX:XX:XX:XX:XX:XX")
client.connect()
client.send_text("Hello World")
client.disconnect()
```

[Learn more about the Library](getting_started/library.md){ .md-button .md-button--primary }
