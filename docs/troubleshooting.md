# Troubleshooting

Here are some common issues and solutions when using `pypixelcolor`.

## Bluetooth Connection Issues

### Device not found

- Ensure the device is powered on.
- Ensure the device is not connected to another phone or computer. The device can only handle one BLE connection at a time.
- Try moving closer to the device.

### Connection Timeout

If you experience timeouts when connecting:

- Restart the Bluetooth service on your computer.
- Power cycle the LED device.

## Linux Specifics

On Linux, you might need to ensure your user has the correct permissions to access the Bluetooth adapter.

1. Ensure `bluez` is installed.
2. Add your user to the `bluetooth` group (if it exists) or check your distribution's documentation for BLE permissions.

```bash
sudo usermod -aG bluetooth $USER
```

You may need to log out and log back in for changes to take effect.
