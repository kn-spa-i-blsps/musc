"""Wysyła komendy do nadajnika LED (Arduino Nano na /dev/ttyUSB0) i wypisuje odpowiedź.
   python3 ard.py "HZ 20" ON      |  python3 ard.py RUN  |  python3 ard.py "?"
Otwarcie portu resetuje Nano, więc czekamy na jego start; ustawienia są w EEPROM."""
import sys
import time

import serial

ser = serial.Serial(sys.argv[2] if sys.argv[1:2] == ["--port"] else "/dev/ttyUSB0", 115200, timeout=0.3)
time.sleep(2.0)
ser.read(4096)
for cmd in [a for a in sys.argv[1:] if a not in ("--port",)]:
    ser.write((cmd + "\n").encode())
    time.sleep(0.3)
    print(f"> {cmd}\n{ser.read(4096).decode(errors='replace').strip()}")
