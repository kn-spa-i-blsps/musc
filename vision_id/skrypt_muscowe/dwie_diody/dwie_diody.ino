/*
 * Dwie diody, każda nadaje kod SWOJEGO ID z własną częstotliwością chipów.
 * Protokół i kody te same co firmware/led_beacon (codes.h z gen_codes.py),
 * więc odbiornik (receiver.py / verify_live.py) je odczyta.
 *
 * Dozwolone częstotliwości = te, które przechodzą weryfikację (~3 klatki na chip):
 *   10 Hz  -> kamera 30 fps (obieg kodu 2.6 s)
 *   20 Hz  -> kamera 60 fps (obieg kodu 1.3 s)
 * Odbiornik dekoduje jedną częstotliwością naraz (--chip-hz), więc dioda z inną
 * częstotliwością niż ustawiona w odbiorniku nie zostanie odczytana.
 *
 * Podłączenie (Arduino Nano):
 *   dioda A: D2 -> rezystor 220 Ω -> anoda (+, dłuższa nóżka), katoda -> GND
 *   dioda B: D3 -> rezystor 220 Ω -> anoda (+, dłuższa nóżka), katoda -> GND
 *
 * Komendy przez Serial Monitor (115200, "Newline"); zapis w EEPROM:
 *   A ID 5    ID diody A (0..127)        B ID 7    ID diody B
 *   A HZ 10   częstotliwość diody A      B HZ 20   częstotliwość diody B
 *   HZ 20     obie diody naraz (tego używa verify_live.py)
 *   ID 5      ID diody A (zgodność z verify_live.py / pi/ard.py)
 *   ON        obie diody świecą stale     RUN   wróć do nadawania
 *   ?         pokaż konfigurację
 */

#include <EEPROM.h>
#include "codes.h"

// --- ustawienia domyślne (gdy EEPROM pusty) ---
const uint8_t PIN_A = 13, PIN_B = 3;
// ID 5 i 30 różnią się w 8 chipach przy każdym przesunięciu (1 i 2 tylko w 4 - minimum)
const uint8_t DEFAULT_ID_A = 5, DEFAULT_ID_B = 30;
// obie diody z TĄ SAMĄ częstotliwością - odbiornik dekoduje jedną naraz (20 Hz = kamera Pi 60 fps)
const uint8_t DEFAULT_HZ_A = 20, DEFAULT_HZ_B = 20;

const uint8_t ALLOWED_HZ[] = {10, 20};     // dopuszczone przez weryfikację
const uint16_t EEPROM_MAGIC = 0xD10E;      // inny niż led_beacon (0xB10D); zmiana = wgranie nowych domyślnych
const int EEPROM_ADDR = 0;

struct LedCfg {
  uint8_t id;
  uint8_t hz;
};

struct Config {
  uint16_t magic;
  LedCfg led[2];
};

struct LedState {
  uint8_t pin;
  uint32_t code;
  uint8_t chipIdx;
  unsigned long chipUs;
  unsigned long nextChipAt;
};

Config cfg = {EEPROM_MAGIC, {{DEFAULT_ID_A, DEFAULT_HZ_A}, {DEFAULT_ID_B, DEFAULT_HZ_B}}};
LedState led[2] = {{PIN_A}, {PIN_B}};
bool steadyOn = false;
String line;

bool hzAllowed(long hz) {
  for (uint8_t i = 0; i < sizeof(ALLOWED_HZ); i++)
    if (hz == ALLOWED_HZ[i]) return true;
  return false;
}

bool cfgValid(const Config &c) {
  if (c.magic != EEPROM_MAGIC) return false;
  for (uint8_t i = 0; i < 2; i++)
    if (c.led[i].id >= N_IDS || !hzAllowed(c.led[i].hz)) return false;
  return true;
}

bool chipAt(uint32_t code, uint8_t i) {
  // bit CHIPS-1 (najstarszy) = pierwszy nadawany chip
  return (code >> (CHIPS - 1 - i)) & 1;
}

void loadCodes() {
  unsigned long now = micros();
  for (uint8_t i = 0; i < 2; i++) {
    led[i].code = pgm_read_dword(&CODES[cfg.led[i].id]);
    led[i].chipUs = 1000000UL / cfg.led[i].hz;
    led[i].chipIdx = 0;
    led[i].nextChipAt = now;
  }
}

void printLed(uint8_t i) {
  // format "ID=.. HZ=.. ramka=..s chipy=.." - ten sam co led_beacon, czyta go verify_live.py
  Serial.print(i == 0 ? F("A: ") : F("B: "));
  Serial.print(F("ID="));
  Serial.print(cfg.led[i].id);
  Serial.print(F(" HZ="));
  Serial.print((float)cfg.led[i].hz);
  Serial.print(F(" ramka="));
  Serial.print(CHIPS / (float)cfg.led[i].hz);
  Serial.print(F("s chipy="));
  for (uint8_t c = 0; c < CHIPS; c++) Serial.print(chipAt(led[i].code, c) ? '1' : '0');
  Serial.println(steadyOn ? F(" [ON]") : F(""));
}

void printConfig() {
  printLed(0);
  printLed(1);
}

void printHelp() {
  Serial.println(F("Komendy: [A|B] ID n | [A|B] HZ f | ON | RUN | ?   (HZ: 10 lub 20)"));
}

// ustawia ID / HZ dla diod z maski (bit0 = A, bit1 = B); false = błąd
bool setParam(uint8_t mask, const String &arg) {
  if (arg.startsWith("ID ")) {
    long id = arg.substring(3).toInt();
    if (id < 0 || id >= N_IDS) {
      Serial.println(F("ID poza zakresem 0..127"));
      return false;
    }
    for (uint8_t i = 0; i < 2; i++) if (mask & (1 << i)) cfg.led[i].id = id;
    return true;
  }
  if (arg.startsWith("HZ ")) {
    long hz = lround(arg.substring(3).toFloat());
    if (!hzAllowed(hz)) {
      Serial.println(F("HZ niedozwolone - tylko 10 (kamera 30 fps) albo 20 (60 fps)"));
      return false;
    }
    for (uint8_t i = 0; i < 2; i++) if (mask & (1 << i)) cfg.led[i].hz = hz;
    return true;
  }
  printHelp();
  return false;
}

void handleCommand(String cmd) {
  cmd.trim();
  cmd.toUpperCase();
  if (cmd.startsWith("A ")) {
    if (!setParam(0b01, cmd.substring(2))) return;
  } else if (cmd.startsWith("B ")) {
    if (!setParam(0b10, cmd.substring(2))) return;
  } else if (cmd.startsWith("HZ ")) {
    if (!setParam(0b11, cmd)) return;            // obie diody
  } else if (cmd.startsWith("ID ")) {
    if (!setParam(0b01, cmd)) return;            // dioda A
  } else if (cmd == "ON") {
    steadyOn = true;
  } else if (cmd == "RUN") {
    steadyOn = false;
  } else if (cmd != "?") {
    printHelp();
    return;
  }
  EEPROM.put(EEPROM_ADDR, cfg);
  loadCodes();
  printConfig();
}

void setup() {
  for (uint8_t i = 0; i < 2; i++) pinMode(led[i].pin, OUTPUT);
  Serial.begin(115200);

  Config stored;
  EEPROM.get(EEPROM_ADDR, stored);
  if (cfgValid(stored)) cfg = stored;
  else EEPROM.put(EEPROM_ADDR, cfg);

  loadCodes();
  printConfig();
}

void loop() {
  unsigned long now = micros();
  for (uint8_t i = 0; i < 2; i++) {
    LedState &l = led[i];
    // stały rozkład (nextChipAt += chipUs), żeby nie dryfować
    if ((long)(now - l.nextChipAt) >= 0) {
      l.nextChipAt += l.chipUs;
      digitalWrite(l.pin, (steadyOn || chipAt(l.code, l.chipIdx)) ? HIGH : LOW);
      l.chipIdx = (l.chipIdx + 1) % CHIPS;
    }
  }

  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (line.length()) handleCommand(line);
      line = "";
    } else if (line.length() < 32) {
      line += c;
    }
  }
}
