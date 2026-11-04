/*
 * Nadajnik ID drona: 4 diody nadają RAZEM, w kółko, kod swojego ID z codes.h.
 * Protokół = vision_id/protocol.py, kody = vision_id/gen_codes.py (wspólne).
 *
 *   kod: 26 chipów, dioda świeci ~75% czasu i gaśnie najwyżej na 1 chip,
 *   128 ID; odbiornik sam znajduje fazę - nie ma synchronizacji ani CRC.
 *
 * Podłączenie (Arduino Nano): D2, D3, D4, D5 -> rezystor -> anoda (+, dłuższa nóżka)
 *                             katoda (-, krótsza nóżka, płaska strona) -> GND
 *   220 Ω: ~13 mA, bezpieczne dla 4 diod; 100 Ω: ~30 mA, jaśniej, tylko 1-2 diody.
 *
 * Komendy przez Serial Monitor (115200, "Newline"):
 *   ID 5     ustaw ID drona (0..127), zapisuje w EEPROM
 *   HZ 10    częstotliwość chipów (10 dla kamery 30 fps, 20 dla 60 fps), zapis w EEPROM
 *   ON       wszystkie diody świecą stale (do ustawienia kamery/progu)
 *   RUN      wróć do nadawania
 *   ?        pokaż konfigurację
 */

#include <EEPROM.h>
#include "codes.h"

const uint8_t LED_PINS[] = {2, 3, 4, 5};
const uint8_t N_LEDS = sizeof(LED_PINS);

const uint16_t EEPROM_MAGIC = 0xB10D;     // zmienione razem z protokołem
struct Config {
  uint16_t magic;
  uint8_t droneId;
  float chipHz;
};

Config cfg = {EEPROM_MAGIC, 1, 10.0};
uint32_t code;
uint8_t chipIdx = 0;
unsigned long chipUs;
unsigned long nextChipAt;
bool steadyOn = false;
String line;

void loadCode() {
  code = pgm_read_dword(&CODES[cfg.droneId]);
  chipUs = (unsigned long)(1000000.0 / cfg.chipHz);
  chipIdx = 0;
}

bool chipAt(uint8_t i) {
  // bit CHIPS-1 (najstarszy) = pierwszy nadawany chip
  return (code >> (CHIPS - 1 - i)) & 1;
}

void setLeds(bool on) {
  for (uint8_t i = 0; i < N_LEDS; i++) digitalWrite(LED_PINS[i], on ? HIGH : LOW);
}

void printConfig() {
  Serial.print(F("ID="));
  Serial.print(cfg.droneId);
  Serial.print(F(" HZ="));
  Serial.print(cfg.chipHz);
  Serial.print(F(" ramka="));
  Serial.print(CHIPS / cfg.chipHz);
  Serial.print(F("s chipy="));
  for (uint8_t i = 0; i < CHIPS; i++) Serial.print(chipAt(i) ? '1' : '0');
  Serial.println(steadyOn ? F(" [ON]") : F(""));
}

void handleCommand(String cmd) {
  cmd.trim();
  cmd.toUpperCase();
  if (cmd.startsWith("ID ")) {
    long id = cmd.substring(3).toInt();
    if (id < 0 || id >= N_IDS) {
      Serial.println(F("ID poza zakresem 0..127"));
      return;
    }
    cfg.droneId = id;
  } else if (cmd.startsWith("HZ ")) {
    float hz = cmd.substring(3).toFloat();
    if (hz < 1 || hz > 200) {
      Serial.println(F("HZ poza zakresem 1..200"));
      return;
    }
    cfg.chipHz = hz;
  } else if (cmd == "ON") {
    steadyOn = true;
  } else if (cmd == "RUN") {
    steadyOn = false;
  } else if (cmd != "?") {
    Serial.println(F("Komendy: ID n | HZ f | ON | RUN | ?"));
    return;
  }
  EEPROM.put(0, cfg);
  loadCode();
  nextChipAt = micros();
  printConfig();
}

void setup() {
  for (uint8_t i = 0; i < N_LEDS; i++) pinMode(LED_PINS[i], OUTPUT);
  Serial.begin(115200);

  Config stored;
  EEPROM.get(0, stored);
  if (stored.magic == EEPROM_MAGIC && stored.droneId < N_IDS && stored.chipHz >= 1 && stored.chipHz <= 200) {
    cfg = stored;
  } else {
    // stara konfiguracja (poprzedni protokół) - przenosimy ID i HZ, jeśli mają sens
    struct { uint16_t magic; uint8_t droneId; float chipHz; } old;
    EEPROM.get(0, old);
    if (old.magic == 0xB10C && old.droneId < N_IDS && old.chipHz >= 1 && old.chipHz <= 200) {
      cfg.droneId = old.droneId;
      cfg.chipHz = old.chipHz;
    }
    EEPROM.put(0, cfg);
  }
  loadCode();
  printConfig();
  nextChipAt = micros();
}

void loop() {
  // nadajemy według stałego rozkładu (nextChipAt += chipUs), żeby nie dryfować
  if ((long)(micros() - nextChipAt) >= 0) {
    nextChipAt += chipUs;
    setLeds(steadyOn || chipAt(chipIdx));
    chipIdx = (chipIdx + 1) % CHIPS;
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
