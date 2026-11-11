/*
 * Blinks drone ID 5 for vision_id/detector.py - nothing else.
 *
 * Wiring: pin 2 -> 220 ohm resistor -> LED + (long leg); LED - (short leg) -> GND.
 * The on-board LED (pin 13) blinks the same code.
 *
 * Receiver: uv run receiver.py --chip-hz 10   (laptop webcam, 30 fps)
 */

// ID 5: 26 chips, repeated forever, 1 = LED on (from vision_id/codebook.py)
const uint8_t CODE[] = {0,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1};
const uint8_t CHIPS = sizeof(CODE);

const unsigned long CHIP_US = 100000;   // 10 Hz = 100 ms per chip (use 50000 for a 60 fps camera + --chip-hz 20)

unsigned long nextChipAt;
uint8_t i = 0;

void setup() {
  pinMode(2, OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);
  nextChipAt = micros();
}

void loop() {
  // fixed schedule instead of delay(), so the timing does not drift
  if ((long)(micros() - nextChipAt) >= 0) {
    nextChipAt += CHIP_US;
    digitalWrite(2, CODE[i]);
    digitalWrite(LED_BUILTIN, CODE[i]);
    i = (i + 1) % CHIPS;
  }
}
