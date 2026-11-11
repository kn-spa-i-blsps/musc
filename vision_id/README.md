# vision_id — identyfikacja drona po mruganiu diod (bez ML)

Dron nadaje swoje ID mruganiem diod, kamera innego drona je odczytuje i liczy
kąt namiaru (azymut/elewację). Odległość daje UWB, kąt daje kamera.

```
firmware/led_beacon/led_beacon.ino   nadajnik (Arduino Nano, 4 diody)
firmware/led_beacon/codes.h          tablica kodów dla Arduino (z gen_codes.py)
gen_codes.py                         generuje tablicę kodów (codebook.py + codes.h)
codebook.py                          tablica kodów dla Pythona (z gen_codes.py)
protocol.py                          dekoder: dopasowanie do wzorców kodów
detector.py                          energia mrugania, śledzenie torów, łączenie w drona, kąt
tracking.py                          filtr Kalmana (stała prędkość) + algorytm węgierski do przydziału
camera.py                            kamera w osobnym wątku (bez gubienia klatek)
receiver.py                          odbiornik: kamera albo plik wideo, podgląd, zapis filmu
verify_live.py                       test na żywo: kamera + Arduino na USB, "DETECTED: MATCHING ID"
simulate.py                          syntetyczne wideo + test całego potoku bez sprzętu
tests/                               testy (uv run --group dev pytest)
```

## Protokół

Wszystkie diody świecą **razem** (ten sam sygnał) i w kółko nadają kod swojego ID:

- **26 chipów** na obieg (2.6 s przy 10 Hz, 1.3 s przy 20 Hz), **128 ID**.
- Dioda świeci **~75% czasu i gaśnie najwyżej na 1 chip** (100 ms przy 10 Hz)
  → prawie zawsze widoczna, śledzenie jej nie gubi.
- Dowolne dwa ID różnią się w **≥ 4 chipach przy każdym przesunięciu w czasie**
  → odbiornik nie potrzebuje synchronizacji ani znajomości fazy.
- Kody wybrał komputer (`gen_codes.py`), ta sama tablica trafia do Pythona i Arduino.

Dekodowanie to **dopasowanie wzorca** (matched filter): przebieg jasności z
ostatnich ~1.2 obiegu porównujemy z każdym kodem, każdą fazą i 3 tempami
(zegar Arduino nie jest idealny). Wygrywa najlepsze dopasowanie, jeśli jest
wysokie i wyraźnie lepsze od najlepszego innego ID. Pojedyncze złe klatki tylko
lekko obniżają wynik. Dodatkowo:

- sygnał musi wyglądać jak nadajnik (ciemność ≤ 1 chip, świeci 62–80%) —
  losowe mruganie, odblaski i migające lampy odpadają od razu,
- pierwsze dopasowanie = **ID wstępne** (`DRON 5?`), drugie zgodne co najmniej
  pół obiegu później = **potwierdzone** (`DRON 5`). Do decyzji roju używać
  tylko potwierdzonych.

Wymaga ok. **3 klatek na chip**:

| kamera | `HZ` w Arduino | obieg kodu |
|--------|----------------|------------|
| 30 fps (kamerka Maca) | 10 | 2.6 s |
| 60 fps (telefon, docelowa) | 20 | 1.3 s |

`verify_live.py` sam mierzy FPS kamery i ustawia HZ w Arduino.

Dlaczego wszystkie diody razem, a nie każda inny bit: patrząc z boku na drona
diody się nakładają i zasłaniają, a kolejności diod na obrazie nie da się
pewnie ustalić. Tak każda widoczna dioda niesie pełne ID, a 4 diody to zapas.

## Odbiornik — co dzieje się z każdą klatką

1. pomniejszenie do 960 px szerokości, kompensacja ruchu kamery (korelacja fazowa),
2. **top-hat** na jasności: zostają tylko małe jasne plamki; okno, ściana,
   biała płytka znikają niezależnie od tego, jak są jasne (działa tak samo z IR),
3. **śledzenie filtrem Kalmana**; bramka rośnie, gdy diody nie widać, więc
   szarpnięcie nie gubi toru; nowy tor tylko na plamce, która właśnie się zapaliła,
4. jasność plamki na torze → dekoder → ID wstępne / potwierdzone,
5. tor, który się zgubił i zaraz odnalazł obok, przejmuje ID wstępnie,
6. tory z tym samym ID = jeden dron; środek → azymut/elewacja.

## Co zrobić krok po kroku

### 1. Płytka

Na każdą diodę: pin Arduino (D2–D5) → **rezystor 220 Ω** → dłuższa nóżka diody (+),
krótsza nóżka (płaska strona obudowy) → GND. Przy 1–2 diodach można dać 100 Ω
(jaśniej, ~30 mA); przy 4 zostań przy 220 Ω.

### 2. Firmware

```bash
cd vision_id/firmware/led_beacon
arduino-cli compile -b arduino:avr:nano:cpu=atmega328 .
arduino-cli upload  -p /dev/cu.usbserial-XXXX -b arduino:avr:nano:cpu=atmega328 .
```

(albo Arduino IDE: Board Arduino Nano, Processor ATmega328P / Old Bootloader).
Serial Monitor 115200, „Newline”: `ID 5` (0..127), `HZ 10`, `ON`, `RUN`, `?`.

### 3. Test na żywo

```bash
cd vision_id
uv sync
uv run verify_live.py
```

Na dole ekranu `DETECTED: MATCHING ID 5`, gdy kamera odczyta to samo, co nadaje
Arduino. Klawisz `d` przełącza podgląd „co widzi maszyna”:
niebieskie = plamki, zielone = właśnie zapalone, czerwone = właśnie zgasłe,
po prawej wykresy jasności torów z dopasowaniem dekodera. `+`/`-` zmienia próg plamki.

### 4. Nagranie (np. telefon 60 fps)

```bash
uv run receiver.py --video nagranie.mov --chip-hz 20 --debug --out wynik.mp4
```

### 5. Bez sprzętu

```bash
uv run simulate.py                      # 4 drony, 30 fps, wynik w konsoli
uv run --group dev pytest -q tests
```

### 6. Raspberry Pi 5 + kamera IMX477 (`pi/`)

Stanowisko testowe: Pi 5 (AI HAT+ Hailo-8), IMX477 z obiektywem M12 90°, Arduino z nadajnikiem
na USB Pi. Kamera 1332x990 @ 60 fps, **ręczna ekspozycja 250 µs**, gain 1, stały balans bieli -
tło prawie czarne, dioda dalej świeci (programowy zamiennik filtra UV z UVDAR).

```
pi/station.py + station.html   strona http://<pi>:8000: podgląd, "Nagraj" z dźwiękami w przeglądarce,
                               lista nagrań z wynikiem dekodera, odtwarzanie z nakładką, pobieranie, usuwanie
pi/decode_rec.py               dekoder na nagraniu (station.py woła go sam po każdym nagraniu)
pi/record.py, sweep.py         nagranie z wiersza poleceń / seria czasów naświetlania
pi/focus.py, ard.py            ustawianie ostrości obiektywu / komendy dla nadajnika
```

Na Pi do `~/vision` kopiujemy `pi/*` oraz `detector.py tracking.py protocol.py codebook.py receiver.py camera.py`,
potem `python3 station.py`. Kamerę może mieć tylko jeden program naraz.

Ustawienia dekodera dla tej kamery: `--preset pi_niebo` (`detector.PRESETS`):
- `led_color="nonblue"`: mapa max(R, G) - B - niebieskie niebo i białe ściany ~0, czerwona/zielona dioda jasna,
- `proc_width=1332`: pełna rozdzielczość (dioda ma 2-3 px; po pomniejszeniu do 960 px ginie),
- `point_lock`: jak w UVDAR - tor przewiduje położenie (filtr Kalmana, stała prędkość) i w każdej
  klatce przeskakuje na jasny punkt diody obok; `signal="feat"`: sygnał z tego punktu w mapie koloru.

Ocena nagrania: `uv run eval_rec.py NAGRANIE --id 5 --preset pi_niebo`

Wyniki z 2026-11-03 (ok. 2 m, ID 5, 20 Hz):

| nagranie | pierwsze ID | klatki z ID potem | fałszywe ID |
|---|---|---|---|
| pokój, tło ściany | 1.1 s | 100% | brak |
| tło nieba (zmierzch), diody w kamerę | 1.3 s | 100% | brak |
| tło nieba, słabe diody / bokiem | 1.1 s | 55% (część czasu diod nie widać) | brak |
| tło nieba, ręka w ruchu | 1.1 s | 88% | brak |

Na Pi dekoder liczy ~29 kl/s (z `motion_comp=False`; pomiar sprzed zamiany przepływu optycznego na filtr Kalmana), więc nagrania dekoduje po fakcie;
31 z 34 ms na klatkę to operacje na całym obrazie (mapa koloru, energia, szczyty) - do 60 fps na żywo
trzeba je rozdzielić na rdzenie albo liczyć tylko w wycinkach.

## Ograniczenia i co dalej

- Docelowo diody IR + kamera, która je widzi; algorytm działa na jasności, więc się nie zmienia.
- Na dronie: kompensacja obrotu kamery z żyroskopu, przepisanie gorących miejsc na C++.
- UWB może podać listę sąsiadów (`TemplateDecoder(candidates=...)`) → szybsze i pewniejsze ID.
- Orientacja drona (solvePnP z 4 diod) nie jest zrobiona — do kąta namiaru niepotrzebna.
