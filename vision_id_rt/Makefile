CXX ?= g++
CXXFLAGS ?= -O3 -mcpu=native -std=c++17 -Wall -Wextra
JPEG_PREFIX ?= $(shell brew --prefix jpeg-turbo 2>/dev/null)
ifneq ($(JPEG_PREFIX),)
CPPFLAGS += -I$(JPEG_PREFIX)/include
LDFLAGS += -L$(JPEG_PREFIX)/lib
endif

CORE = src/decoder.cpp src/receiver.cpp src/tracking.cpp src/constellation.cpp src/motion.cpp src/fc.cpp
PYTHON ?= python3
PY_MODULE = build/ledrx$(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
PY_LDFLAGS = $(if $(filter Darwin,$(shell uname)),-undefined dynamic_lookup,)

all: build/ledrx_rec

python: $(PY_MODULE)

build/ledrx_rec: $(CORE) src/recording.cpp src/ledrx_rec.cpp src/*.hpp src/codebook.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $(CORE) src/recording.cpp src/ledrx_rec.cpp $(LDFLAGS) -lturbojpeg -lpthread

$(PY_MODULE): $(CORE) src/pyledrx.cpp src/*.hpp src/codebook.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -shared -fPIC -fvisibility=hidden $(shell $(PYTHON) -m pybind11 --includes 2>/dev/null || $(PYTHON)-config --includes) -o $@ $(CORE) src/pyledrx.cpp $(PY_LDFLAGS)

clean:
	rm -rf build

.PHONY: all python clean
