CXX ?= g++
CXXFLAGS ?= -O3 -mcpu=native -std=c++17 -Wall -Wextra
JPEG_PREFIX ?= $(shell brew --prefix jpeg-turbo 2>/dev/null)
ifneq ($(JPEG_PREFIX),)
CPPFLAGS += -I$(JPEG_PREFIX)/include
LDFLAGS += -L$(JPEG_PREFIX)/lib
endif

SRC = src/decoder.cpp src/receiver.cpp src/recording.cpp src/ledrx_rec.cpp

build/ledrx_rec: $(SRC) src/*.hpp src/codebook.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $(SRC) $(LDFLAGS) -lturbojpeg -lpthread

clean:
	rm -rf build

.PHONY: clean
