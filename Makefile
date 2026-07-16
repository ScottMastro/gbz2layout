# gbz2layout — build the tool + probe against the vendored prefix.
PREFIX  := $(CURDIR)/local
CXX     := g++
# Target ISA. Defaults to -march=native (fastest on the build host), but that
# emits instructions the build CPU has and others may not: building on a login
# node and running on older compute nodes dies with SIGILL at startup. On a
# heterogeneous cluster override with a portable baseline, e.g.
#   make tool-nocuda ARCH="-march=x86-64-v2"     (gcc 11+; sse4.2+popcnt)
#   make tool-nocuda ARCH="-msse4.2"             (older gcc)
# NB: sdsl-lite hardcodes -march=native in its CMakeLists too, so it must be
# rebuilt with the same baseline or it will SIGILL regardless of this flag.
ARCH    ?= -march=native
CXXFLAGS:= -std=c++17 -O3 -fopenmp -pthread $(ARCH) -Wall -I$(PREFIX)/include
# Link libhandlegraph statically (prefix ships both .a and .so) so we don't need
# LD_LIBRARY_PATH at runtime. Order matters for static linking.
LIBDIR  := $(PREFIX)/lib
LIBS    := -Wl,-Bstatic -lgbwtgraph -lgbwt -lhandlegraph -lsdsl -ldivsufsort -ldivsufsort64 \
           -Wl,-Bdynamic -fopenmp -pthread
LDFLAGS := -L$(LIBDIR)

# CUDA (optional GPU backend for the minibatch). RTX 4050 = sm_89.
NVCC     := nvcc
NVCCFLAGS:= -std=c++17 -O3 -arch=sm_89 -Isrc
CUDALIBS := -lcudart -lcuda

BUILD   := build
SRC     := src

.PHONY: all clean probe tool tool-nocuda test test-components

all: $(BUILD)/gbz2layout $(BUILD)/xp_probe

# GPU backend (nvcc-compiled object; isolated from the templated headers)
$(BUILD)/sgd_minibatch_gpu.o: $(SRC)/sgd_minibatch_gpu.cu $(SRC)/sgd_minibatch_gpu.hpp $(SRC)/third_party/dirty_zipfian_int_distribution.h | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -c $(SRC)/sgd_minibatch_gpu.cu -o $@

# main tool
$(BUILD)/gbz2layout: $(SRC)/gbz2layout.cpp $(SRC)/xp.cpp $(SRC)/xp.hpp $(SRC)/sgd_layout.cpp $(SRC)/sgd_layout.hpp $(SRC)/sgd_minibatch.cpp $(SRC)/sgd_minibatch.hpp $(SRC)/sgd_minibatch_gpu.hpp $(BUILD)/sgd_minibatch_gpu.o $(SRC)/export_gbz.cpp $(SRC)/export_gbz.hpp $(SRC)/odgi_lay.cpp $(SRC)/odgi_lay.hpp $(SRC)/components.cpp $(SRC)/components.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $(SRC)/gbz2layout.cpp $(SRC)/xp.cpp $(SRC)/sgd_layout.cpp $(SRC)/sgd_minibatch.cpp $(SRC)/export_gbz.cpp $(SRC)/odgi_lay.cpp $(SRC)/components.cpp $(BUILD)/sgd_minibatch_gpu.o $(LDFLAGS) $(LIBS) $(CUDALIBS) -o $@

# CUDA-free build for hosts without a GPU (e.g. cluster compute nodes). Links a
# stub GpuLayout (available()==false) instead of the nvcc object, so no CUDA
# runtime is required. --gpu is unavailable in this build; everything else
# (layout on CPU, --export-gbz, --export-all-gbz) works identically.
tool-nocuda: $(BUILD)/gbz2layout-nocuda
$(BUILD)/gbz2layout-nocuda: $(SRC)/gbz2layout.cpp $(SRC)/xp.cpp $(SRC)/xp.hpp $(SRC)/sgd_layout.cpp $(SRC)/sgd_layout.hpp $(SRC)/sgd_minibatch.cpp $(SRC)/sgd_minibatch.hpp $(SRC)/sgd_minibatch_gpu.hpp $(SRC)/sgd_minibatch_gpu_stub.cpp $(SRC)/export_gbz.cpp $(SRC)/export_gbz.hpp $(SRC)/odgi_lay.cpp $(SRC)/odgi_lay.hpp $(SRC)/components.cpp $(SRC)/components.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $(SRC)/gbz2layout.cpp $(SRC)/xp.cpp $(SRC)/sgd_layout.cpp $(SRC)/sgd_minibatch.cpp $(SRC)/export_gbz.cpp $(SRC)/odgi_lay.cpp $(SRC)/components.cpp $(SRC)/sgd_minibatch_gpu_stub.cpp $(LDFLAGS) $(LIBS) -o $@

# component + packing self-test (synthetic multi-component GBZs)
$(BUILD)/test_components: $(SRC)/test_components.cpp $(SRC)/components.cpp $(SRC)/components.hpp $(SRC)/xp.cpp $(SRC)/xp.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $(SRC)/test_components.cpp $(SRC)/components.cpp $(SRC)/xp.cpp $(LDFLAGS) $(LIBS) -o $@

test-components: $(BUILD)/test_components
	./$(BUILD)/test_components

# xp self-test
$(BUILD)/test_xp: $(SRC)/test_xp.cpp $(SRC)/xp.cpp $(SRC)/xp.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $(SRC)/test_xp.cpp $(SRC)/xp.cpp $(LDFLAGS) $(LIBS) -o $@

# layout quality eval
$(BUILD)/eval_layout: $(SRC)/eval_layout.cpp $(SRC)/xp.cpp $(SRC)/xp.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $(SRC)/eval_layout.cpp $(SRC)/xp.cpp $(LDFLAGS) $(LIBS) -o $@

eval: $(BUILD)/eval_layout

# crossing-count (legibility) metric — pure geometry, no GBZ needed
$(BUILD)/eval_crossings: $(SRC)/eval_crossings.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $(SRC)/eval_crossings.cpp -o $@

crossings: $(BUILD)/eval_crossings

# original probe (kept)
$(BUILD)/xp_probe: $(SRC)/xp_probe.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $(SRC)/xp_probe.cpp $(LDFLAGS) $(LIBS) -o $@

test: $(BUILD)/test_xp
probe: $(BUILD)/xp_probe
tool: $(BUILD)/gbz2layout

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -f $(BUILD)/gbz2layout $(BUILD)/gbz2layout-nocuda $(BUILD)/test_xp $(BUILD)/test_components $(BUILD)/xp_probe
