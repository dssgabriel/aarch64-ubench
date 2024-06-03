CXX = clang++
CFLAGS += -O3

aarch64-ubench: aarch64_ubench.S aarch64_ubench.cpp
	$(CXX) $(CFLAGS) -march=armv8-a+sve $^ -o $@

clean:
	@rm -fr aarch64-ubench
