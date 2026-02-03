# **Build Sleef**

### Download and Build

```bash
# Clone SLEEF repository
git clone https://github.com/shibatch/sleef.git
cd sleef

# Create build directory
mkdir build && cd build

# Configure with optimizations
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TESTS=OFF

# Build (use all cores)
sudo make -j$(nproc)

# Install
sudo make install

# Update library cache
sudo ldconfig
```

### Verify Installation

```bash
# Check library
ls -la /usr/local/lib/libsleef*

# Check headers
ls -la /usr/local/include/sleef.h

# Test linking
echo '#include ' | gcc -x c - -lsleef -o /dev/null 2>&1 && echo "SLEEF OK"
```
