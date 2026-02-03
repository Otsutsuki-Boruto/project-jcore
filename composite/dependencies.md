# **Dependencies**

```bash
sudo apt update

# OpenBLAS (development and runtime)
sudo apt install libopenblas-dev libopenblas-openmp-dev

# BLIS (development headers + library)
sudo apt install libblis-dev libblis-openmp-dev

# LIBXSMM (if packaged)
sudo apt install libxsmm-dev

# Prerequisites

sudo apt update
sudo apt install build-essential cmake git gfortran pkg-config

```

Alternatively, build from source (recommended for performance / latest features)

```bash
sudo apt update
sudo apt install build-essential gfortran pkg-config

# OpenBLAS

git clone git@github.com:xianyi/OpenBLAS.git
cd OpenBLAS
make -j"$(nproc)"
sudo make install
cd ..

# BLIS

# 1. Clone using SSH (you already did this)
git clone git@github.com:flame/blis.git
cd blis

# 2. Configure build
# Replace "auto" with a specific architecture for best performance
# Common options: haswell, zen4, skylake, generic
./configure auto

# Example for AVX2 Haswell CPU:
# ./configure haswell

# 3. Build
make -j"$(nproc)"

# 4. Install system-wide
sudo make install

# LIBXSMM

echo "[+] Cloning and building LIBXSMM (SSH)..."
git clone git@github.com:libxsmm/libxsmm.git
cd libxsmm
make -j"$(nproc)" STATIC=0
sudo make install PREFIX=/usr/local
cd ..

echo "[+] Updating dynamic linker cache..."
sudo ldconfig

echo "[✓] Installation complete. OpenBLAS, BLIS, and LIBXSMM are now available system-wide."

# Optionally you can:
# make PREFIX=/usr/local install

```
