## Run these before running compilation command:

```bash
export KMP_AFFINITY=granularity=fine,compact,1,0

export MKL_NUM_THREADS=12
export OMP_NUM_THREADS=12
export MKL_DYNAMIC=FALSE

export LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/2025.3/lib/intel64:$LD_LIBRARY_PATH
```

## Compile MKL

```bash
g++ -O3 -march=native project_test/mkl_sgemm_test.cpp -I/opt/intel/oneapi/mkl/2025.3/include \
    -L/opt/intel/oneapi/mkl/2025.3/lib/intel64 -lmkl_rt -lpthread -lm -ldl -o project_test/mkl_test
```
