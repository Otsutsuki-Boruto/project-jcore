#include "kernels_blis.h"
#include "blis/cblas.h"
#include <blis/blis.h>
#include <cstddef>

extern "C" void blis_sgemm(const float *A, const float *B, float *C,
                           size_t M, size_t N, size_t K)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // CBLAS expects column-major; for row-major we swap matrices and dimensions
    cblas_sgemm(
        CblasRowMajor,       // Row-major matrices
        CblasNoTrans,        // A not transposed
        CblasNoTrans,        // B not transposed
        static_cast<int>(M), // rows of A
        static_cast<int>(N), // cols of B
        static_cast<int>(K), // shared dimension
        alpha,               // scaling factor
        A,                   // matrix A
        static_cast<int>(K), // lda = leading dim of A (row-major: K)
        B,                   // matrix B
        static_cast<int>(N), // ldb = leading dim of B (row-major: N)
        beta,                // scaling factor for C
        C,                   // matrix C
        static_cast<int>(N)  // ldc = leading dim of C (row-major: N)
    );
}