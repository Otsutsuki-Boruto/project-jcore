package com.program;

/**
 * Plain Java Matrix Multiplication Benchmark
 * Naive implementation for comparison with FFM neural layer
 */
public class JavaMatMulBenchmark {

    /**
     * Naive matrix multiplication: C = A * B
     * Uses standard triple-nested loop
     */
    public static void matmul(float[][] A, float[][] B, float[][] C) {
        int M = A.length;
        int K = A[0].length;
        int N = B[0].length;

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
    }

    /**
     * Cache-optimized matrix multiplication using blocking/tiling
     * Better performance than naive approach
     */
    public static void matmulBlocked(float[][] A, float[][] B, float[][] C, int blockSize) {
        int M = A.length;
        int K = A[0].length;
        int N = B[0].length;

        // Initialize C to zero
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                C[i][j] = 0.0f;
            }
        }

        // Blocked multiplication
        for (int i0 = 0; i0 < M; i0 += blockSize) {
            for (int j0 = 0; j0 < N; j0 += blockSize) {
                for (int k0 = 0; k0 < K; k0 += blockSize) {
                    // Multiply block
                    int iMax = Math.min(i0 + blockSize, M);
                    int jMax = Math.min(j0 + blockSize, N);
                    int kMax = Math.min(k0 + blockSize, K);

                    for (int i = i0; i < iMax; i++) {
                        for (int j = j0; j < jMax; j++) {
                            float sum = C[i][j];
                            for (int k = k0; k < kMax; k++) {
                                sum += A[i][k] * B[k][j];
                            }
                            C[i][j] = sum;
                        }
                    }
                }
            }
        }
    }

    /**
     * Initialize matrix with deterministic values
     */
    public static void initDeterministic(float[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        int idx = 0;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = (float)((idx % 256) / 255.0);
                idx++;
            }
        }
    }

    /**
     * Run benchmark for given matrix size
     */
    public static void benchmark(int M, int N, int K, int iterations, boolean useBlocking) {
        System.out.printf("Java MatMul [%d x %d x %d] - %s%n",
                M, K, N, useBlocking ? "BLOCKED" : "NAIVE");
        System.out.println("Allocating matrices...");

        float[][] A = new float[M][K];
        float[][] B = new float[K][N];
        float[][] C = new float[M][N];

        System.out.println("Initializing matrices...");
        initDeterministic(A);
        initDeterministic(B);

        System.out.println("Warming up...");
        // Warmup
        if (useBlocking) {
            matmulBlocked(A, B, C, 64);
        } else {
            matmul(A, B, C);
        }

        System.out.println("Running timed iterations...");
        // Timed runs
        long t0 = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            if (useBlocking) {
                matmulBlocked(A, B, C, 64);
            } else {
                matmul(A, B, C);
            }
        }
        long t1 = System.nanoTime();

        double avgMs = (t1 - t0) / 1e6 / iterations;
        double flops = 2.0 * M * N * K;
        double gflops = (flops / 1e9) / (avgMs / 1000.0);

        System.out.println("=" .repeat(50));
        System.out.printf("MatMul [%d x %d x %d]  %.3f ms  %.2f GFLOPS%n",
                M, K, N, avgMs, gflops);
        System.out.println("=" .repeat(50));
        System.out.println();
    }

    public static void main(String[] args) {
        System.out.println("Plain Java Matrix Multiplication Benchmark");
        System.out.println("Java Version: " + System.getProperty("java.version"));
        System.out.println("JVM: " + System.getProperty("java.vm.name"));
        System.out.println();

        // Small test first
        System.out.println("Running warmup with small matrices...");
        benchmark(512, 512, 512, 1, true);

        // Test cases matching FFM benchmark
        int[][] cases = {
                {4096, 4096, 4096, 2},
                {8192, 8192, 8192, 1}
        };

        for (int[] c : cases) {
            int M = c[0], N = c[1], K = c[2], it = c[3];

            System.out.println("\n--- NAIVE IMPLEMENTATION ---");
            // Note: Naive implementation will be VERY slow for 8192
            if (M <= 4096) {
                benchmark(M, N, K, it, false);
            } else {
                System.out.println("Skipping naive implementation for 8192 (too slow)");
                System.out.println();
            }

            System.out.println("--- BLOCKED IMPLEMENTATION ---");
            benchmark(M, N, K, it, true);
        }

        System.out.println("\nNote: For fair comparison with optimized libraries,");
        System.out.println("consider using -XX:+UseAVX2 or other JVM optimizations.");
    }
}