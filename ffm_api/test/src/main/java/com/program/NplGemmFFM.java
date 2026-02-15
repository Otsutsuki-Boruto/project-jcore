package com.program;

import jdk.jfr.MemoryAddress;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.*;

public class NplGemmFFM {

    static final int NPL_MAX_DIMS = 8;

    /* ================= layouts ================= */

    // npl_config_t
    static final MemoryLayout NPL_CONFIG = MemoryLayout.structLayout(
            JAVA_LONG.withName("num_threads"),       // 0..7
            JAVA_INT.withName("param1"),             // 8..11
            JAVA_INT.withName("param2"),             // 12..15
            JAVA_INT.withName("param3"),             // 16..19
            JAVA_INT.withName("param4"),             // 20..23
            JAVA_INT.withName("param5"),             // 24..27
            JAVA_INT.withName("param6"),             // 28..31
            JAVA_INT.withName("param7"),             // 32..35
            MemoryLayout.paddingLayout(4),   // 36..39 → pad 4 bytes so next long is at offset 40
            JAVA_LONG.withName("param8"),            // 40..47
            JAVA_INT.withName("param9")              // 48..51
    );


    // npl_tensor_t
    static final MemoryLayout NPL_TENSOR = MemoryLayout.structLayout(
            ADDRESS.withName("data"),
            JAVA_LONG.withName("ndim"),
            MemoryLayout.sequenceLayout(NPL_MAX_DIMS, JAVA_LONG).withName("shape"),
            MemoryLayout.sequenceLayout(NPL_MAX_DIMS, JAVA_LONG).withName("strides"),
            JAVA_INT.withName("dtype"),
            JAVA_INT.withName("layout"),
            JAVA_LONG.withName("size_bytes"),
            JAVA_INT.withName("is_contiguous"),
            MemoryLayout.paddingLayout(4) // ensure alignment for 64-bit
    );

    // npl_perf_stats_t
    static final MemoryLayout NPL_PERF = MemoryLayout.structLayout(
            JAVA_DOUBLE.withName("elapsed_ms"),
            JAVA_DOUBLE.withName("gflops"),
            JAVA_DOUBLE.withName("bandwidth_gbps"),
            JAVA_LONG.withName("operations_fused"),
            JAVA_LONG.withName("memory_saved_bytes"),
            ADDRESS.withName("backend_used"),
            JAVA_INT.withName("was_fused")
    );

    /* ================= linker ================= */

    static final Linker LINKER = Linker.nativeLinker();
    static final SymbolLookup LOOKUP =
           SymbolLookup.libraryLookup("/mnt/localdisk/AI_Engineering_Career/JCoreCustoms/advanced/lib/shared/libneural_layer.so", Arena.global());

    static MethodHandle bind(String name, FunctionDescriptor fd) {
        return LINKER.downcallHandle(LOOKUP.find(name).orElseThrow(), fd);
    }

    /* ================= native bindings ================= */

    static final MethodHandle getDefault =
            bind("npl_get_default_config",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS));

    static final MethodHandle init =
            bind("npl_init",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS));

    static final MethodHandle createTensor =
            bind("npl_create_tensor",
                    FunctionDescriptor.of(JAVA_INT,
                            ADDRESS, JAVA_LONG, ADDRESS, JAVA_INT, JAVA_INT, ADDRESS));

    static final MethodHandle allocTensor =
            bind("npl_allocate_tensor",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS));

    static final MethodHandle freeTensor =
            bind("npl_free_tensor",
                    FunctionDescriptor.ofVoid(ADDRESS));

    static final MethodHandle matmul =
            bind("npl_matmul",
                    FunctionDescriptor.of(JAVA_INT,
                            ADDRESS, ADDRESS, ADDRESS,
                            JAVA_FLOAT, JAVA_FLOAT,
                            ADDRESS));

    /* ================= deterministic fill ================= */

    static void initDeterministic(MemorySegment tensor, Arena arena) {
        // Get the size in bytes of the tensor
        long sizeBytes = tensor.get(JAVA_LONG,
                NPL_TENSOR.byteOffset(MemoryLayout.PathElement.groupElement("size_bytes"))
        );

        // Get the data pointer as a MemorySegment
        MemorySegment dataAddr = tensor.get(ADDRESS,
                NPL_TENSOR.byteOffset(MemoryLayout.PathElement.groupElement("data"))
        );

        // Reinterpret the address with proper size bounds
        MemorySegment data = dataAddr.reinterpret(sizeBytes, arena, null);

        long n = sizeBytes / 4; // float = 4 bytes

        for (long i = 0; i < n; i++) {
            float v = (float) ((i % 256) / 255.0);
            data.setAtIndex(JAVA_FLOAT, i, v);
        }
    }

    /* ================= main ================= */

    public static void main(String[] args) throws Throwable {
        try (Arena arena = Arena.ofConfined()) {

            /* ---- init ---- */
            MemorySegment config = arena.allocate(NPL_CONFIG);
            int rc1 = (int) getDefault.invokeExact(config);
            int rc2 = (int) init.invokeExact(config);

            int[][] cases = {
                    {4096, 4096, 4096, 2},
                    {8192, 8192, 8192, 1}
            };

            for (int[] c : cases) {
                int M = c[0], N = c[1], K = c[2], it = c[3];

                MemorySegment A = arena.allocate(NPL_TENSOR);
                MemorySegment B = arena.allocate(NPL_TENSOR);
                MemorySegment C = arena.allocate(NPL_TENSOR);

                MemorySegment shapeA = arena.allocateFrom(JAVA_LONG, (long) M, (long) K);
                MemorySegment shapeB = arena.allocateFrom(JAVA_LONG, (long) K, (long) N);
                MemorySegment shapeC = arena.allocateFrom(JAVA_LONG, (long) M, (long) N);

                int rc3 = (int) createTensor.invokeExact(MemorySegment.NULL, 2L, shapeA, 0, 0, A);
                int rc4 = (int) createTensor.invokeExact(MemorySegment.NULL, 2L, shapeB, 0, 0, B);
                int rc5 = (int) createTensor.invokeExact(MemorySegment.NULL, 2L, shapeC, 0, 0, C);

                int rc6 = (int) allocTensor.invokeExact(A);
                int rc7 = (int) allocTensor.invokeExact(B);
                int rc8 = (int) allocTensor.invokeExact(C);

                initDeterministic(A, arena);
                initDeterministic(B, arena);

                MemorySegment stats = arena.allocate(NPL_PERF);

                /* warmup */
                int rc9 = (int) matmul.invokeExact(A, B, C, 1f, 0f, stats);

                /* timed */
                long t0 = System.nanoTime();
                for (int i = 0; i < it; i++){
                  int rcs  =  (int) matmul.invokeExact(A, B, C, 1f, 0f, stats);
                }
                long t1 = System.nanoTime();

                double avgMs = (t1 - t0) / 1e6 / it;
                double flops = 2.0 * M * N * K;
                double gflops = (flops / 1e9) / (avgMs / 1000.0);

                System.out.printf("MatMul [%d x %d x %d]  %.3f ms  %.2f GFLOPS%n",
                        M, K, N, avgMs, gflops);

                freeTensor.invokeExact(A);
                freeTensor.invokeExact(B);
                freeTensor.invokeExact(C);
            }
        }
    }
}
