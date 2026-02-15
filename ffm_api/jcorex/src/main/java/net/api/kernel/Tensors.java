package net.api.kernel;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemoryLayout;
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static net.api.kernel.Config.bind;

/**
 * Tensors class contains public methods handles for creating, allocating, and freeing tensors
 * */
public class Tensors {
    private static final int NPL_MAX_DIMS = 8;

    private Tensors() {} /*NO ARG CONSTRUCTOR MADE PRIVATE*/

    /* ================================= Layouts ======================================= */

    /* npl_tensor_t */
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

    /* npl_perf_stats_t */
    static final MemoryLayout NPL_PERF = MemoryLayout.structLayout(
            JAVA_DOUBLE.withName("elapsed_ms"),
            JAVA_DOUBLE.withName("gflops"),
            JAVA_DOUBLE.withName("bandwidth_gbps"),
            JAVA_LONG.withName("operations_fused"),
            JAVA_LONG.withName("memory_saved_bytes"),
            ADDRESS.withName("backend_used"),
            JAVA_INT.withName("was_fused")
    );

    /* ================================= Native Bindings ======================================= */

    static final MethodHandle createTensor =
            bind("npl_create_tensor", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_INT, JAVA_INT, ADDRESS));

    static final MethodHandle allocTensor =
            bind("npl_allocate_tensor",FunctionDescriptor.of(JAVA_INT, ADDRESS));

    static final MethodHandle freeTensor =
            bind("npl_free_tensor", FunctionDescriptor.ofVoid(ADDRESS));
}
