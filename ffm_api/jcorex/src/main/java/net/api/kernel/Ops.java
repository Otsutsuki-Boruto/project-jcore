package net.api.kernel;

import java.lang.foreign.FunctionDescriptor;
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.*;
import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static net.api.kernel.Config.bind;

/**
 * Ops class contains public methods handles for operations of tensors
 * */
public class Ops {

    private Ops() {} /*NO ARG CONSTRUCTOR MADE PRIVATE*/

    /* ================================= Layouts ======================================= */

    static final MethodHandle matmul =
            bind("npl_matmul", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_FLOAT, JAVA_FLOAT, ADDRESS));
}
