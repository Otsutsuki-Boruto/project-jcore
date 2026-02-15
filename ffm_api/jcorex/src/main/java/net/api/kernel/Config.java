package net.api.kernel;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.*;
import static java.lang.foreign.ValueLayout.ADDRESS;

/**
 * Config class contains public methods for initialization and shutdown of "Neural Layer". It automatically loads the library.
 * */
public class Config {

    private Config() {} // NO OBJECT IS REQUIRED

    /* ================================= Layouts ======================================= */

    /* npl_config_t */
    static final MemoryLayout NPL_CONFIG = MemoryLayout.structLayout(
            JAVA_LONG.withName("num_threads"),       /* 0..7 */
            JAVA_INT.withName("param1"),             /* 8..11 */
            JAVA_INT.withName("param2"),             /* 12..15 */
            JAVA_INT.withName("param3"),             /* 16..19 */
            JAVA_INT.withName("param4"),             /* 20..23 */
            JAVA_INT.withName("param5"),             /* 24..27 */
            JAVA_INT.withName("param6"),             /* 28..31 */
            JAVA_INT.withName("param7"),             /* 32..35 */
            MemoryLayout.paddingLayout(4),   /* 36..39 → pad 4 bytes so next long is at offset 40 */
            JAVA_LONG.withName("param8"),            /* 40..47 */
            JAVA_INT.withName("param9")              /* 48..51 */
    );

    /* ================================= Linker ======================================= */

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup LOOKUP =
            SymbolLookup.libraryLookup("/mnt/localdisk/AI_Engineering_Career/JCoreCustoms/advanced/lib/shared/libneural_layer.so", Arena.global());

    static MethodHandle bind(String name, FunctionDescriptor fd) {
        return LINKER.downcallHandle(LOOKUP.find(name).orElseThrow(), fd);
    }

    /* ================================= Native Bindings ======================================= */

    /**Get default config*/
    public static final MethodHandle getDefaultConfig = bind("npl_get_default_config", FunctionDescriptor.of(JAVA_INT, ADDRESS));

    /**Initialize Npl*/
    public static final MethodHandle initNpl = bind("npl_init", FunctionDescriptor.of(JAVA_INT, ADDRESS));

    /**Shutdown Npl*/
    public static final MethodHandle shutdownNpl = bind("npl_shutdown", FunctionDescriptor.ofVoid());

    /**Check if Npl is Initialized*/
    public static final MethodHandle isNplInitialized = bind("npl_is_initialized", FunctionDescriptor.of(JAVA_INT));

}
