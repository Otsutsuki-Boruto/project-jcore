#!/bin/bash

# Execute the following in Terminal First. Change the path accordingly.
export LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export LD_LIBRARY_PATH=/mnt/localdisk/AI_Engineering_Career/JCoreCustoms/advanced/lib/shared:/mnt/localdisk/AI_Engineering_Career/JCoreCustoms/advanced/llvm/lib/shared:$LD_LIBRARY_PATH