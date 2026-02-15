For FFM API to work, Load the system library using "System.load" and make the Symbol Lookup loader to look for already loaded libraries. It is
essential for it to load its dependent shared libraries which it can easily. To make it happen, execute the script "run_neural.sh". The path will
get attach to it, and then you can successfully execute and create FFM API.

```bash
# Compile it
javac path/to/file.java

# Execute it
java --enable-native-access=ALL-UNNAMED com.program.NplGemmFFM
```