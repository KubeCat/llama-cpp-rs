// Minimal OpenCL stub for cross-compilation linking.
// The real libOpenCL.so is provided by the device GPU driver at runtime
// (typically /system/vendor/lib64/libOpenCL.so on Android).
//
// This stub exports the symbols that the linker needs to resolve during
// the build; they are never called because the dynamic linker replaces
// them with the device's implementation at load time.

typedef int cl_int;
typedef unsigned int cl_uint;
typedef void* cl_platform_id;

cl_int clGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms) {
    (void)num_entries; (void)platforms; (void)num_platforms;
    return -1; // CL_DEVICE_NOT_FOUND
}
