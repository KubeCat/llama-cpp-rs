// opencl_dynload.c — Runtime OpenCL loader via dlopen/dlsym
//
// Provides OpenCL function symbols that forward to the device's libOpenCL.so
// loaded at runtime. This allows cross-compiling without linking against
// libOpenCL.so — the real driver is loaded at first use.
//
// On Android, the GPU vendor's libOpenCL.so lives at a device-specific path
// (e.g. /system/vendor/lib64/libOpenCL.so). This shim tries multiple paths.

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <dlfcn.h>
#include <stddef.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Dynamic loader
// ---------------------------------------------------------------------------

static void *g_opencl_lib = NULL;
static int   g_opencl_init_done = 0;

static void opencl_dynload_init(void) {
    if (g_opencl_init_done) return;
    g_opencl_init_done = 1;

    static const char *paths[] = {
        "libOpenCL.so",
        "libOpenCL.so.1",
        "/system/vendor/lib64/libOpenCL.so",
        "/system/lib64/libOpenCL.so",
        "/vendor/lib64/libOpenCL.so",
        // 32-bit fallbacks
        "/system/vendor/lib/libOpenCL.so",
        "/system/lib/libOpenCL.so",
        "/vendor/lib/libOpenCL.so",
        NULL
    };

    for (int i = 0; paths[i]; i++) {
        g_opencl_lib = dlopen(paths[i], RTLD_NOW | RTLD_LOCAL);
        if (g_opencl_lib) return;
    }

    fprintf(stderr, "opencl_dynload: failed to load libOpenCL.so: %s\n",
            dlerror());
}

static void *opencl_sym(const char *name) {
    opencl_dynload_init();
    if (!g_opencl_lib) return NULL;
    void *sym = dlsym(g_opencl_lib, name);
    if (!sym) {
        fprintf(stderr, "opencl_dynload: symbol %s not found: %s\n",
                name, dlerror());
    }
    return sym;
}

// ---------------------------------------------------------------------------
// Function pointer cache + wrapper macros
// ---------------------------------------------------------------------------

// Each wrapper: on first call, resolve the real symbol via dlsym, cache it,
// then forward all subsequent calls with zero overhead beyond the NULL check.

#define DECL_FP(name) static __typeof__(&name) fp_##name = NULL

// Macro for functions returning cl_int (most CL functions).
// Returns CL_INVALID_PLATFORM (-32) if the library failed to load.
#define LOAD_OR_FAIL(name) do { \
    if (!fp_##name) { \
        fp_##name = (__typeof__(&name))opencl_sym(#name); \
        if (!fp_##name) return -32; \
    } \
} while (0)

// Macro for functions returning a handle (cl_context, cl_mem, etc.).
// Sets errcode_ret to CL_INVALID_PLATFORM and returns NULL.
#define LOAD_OR_NULL(name, err_ptr) do { \
    if (!fp_##name) { \
        fp_##name = (__typeof__(&name))opencl_sym(#name); \
        if (!fp_##name) { if (err_ptr) *(err_ptr) = -32; return NULL; } \
    } \
} while (0)

// ---------------------------------------------------------------------------
// Platform APIs
// ---------------------------------------------------------------------------

cl_int CL_API_CALL clGetPlatformIDs(
        cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms) {
    DECL_FP(clGetPlatformIDs);
    LOAD_OR_FAIL(clGetPlatformIDs);
    return fp_clGetPlatformIDs(num_entries, platforms, num_platforms);
}

cl_int CL_API_CALL clGetPlatformInfo(
        cl_platform_id platform, cl_platform_info param_name,
        size_t param_value_size, void *param_value,
        size_t *param_value_size_ret) {
    DECL_FP(clGetPlatformInfo);
    LOAD_OR_FAIL(clGetPlatformInfo);
    return fp_clGetPlatformInfo(platform, param_name, param_value_size,
                                param_value, param_value_size_ret);
}

// ---------------------------------------------------------------------------
// Device APIs
// ---------------------------------------------------------------------------

cl_int CL_API_CALL clGetDeviceIDs(
        cl_platform_id platform, cl_device_type device_type,
        cl_uint num_entries, cl_device_id *devices, cl_uint *num_devices) {
    DECL_FP(clGetDeviceIDs);
    LOAD_OR_FAIL(clGetDeviceIDs);
    return fp_clGetDeviceIDs(platform, device_type, num_entries, devices,
                             num_devices);
}

cl_int CL_API_CALL clGetDeviceInfo(
        cl_device_id device, cl_device_info param_name,
        size_t param_value_size, void *param_value,
        size_t *param_value_size_ret) {
    DECL_FP(clGetDeviceInfo);
    LOAD_OR_FAIL(clGetDeviceInfo);
    return fp_clGetDeviceInfo(device, param_name, param_value_size,
                              param_value, param_value_size_ret);
}

// ---------------------------------------------------------------------------
// Context APIs
// ---------------------------------------------------------------------------

cl_context CL_API_CALL clCreateContext(
        const cl_context_properties *properties, cl_uint num_devices,
        const cl_device_id *devices,
        void (CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *),
        void *user_data, cl_int *errcode_ret) {
    DECL_FP(clCreateContext);
    LOAD_OR_NULL(clCreateContext, errcode_ret);
    return fp_clCreateContext(properties, num_devices, devices, pfn_notify,
                              user_data, errcode_ret);
}

cl_int CL_API_CALL clReleaseContext(cl_context context) {
    DECL_FP(clReleaseContext);
    LOAD_OR_FAIL(clReleaseContext);
    return fp_clReleaseContext(context);
}

// ---------------------------------------------------------------------------
// Command Queue APIs
// ---------------------------------------------------------------------------

cl_command_queue CL_API_CALL clCreateCommandQueue(
        cl_context context, cl_device_id device,
        cl_command_queue_properties properties, cl_int *errcode_ret) {
    DECL_FP(clCreateCommandQueue);
    LOAD_OR_NULL(clCreateCommandQueue, errcode_ret);
    return fp_clCreateCommandQueue(context, device, properties, errcode_ret);
}

// ---------------------------------------------------------------------------
// Memory Object APIs
// ---------------------------------------------------------------------------

cl_mem CL_API_CALL clCreateBuffer(
        cl_context context, cl_mem_flags flags, size_t size,
        void *host_ptr, cl_int *errcode_ret) {
    DECL_FP(clCreateBuffer);
    LOAD_OR_NULL(clCreateBuffer, errcode_ret);
    return fp_clCreateBuffer(context, flags, size, host_ptr, errcode_ret);
}

cl_mem CL_API_CALL clCreateSubBuffer(
        cl_mem buffer, cl_mem_flags flags,
        cl_buffer_create_type buffer_create_type,
        const void *buffer_create_info, cl_int *errcode_ret) {
    DECL_FP(clCreateSubBuffer);
    LOAD_OR_NULL(clCreateSubBuffer, errcode_ret);
    return fp_clCreateSubBuffer(buffer, flags, buffer_create_type,
                                buffer_create_info, errcode_ret);
}

cl_mem CL_API_CALL clCreateImage(
        cl_context context, cl_mem_flags flags,
        const cl_image_format *image_format,
        const cl_image_desc *image_desc,
        void *host_ptr, cl_int *errcode_ret) {
    DECL_FP(clCreateImage);
    LOAD_OR_NULL(clCreateImage, errcode_ret);
    return fp_clCreateImage(context, flags, image_format, image_desc,
                            host_ptr, errcode_ret);
}

cl_int CL_API_CALL clReleaseMemObject(cl_mem memobj) {
    DECL_FP(clReleaseMemObject);
    LOAD_OR_FAIL(clReleaseMemObject);
    return fp_clReleaseMemObject(memobj);
}

// ---------------------------------------------------------------------------
// Program Object APIs
// ---------------------------------------------------------------------------

cl_program CL_API_CALL clCreateProgramWithSource(
        cl_context context, cl_uint count, const char **strings,
        const size_t *lengths, cl_int *errcode_ret) {
    DECL_FP(clCreateProgramWithSource);
    LOAD_OR_NULL(clCreateProgramWithSource, errcode_ret);
    return fp_clCreateProgramWithSource(context, count, strings, lengths,
                                        errcode_ret);
}

cl_int CL_API_CALL clBuildProgram(
        cl_program program, cl_uint num_devices,
        const cl_device_id *device_list, const char *options,
        void (CL_CALLBACK *pfn_notify)(cl_program, void *),
        void *user_data) {
    DECL_FP(clBuildProgram);
    LOAD_OR_FAIL(clBuildProgram);
    return fp_clBuildProgram(program, num_devices, device_list, options,
                             pfn_notify, user_data);
}

cl_int CL_API_CALL clGetProgramBuildInfo(
        cl_program program, cl_device_id device,
        cl_program_build_info param_name, size_t param_value_size,
        void *param_value, size_t *param_value_size_ret) {
    DECL_FP(clGetProgramBuildInfo);
    LOAD_OR_FAIL(clGetProgramBuildInfo);
    return fp_clGetProgramBuildInfo(program, device, param_name,
                                    param_value_size, param_value,
                                    param_value_size_ret);
}

cl_int CL_API_CALL clReleaseProgram(cl_program program) {
    DECL_FP(clReleaseProgram);
    LOAD_OR_FAIL(clReleaseProgram);
    return fp_clReleaseProgram(program);
}

// ---------------------------------------------------------------------------
// Kernel Object APIs
// ---------------------------------------------------------------------------

cl_kernel CL_API_CALL clCreateKernel(
        cl_program program, const char *kernel_name, cl_int *errcode_ret) {
    DECL_FP(clCreateKernel);
    LOAD_OR_NULL(clCreateKernel, errcode_ret);
    return fp_clCreateKernel(program, kernel_name, errcode_ret);
}

cl_int CL_API_CALL clSetKernelArg(
        cl_kernel kernel, cl_uint arg_index, size_t arg_size,
        const void *arg_value) {
    DECL_FP(clSetKernelArg);
    LOAD_OR_FAIL(clSetKernelArg);
    return fp_clSetKernelArg(kernel, arg_index, arg_size, arg_value);
}

cl_int CL_API_CALL clGetKernelInfo(
        cl_kernel kernel, cl_kernel_info param_name,
        size_t param_value_size, void *param_value,
        size_t *param_value_size_ret) {
    DECL_FP(clGetKernelInfo);
    LOAD_OR_FAIL(clGetKernelInfo);
    return fp_clGetKernelInfo(kernel, param_name, param_value_size,
                              param_value, param_value_size_ret);
}

cl_int CL_API_CALL clGetKernelWorkGroupInfo(
        cl_kernel kernel, cl_device_id device,
        cl_kernel_work_group_info param_name, size_t param_value_size,
        void *param_value, size_t *param_value_size_ret) {
    DECL_FP(clGetKernelWorkGroupInfo);
    LOAD_OR_FAIL(clGetKernelWorkGroupInfo);
    return fp_clGetKernelWorkGroupInfo(kernel, device, param_name,
                                       param_value_size, param_value,
                                       param_value_size_ret);
}

cl_int CL_API_CALL clGetKernelSubGroupInfo(
        cl_kernel kernel, cl_device_id device,
        cl_kernel_sub_group_info param_name, size_t input_value_size,
        const void *input_value, size_t param_value_size,
        void *param_value, size_t *param_value_size_ret) {
    DECL_FP(clGetKernelSubGroupInfo);
    LOAD_OR_FAIL(clGetKernelSubGroupInfo);
    return fp_clGetKernelSubGroupInfo(kernel, device, param_name,
                                      input_value_size, input_value,
                                      param_value_size, param_value,
                                      param_value_size_ret);
}

// ---------------------------------------------------------------------------
// Event Object APIs
// ---------------------------------------------------------------------------

cl_int CL_API_CALL clWaitForEvents(
        cl_uint num_events, const cl_event *event_list) {
    DECL_FP(clWaitForEvents);
    LOAD_OR_FAIL(clWaitForEvents);
    return fp_clWaitForEvents(num_events, event_list);
}

cl_int CL_API_CALL clReleaseEvent(cl_event event) {
    DECL_FP(clReleaseEvent);
    LOAD_OR_FAIL(clReleaseEvent);
    return fp_clReleaseEvent(event);
}

cl_int CL_API_CALL clGetEventProfilingInfo(
        cl_event event, cl_profiling_info param_name,
        size_t param_value_size, void *param_value,
        size_t *param_value_size_ret) {
    DECL_FP(clGetEventProfilingInfo);
    LOAD_OR_FAIL(clGetEventProfilingInfo);
    return fp_clGetEventProfilingInfo(event, param_name, param_value_size,
                                      param_value, param_value_size_ret);
}

// ---------------------------------------------------------------------------
// Flush and Finish APIs
// ---------------------------------------------------------------------------

cl_int CL_API_CALL clFlush(cl_command_queue command_queue) {
    DECL_FP(clFlush);
    LOAD_OR_FAIL(clFlush);
    return fp_clFlush(command_queue);
}

cl_int CL_API_CALL clFinish(cl_command_queue command_queue) {
    DECL_FP(clFinish);
    LOAD_OR_FAIL(clFinish);
    return fp_clFinish(command_queue);
}

// ---------------------------------------------------------------------------
// Enqueued Commands APIs
// ---------------------------------------------------------------------------

cl_int CL_API_CALL clEnqueueReadBuffer(
        cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read,
        size_t offset, size_t size, void *ptr,
        cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
        cl_event *event) {
    DECL_FP(clEnqueueReadBuffer);
    LOAD_OR_FAIL(clEnqueueReadBuffer);
    return fp_clEnqueueReadBuffer(command_queue, buffer, blocking_read,
                                   offset, size, ptr,
                                   num_events_in_wait_list, event_wait_list,
                                   event);
}

cl_int CL_API_CALL clEnqueueWriteBuffer(
        cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write,
        size_t offset, size_t size, const void *ptr,
        cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
        cl_event *event) {
    DECL_FP(clEnqueueWriteBuffer);
    LOAD_OR_FAIL(clEnqueueWriteBuffer);
    return fp_clEnqueueWriteBuffer(command_queue, buffer, blocking_write,
                                    offset, size, ptr,
                                    num_events_in_wait_list, event_wait_list,
                                    event);
}

cl_int CL_API_CALL clEnqueueFillBuffer(
        cl_command_queue command_queue, cl_mem buffer,
        const void *pattern, size_t pattern_size,
        size_t offset, size_t size,
        cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
        cl_event *event) {
    DECL_FP(clEnqueueFillBuffer);
    LOAD_OR_FAIL(clEnqueueFillBuffer);
    return fp_clEnqueueFillBuffer(command_queue, buffer, pattern, pattern_size,
                                   offset, size,
                                   num_events_in_wait_list, event_wait_list,
                                   event);
}

cl_int CL_API_CALL clEnqueueCopyBuffer(
        cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_buffer,
        size_t src_offset, size_t dst_offset, size_t size,
        cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
        cl_event *event) {
    DECL_FP(clEnqueueCopyBuffer);
    LOAD_OR_FAIL(clEnqueueCopyBuffer);
    return fp_clEnqueueCopyBuffer(command_queue, src_buffer, dst_buffer,
                                   src_offset, dst_offset, size,
                                   num_events_in_wait_list, event_wait_list,
                                   event);
}

cl_int CL_API_CALL clEnqueueNDRangeKernel(
        cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim,
        const size_t *global_work_offset, const size_t *global_work_size,
        const size_t *local_work_size,
        cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
        cl_event *event) {
    DECL_FP(clEnqueueNDRangeKernel);
    LOAD_OR_FAIL(clEnqueueNDRangeKernel);
    return fp_clEnqueueNDRangeKernel(command_queue, kernel, work_dim,
                                      global_work_offset, global_work_size,
                                      local_work_size,
                                      num_events_in_wait_list, event_wait_list,
                                      event);
}

cl_int CL_API_CALL clEnqueueMarkerWithWaitList(
        cl_command_queue command_queue,
        cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
        cl_event *event) {
    DECL_FP(clEnqueueMarkerWithWaitList);
    LOAD_OR_FAIL(clEnqueueMarkerWithWaitList);
    return fp_clEnqueueMarkerWithWaitList(command_queue,
                                           num_events_in_wait_list,
                                           event_wait_list, event);
}

cl_int CL_API_CALL clEnqueueBarrierWithWaitList(
        cl_command_queue command_queue,
        cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
        cl_event *event) {
    DECL_FP(clEnqueueBarrierWithWaitList);
    LOAD_OR_FAIL(clEnqueueBarrierWithWaitList);
    return fp_clEnqueueBarrierWithWaitList(command_queue,
                                            num_events_in_wait_list,
                                            event_wait_list, event);
}
