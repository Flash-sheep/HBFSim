// Optional, passive Driver API observer. No constructor and no CUDA API calls.
#include <cupti.h>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <dlfcn.h>

namespace {
std::mutex mutex;
std::ofstream output;
CUpti_SubscriberHandle subscriber{};
bool active = false;
std::atomic<unsigned long long> driver_callbacks{0}, recorded{0}, write_errors{0};

std::string quoted(const char* value) {
    std::ostringstream text; text << '"';
    if (value) for (size_t i=0; value[i] && i<16384; ++i) {
        const auto c = static_cast<unsigned char>(value[i]);
        if (c=='"' || c=='\\') text << '\\' << char(c);
        else if (c>=32 && c<127) text << char(c);
        else text << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c);
    }
    text << '"'; return text.str();
}

std::string pointer(const void* value) {
    if (!value) return "null";
    std::ostringstream text; text << '"' << value << '"'; return text.str();
}

void CUPTIAPI callback(void*, CUpti_CallbackDomain domain, CUpti_CallbackId id, const void* data) {
    if (domain != CUPTI_CB_DOMAIN_DRIVER_API || !data) return;
    ++driver_callbacks;
    const auto& d = *static_cast<const CUpti_CallbackData*>(data);
    const bool launch = d.functionName && std::strstr(d.functionName, "Launch");
    const bool lookup = id == CUPTI_DRIVER_TRACE_CBID_cuModuleGetFunction;
    if (!launch && !lookup) return;
    const void* function = nullptr;
    const void* module = nullptr;
    const char* lookup_name = nullptr;
    bool supported = false;
    if (d.functionParams && (id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel ||
                            id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz)) {
        function = static_cast<const cuLaunchKernel_params*>(d.functionParams)->f;
        supported = true;
    } else if (d.functionParams && (id == CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel ||
                                   id == CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz)) {
        function = static_cast<const cuLaunchCooperativeKernel_params*>(d.functionParams)->f;
        supported = true;
    } else if (lookup && d.functionParams) {
        const auto& p = *static_cast<const cuModuleGetFunction_params*>(d.functionParams);
        module = p.hmod; lookup_name = p.name;
        if (d.callbackSite == CUPTI_API_EXIT && d.functionReturnValue &&
            *static_cast<const CUresult*>(d.functionReturnValue) == CUDA_SUCCESS && p.hfunc)
            function = *p.hfunc;
        supported = true;
    }
    // DSO of a lookup's symbol string is supplementary evidence only. Heap or
    // driver-owned strings legitimately have no host-container identity here.
    Dl_info info{};
    const char* name_dso = lookup_name && dladdr(lookup_name, &info) ? info.dli_fname : "";
    std::ostringstream line;
    line << "{\"schema_version\":1,\"event\":" << quoted(launch ? "driver_launch" : "module_function_lookup")
         << ",\"cbid\":" << id << ",\"api\":" << quoted(d.functionName)
         << ",\"site\":" << quoted(d.callbackSite == CUPTI_API_ENTER ? "ENTER" : "EXIT")
         << ",\"correlation_id\":" << d.correlationId << ",\"context_uid\":" << d.contextUid
         << ",\"context\":" << pointer(d.context) << ",\"function\":" << pointer(function)
         << ",\"module\":" << pointer(module) << ",\"kernel\":" << quoted(d.symbolName)
         << ",\"lookup_name\":" << quoted(lookup_name) << ",\"lookup_name_dso\":" << quoted(name_dso)
         << ",\"supported_parameter_layout\":" << (supported ? "true" : "false")
         << ",\"result\":";
    if (d.callbackSite == CUPTI_API_EXIT && d.functionReturnValue)
        line << int(*static_cast<const CUresult*>(d.functionReturnValue));
    else line << "null";
    line << "}\n";
    std::lock_guard<std::mutex> guard(mutex);
    output << line.str(); output.flush();
    ++recorded;
    if (!output) ++write_errors;
}
}

extern "C" int llama2_cupti_observer_start(const char* path) {
    if (active || !path || !*path) return -1;
    output.open(path, std::ios::out | std::ios::app);
    if (!output) return -2;
    auto status = cuptiSubscribe(&subscriber, callback, nullptr);
    if (status != CUPTI_SUCCESS) { output.close(); return int(status); }
    status = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
    if (status != CUPTI_SUCCESS) { cuptiUnsubscribe(subscriber); output.close(); return int(status); }
    active = true; return 0;
}

extern "C" int llama2_cupti_observer_stop(unsigned long long* callbacks,
                                          unsigned long long* rows,
                                          unsigned long long* errors) {
    if (!active) return -1;
    const auto status = cuptiUnsubscribe(subscriber);
    if (callbacks) *callbacks = driver_callbacks.load();
    if (rows) *rows = recorded.load();
    if (errors) *errors = write_errors.load();
    active = false; output.close(); return int(status);
}
