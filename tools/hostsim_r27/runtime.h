#pragma once
#include <cstdint>
#include <functional>
#include <string>
namespace r27 {
uint64_t enqueue(void* stream, std::function<void()> function);
void synchronize(void* stream);
void synchronize_device();
int failure(const std::string& message) noexcept;
int failure(const char* message) noexcept;
}
extern "C" {
int R27StreamCreate(void**);
int R27StreamDestroy(void*);
int R27StreamSync(void*);
int R27DeviceSync();
int R27StreamQuery(void*, int*);
int R27EventCreate(void**);
int R27EventDestroy(void*);
int R27EventRecord(void*,void*);
int R27EventWait(void*,void*);
int R27EventSync(void*);
int R27EventQuery(void*,int*);
int R27CopyAsync(void*,size_t,const void*,size_t,void*);
int R27Copy(void*,size_t,const void*,size_t);
int R27HostSimSetPaused(bool);
int R27HostSimSetStreamPaused(void*,bool);
// Test-only. nth_call is relative to activation. repetitions=-1 persists.
int R27HostSimInjectFault(const char* operation,uint64_t nth_call,bool after_enqueue,int64_t repetitions);
int R27HostSimClearFault();
const char* R27HostSimDiagnosticsJson();
const char* R27HostSimLastError();
}
