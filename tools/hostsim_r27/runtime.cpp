#include "runtime.h"
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>
#include <atomic>
#include <cstring>
#include <chrono>
#include <sys/syscall.h>
#include <unistd.h>

namespace {
std::mutex gate_mutex;
std::condition_variable gate_cv;
bool paused=false;
std::atomic<uint64_t> submitted{0},completed{0},errors{0},workers{0},records{0},waits{0};
enum FaultOperation {NoFault=0,EventRecord=1,EventWait=2,StreamQuery=3,EventQuery=4};
std::atomic<int> fault_operation{NoFault};
std::atomic<bool> fault_after{false};
std::atomic<uint64_t> fault_nth{0},fault_seen{0},injected_faults{0};
std::atomic<int64_t> fault_remaining{0};
struct Fault {bool before=false,after=false;};
Fault fault_for(FaultOperation operation) noexcept {
    if(fault_operation.load(std::memory_order_acquire)!=operation)return {};
    if(fault_seen.fetch_add(1)+1<fault_nth.load())return {};
    auto remaining=fault_remaining.load();
    while(remaining>0&&!fault_remaining.compare_exchange_weak(remaining,remaining-1)){}
    if(remaining==0)return {};
    const bool after=fault_after.load();
    if(remaining==1)fault_operation.store(NoFault,std::memory_order_release);
    ++injected_faults;return {!after,after};
}
thread_local std::string last_error,diagnostic;
thread_local bool diagnostic_oom=false;
struct Stream {
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::pair<uint64_t,std::function<void()>>> queue;
    uint64_t next=0,done=0;
    bool stop=false,test_paused=false,failed=false;
    std::atomic<uint64_t> tid{0};
    std::string error;
    std::thread worker;
    ~Stream() {
        {std::lock_guard<std::mutex> lock(mutex);stop=true;cv.notify_all();}
        if(worker.joinable())worker.join();
    }
    uint64_t push(std::function<void()> function) {
        std::unique_lock<std::mutex> lock(mutex);
        if(failed)throw std::runtime_error(error.empty()?"hostsim worker failed":error);
        cv.wait(lock,[&]{return queue.size()<4096||failed;});
        if(failed)throw std::runtime_error(error.empty()?"hostsim worker failed":error);
        if(!worker.joinable()) {
            worker=std::thread([this]{
                tid=static_cast<uint64_t>(syscall(SYS_gettid));
                for(;;) {
                    std::pair<uint64_t,std::function<void()>> item;
                    {std::unique_lock<std::mutex> lock(mutex);cv.wait(lock,[&]{return stop||(!test_paused&&!queue.empty());});
                     if(stop&&queue.empty())break;item=std::move(queue.front());queue.pop_front();cv.notify_all();}
                    {std::unique_lock<std::mutex> lock(gate_mutex);gate_cv.wait(lock,[]{return !paused;});}
                    bool execute=true;{std::lock_guard<std::mutex> lock(mutex);execute=!failed;}
                    try {if(execute)item.second();}
                    catch(const std::exception& exc) {std::lock_guard<std::mutex> lock(mutex);failed=true;try{error=exc.what();}catch(...){}++errors;}
                    catch(...) {std::lock_guard<std::mutex> lock(mutex);failed=true;try{error="unknown worker error";}catch(...){}++errors;}
                    {std::lock_guard<std::mutex> lock(mutex);done=item.first;++completed;cv.notify_all();}
                }
                --workers;
            });
            ++workers;
        }
        const auto ticket=next+1;queue.emplace_back(ticket,std::move(function));next=ticket;++submitted;cv.notify_all();return ticket;
    }
    uint64_t tail(){std::lock_guard<std::mutex> lock(mutex);return next;}
    bool ready(uint64_t ticket){std::lock_guard<std::mutex> lock(mutex);if(failed)throw std::runtime_error(error.empty()?"hostsim worker failed":error);return done>=ticket;}
    void wait(uint64_t ticket) {
        std::unique_lock<std::mutex> lock(mutex);
        if(!cv.wait_for(lock,std::chrono::seconds(60),[&]{return done>=ticket;}))throw std::runtime_error("hostsim stream dependency timeout");
        if(failed)throw std::runtime_error(error.empty()?"hostsim worker failed":error);
    }
};
struct Event {std::shared_ptr<Stream> stream;uint64_t ticket=0,generation=0;};
struct State {
    std::mutex mutex;
    uintptr_t token=16;
    std::map<void*,std::shared_ptr<Stream>> streams{{nullptr,std::make_shared<Stream>()}};
    std::map<void*,std::shared_ptr<Event>> events;
};
State state;
std::shared_ptr<Stream> stream(void* handle) {
    std::lock_guard<std::mutex> lock(state.mutex);auto it=state.streams.find(handle);
    if(it==state.streams.end())throw std::runtime_error("unknown/stale stream");return it->second;
}
std::pair<std::shared_ptr<Stream>,uint64_t> event_ticket(void* handle) {
    std::lock_guard<std::mutex> lock(state.mutex);auto it=state.events.find(handle);
    if(it==state.events.end())throw std::runtime_error("unknown/stale event");
    return {it->second->stream,it->second->ticket};
}
template<class F> int checked(F function) noexcept {try{function();return 0;}catch(const std::exception& exc){return r27::failure(exc.what());}catch(...){return r27::failure("unknown runtime error");}}
}
namespace r27 {
int failure(const char* message) noexcept {try{last_error=message;diagnostic_oom=false;}catch(...){diagnostic_oom=true;}++errors;return 100001;}
int failure(const std::string& message) noexcept {return failure(message.c_str());}
uint64_t enqueue(void* handle,std::function<void()> function){return stream(handle)->push(std::move(function));}
void synchronize(void* handle){auto value=stream(handle);value->wait(value->tail());}
void synchronize_device(){std::vector<std::shared_ptr<Stream>> values;{std::lock_guard<std::mutex> lock(state.mutex);for(auto& item:state.streams)values.push_back(item.second);}for(auto& value:values)value->wait(value->tail());}
}
extern "C" {
int R27StreamCreate(void** out){return checked([&]{if(!out)throw std::runtime_error("null stream output");std::lock_guard<std::mutex> lock(state.mutex);*out=reinterpret_cast<void*>(state.token+=16);state.streams[*out]=std::make_shared<Stream>();});}
int R27StreamDestroy(void* handle){return checked([&]{if(!handle)throw std::runtime_error("cannot destroy default stream");r27::synchronize(handle);std::lock_guard<std::mutex> lock(state.mutex);if(!state.streams.erase(handle))throw std::runtime_error("stale stream");});}
int R27StreamSync(void* handle){return checked([&]{r27::synchronize(handle);});}
int R27DeviceSync(){return checked([]{r27::synchronize_device();});}
int R27StreamQuery(void* handle,int* out){if(out)*out=1;if(fault_for(StreamQuery).before)return r27::failure("injected stream query failure");return checked([&]{if(!out)throw std::runtime_error("null query output");auto value=stream(handle);*out=value->ready(value->tail())?0:1;});}
int R27EventCreate(void** out){return checked([&]{if(!out)throw std::runtime_error("null event output");std::lock_guard<std::mutex> lock(state.mutex);*out=reinterpret_cast<void*>(state.token+=16);state.events[*out]=std::make_shared<Event>();});}
int R27EventDestroy(void* handle){return checked([&]{std::lock_guard<std::mutex> lock(state.mutex);if(!state.events.erase(handle))throw std::runtime_error("stale event");});}
int R27EventRecord(void* handle,void* target){const auto fault=fault_for(EventRecord);if(fault.before)return r27::failure("injected event record failure before enqueue");const auto result=checked([&]{auto value=stream(target);auto ticket=value->push([]{});std::lock_guard<std::mutex> lock(state.mutex);auto it=state.events.find(handle);if(it==state.events.end())throw std::runtime_error("stale event");it->second->stream=value;it->second->ticket=ticket;++it->second->generation;++records;});return result==0&&fault.after?r27::failure("injected event record failure after enqueue"):result;}
int R27EventWait(void* target,void* handle){const auto fault=fault_for(EventWait);if(fault.before)return r27::failure("injected event wait failure before enqueue");const auto result=checked([&]{auto item=event_ticket(handle);if(!item.first)throw std::runtime_error("wait on unrecorded event");r27::enqueue(target,[item]{item.first->wait(item.second);});++waits;});return result==0&&fault.after?r27::failure("injected event wait failure after enqueue"):result;}
int R27EventSync(void* handle){return checked([&]{auto item=event_ticket(handle);if(item.first)item.first->wait(item.second);});}
int R27EventQuery(void* handle,int* out){if(out)*out=0;if(fault_for(EventQuery).before)return r27::failure("injected event query failure");return checked([&]{if(!out)throw std::runtime_error("null event query");auto item=event_ticket(handle);*out=(!item.first||item.first->ready(item.second))?1:0;});}
int R27CopyAsync(void* dst,size_t maximum,const void* src,size_t count,void* target){return checked([&]{if(count>maximum||(!dst&&count)||(!src&&count))throw std::runtime_error("invalid memcpy");r27::enqueue(target,[=]{if(count)std::memcpy(dst,src,count);});});}
int R27Copy(void* dst,size_t maximum,const void* src,size_t count){return checked([&]{if(count>maximum||(!dst&&count)||(!src&&count))throw std::runtime_error("invalid memcpy");r27::synchronize_device();if(count)std::memcpy(dst,src,count);});}
int R27HostSimSetPaused(bool value){std::lock_guard<std::mutex> lock(gate_mutex);paused=value;gate_cv.notify_all();return 0;}
int R27HostSimSetStreamPaused(void* handle,bool value){return checked([&]{auto item=stream(handle);std::lock_guard<std::mutex> lock(item->mutex);item->test_paused=value;item->cv.notify_all();});}
int R27HostSimInjectFault(const char* operation,uint64_t nth,bool after,int64_t repetitions){return checked([&]{
    if(!operation||nth==0||repetitions==0||repetitions < -1)throw std::runtime_error("invalid fault configuration");
    FaultOperation kind=NoFault;
    if(!std::strcmp(operation,"event_record"))kind=EventRecord;
    else if(!std::strcmp(operation,"event_wait"))kind=EventWait;
    else if(!std::strcmp(operation,"stream_query"))kind=StreamQuery;
    else if(!std::strcmp(operation,"event_query"))kind=EventQuery;
    else throw std::runtime_error("unknown fault operation");
    if(after&&(kind==StreamQuery||kind==EventQuery))throw std::runtime_error("query has no enqueue phase");
    fault_operation.store(NoFault,std::memory_order_release);fault_nth=nth;fault_seen=0;fault_after=after;fault_remaining=repetitions;fault_operation.store(kind,std::memory_order_release);
});}
int R27HostSimClearFault(){fault_operation.store(NoFault,std::memory_order_release);return 0;}
const char* R27HostSimLastError(){return diagnostic_oom?"hostsim diagnostic allocation failed":last_error.c_str();}
const char* R27HostSimDiagnosticsJson(){std::lock_guard<std::mutex> lock(state.mutex);std::ostringstream out;out<<"{\"backend_kind\":\"HOST_SIM_R27_CPU\",\"numeric_supported\":true,\"real_sdk_qualified\":false,\"submitted\":"<<submitted<<",\"completed\":"<<completed<<",\"errors\":"<<errors<<",\"workers\":"<<workers<<",\"event_records\":"<<records<<",\"event_waits\":"<<waits<<",\"injected_faults\":"<<injected_faults<<",\"fault_active\":"<<(fault_operation.load()!=NoFault?"true":"false")<<",\"live_events\":"<<state.events.size()<<",\"live_streams\":"<<state.streams.size()<<",\"streams\":[";bool comma=false;for(const auto& item:state.streams){std::lock_guard<std::mutex> stream_lock(item.second->mutex);if(comma)out<<',';comma=true;out<<"{\"handle\":"<<reinterpret_cast<uintptr_t>(item.first)<<",\"worker_tid\":"<<item.second->tid<<",\"submitted\":"<<item.second->next<<",\"completed\":"<<item.second->done<<'}';}out<<"]}";diagnostic=out.str();return diagnostic.c_str();}
}
