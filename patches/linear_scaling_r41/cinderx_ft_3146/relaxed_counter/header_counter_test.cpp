#include <Python.h>
#include COUNTER_HEADER
#include <atomic>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

#ifndef Py_GIL_DISABLED
#error The actual target FT headers must define Py_GIL_DISABLED
#endif

void require(bool condition,const char* message) {
  if (!condition) throw std::runtime_error(message);
}

int main() {
  CodeExtra serial{};
  for (uint64_t index=1;index<=10000;++index) {
    Ci_code_extra_incr_calls(&serial);
    require(Ci_code_extra_get_calls(&serial)==index,"sequential exact count/threshold changed");
  }
  CodeExtra concurrent{};
  constexpr unsigned writers=4,readers=4,iterations=200000;
  std::atomic<unsigned> finished{0},failures{0},threshold_seen{0};
  std::atomic<uint64_t> observations{0};
  std::vector<std::thread> worker_threads;
  for (unsigned index=0;index<readers;++index) {
    worker_threads.emplace_back([&] {
      uint64_t previous=0;
      do {
        const uint64_t now=Ci_code_extra_get_calls(&concurrent);
        if (now<previous || now>writers*iterations) ++failures;
        if (now>=1000) threshold_seen.fetch_or(1,std::memory_order_relaxed);
        previous=now;++observations;
      } while(finished.load(std::memory_order_acquire)!=writers);
      if (Ci_code_extra_get_calls(&concurrent)!=writers*iterations) ++failures;
    });
  }
  for (unsigned index=0;index<writers;++index) {
    worker_threads.emplace_back([&] {
      for (unsigned call=0;call<iterations;++call) Ci_code_extra_incr_calls(&concurrent);
      finished.fetch_add(1,std::memory_order_acq_rel);
    });
  }
  for(auto& worker:worker_threads) worker.join();
  require(!failures.load(),"nonmonotonic/count/threshold observation");
  require(threshold_seen.load()==1 && observations.load()>0,"threshold or observations missing");
  require(Ci_code_extra_get_calls(&concurrent)==writers*iterations,"joined final count must be exact");
  CodeExtra rollover{};
  __atomic_store_n(&rollover.calls,std::numeric_limits<uint64_t>::max()-1,__ATOMIC_RELAXED);
  Ci_code_extra_incr_calls(&rollover);
  require(Ci_code_extra_get_calls(&rollover)==std::numeric_limits<uint64_t>::max(),"uint64 max changed");
  Ci_code_extra_incr_calls(&rollover);
  require(Ci_code_extra_get_calls(&rollover)==0,"uint64 modulo rollover changed");
  std::cout<<"PASS actual FT header: 4 writers exact800000, 4 monotonic readers, thresholds and uint64 rollover; observations="
    <<observations.load()<<" layout="<<sizeof(CodeExtra)<<" offset="<<offsetof(CodeExtra,calls)<<'\n';
}
