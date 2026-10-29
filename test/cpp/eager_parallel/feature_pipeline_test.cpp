#include "torch_npu/csrc/core/npu/eager_parallel/DependencyScheduler.h"

#include <cmath>
#include <deque>
#include <iostream>
#include <random>

using namespace c10_npu::eager_parallel;

#define CHECK(expr) do { if (!(expr)) throw std::runtime_error( \
    std::string(__func__) + ": " #expr); } while (false)

// A deterministic device-stream emulator. Kernels run only when the chosen
// stream is stepped, and stream waits really block consumption of their events.
// Scheduler, effects, stream choice, event fan-in and retirement are production core.
class EmulatedStreams final : public EventBackend {
public:
    enum class Kind { Kernel, Wait, Record };
    struct Command { Kind kind; std::function<void()> kernel; EventToken event; };
    struct Event { bool published = false, complete = false, retired = false; };
    std::map<EventToken, Event> events;
    std::array<std::deque<Command>, 3> streams;
    std::size_t waits = 0, retired = 0, kernels = 0;

    void reserve_event(EventToken event, StreamId) override
    { CHECK(events.emplace(event, Event{}).second); }
    void request_record(EventToken event, StreamId stream) override
    {
        queue(stream).push_back({Kind::Record, {}, event});
        events.at(event).published = true; // record is now in the device stream
    }
    Publication publication(EventToken event) override
    { return events.at(event).published ? Publication::Published : Publication::Pending; }
    void enqueue_wait(StreamId stream, EventToken event) override
    {
        CHECK(events.at(event).published);
        CHECK(!events.at(event).retired);
        queue(stream).push_back({Kind::Wait, {}, event});
        ++waits;
    }
    Completion device_complete(EventToken event) override
    { return events.at(event).complete ? Completion::Complete : Completion::Pending; }
    void retire_event(EventToken event) override
    {
        CHECK(events.at(event).complete);
        CHECK(!events.at(event).retired);
        events.at(event).retired = true;
        ++retired;
    }
    void kernel(StreamId stream, std::function<void()> body)
    { queue(stream).push_back({Kind::Kernel, std::move(body), {}}); }
    bool step(StreamId stream)
    {
        auto& commands = queue(stream);
        if (commands.empty()) return false;
        auto& first = commands.front();
        if (first.kind == Kind::Wait) {
            CHECK(!events.at(first.event).retired);
            if (!events.at(first.event).complete) return false;
        } else if (first.kind == Kind::Record) {
            CHECK(!events.at(first.event).retired);
            events.at(first.event).complete = true;
        } else {
            first.kernel();
            ++kernels;
        }
        commands.pop_front();
        return true;
    }
    bool empty() const
    {
        for (const auto& stream : streams) if (!stream.empty()) return false;
        return true;
    }
private:
    std::deque<Command>& queue(StreamId stream)
    { return streams.at(static_cast<std::size_t>(stream)); }
};

template<class T> struct Buffer {
    std::shared_ptr<std::vector<T>> values;
    AllocationLease lease;
};

template<class T> Buffer<T> buffer(std::uint64_t identity, std::size_t count, bool initialized)
{
    auto values = std::make_shared<std::vector<T>>(count);
    return {values, {{identity, 1}, count * sizeof(T), values,
        initialized ? EntryReadiness::DeviceComplete : EntryReadiness::NewAllocation}};
}

void check_pipeline(unsigned seed)
{
    constexpr std::size_t tables = 26, batch = 3, dim = 4, rows = 7;
    constexpr std::size_t features = tables + 1, pairs = features * (features - 1) / 2;
    auto backend = std::make_shared<EmulatedStreams>();
    DependencyScheduler scheduler(backend);
    std::uint64_t allocation_id = 1, sequence = 0;
    auto dense = buffer<double>(allocation_id++, batch * dim, true);
    for (std::size_t i = 0; i < dense.values->size(); ++i)
        (*dense.values)[i] = static_cast<double>(i + 1) / 8.0;
    std::vector<Buffer<double>> weights, embeddings;
    std::vector<Buffer<std::size_t>> indices, offsets;
    std::vector<std::size_t> completion_order;
    std::array<bool, tables> producer_ready{};
    bool cat_ran = false, gram_ran = false, index_ran = false, final_ran = false;

    auto enqueue = [&](NodeKind kind, std::vector<Access> accesses, std::function<void()> body) {
        FrozenCall frozen;
        frozen.logical_sequence = ++sequence;
        frozen.kind = kind;
        frozen.accesses = std::move(accesses);
        auto id = scheduler.admit(std::move(frozen));
        auto result = scheduler.submit(id, [&](StreamId stream) {
            backend->kernel(stream, std::move(body));
            return LaunchResult::Submitted;
        });
        CHECK(result == SubmitResult::Submitted);
        return id;
    };

    for (std::size_t table = 0; table < tables; ++table) {
        weights.push_back(buffer<double>(allocation_id++, rows * dim, true));
        indices.push_back(buffer<std::size_t>(allocation_id++, 5, true));
        offsets.push_back(buffer<std::size_t>(allocation_id++, batch + 1, true));
        embeddings.push_back(buffer<double>(allocation_id++, batch * dim, false));
        auto weight = weights.back(), output = embeddings.back();
        auto index = indices.back(), offset = offsets.back();
        for (std::size_t i = 0; i < weight.values->size(); ++i)
            (*weight.values)[i] = static_cast<double>((table + 3) * (i + 1)) / 16.0;
        *index.values = {table % rows, (table + 2) % rows, table % rows,
            (table + 5) % rows, (table + 1) % rows};
        *offset.values = {0, 2, 2, 5}; // repeated index and an empty bag
        enqueue(NodeKind::Producer, {{weight.lease, AccessMode::Read}, {index.lease, AccessMode::Read},
            {offset.lease, AccessMode::Read}, {output.lease, AccessMode::Write}},
            [&, table, weight, index, offset, output] {
                for (std::size_t bag = 0; bag < batch; ++bag)
                    for (std::size_t p = (*offset.values)[bag]; p < (*offset.values)[bag + 1]; ++p)
                        for (std::size_t d = 0; d < dim; ++d)
                            (*output.values)[bag * dim + d] += (*weight.values)[(*index.values)[p] * dim + d];
                producer_ready[table] = true;
                completion_order.push_back(table);
            });
    }

    auto packed = buffer<double>(allocation_id++, batch * features * dim, false);
    std::vector<Access> cat_accesses{{dense.lease, AccessMode::Read}, {packed.lease, AccessMode::Write}};
    for (const auto& embedding : embeddings) cat_accesses.push_back({embedding.lease, AccessMode::Read});
    enqueue(NodeKind::Consumer, std::move(cat_accesses), [&, packed] {
        for (bool ready : producer_ready) CHECK(ready);
        for (std::size_t b = 0; b < batch; ++b)
            for (std::size_t f = 0; f < features; ++f)
                for (std::size_t d = 0; d < dim; ++d)
                    (*packed.values)[(b * features + f) * dim + d] =
                        f == 0 ? (*dense.values)[b * dim + d] : (*embeddings[f - 1].values)[b * dim + d];
        cat_ran = true;
    });
    CHECK(backend->waits == 2); // 26 tensors reduce to the two side-stream tails

    auto gram = buffer<double>(allocation_id++, batch * features * features, false);
    // T and transpose(T) share the same allocation and have two read effects.
    enqueue(NodeKind::Consumer, {{packed.lease, AccessMode::Read}, {packed.lease, AccessMode::Read},
        {gram.lease, AccessMode::Write}}, [&, packed, gram] {
            CHECK(cat_ran);
            for (std::size_t b = 0; b < batch; ++b)
                for (std::size_t i = 0; i < features; ++i)
                    for (std::size_t j = 0; j < features; ++j)
                        for (std::size_t d = 0; d < dim; ++d)
                            (*gram.values)[(b * features + i) * features + j] +=
                                (*packed.values)[(b * features + i) * dim + d] *
                                (*packed.values)[(b * features + j) * dim + d];
            gram_ran = true;
        });

    auto li = buffer<std::size_t>(allocation_id++, pairs, true);
    auto lj = buffer<std::size_t>(allocation_id++, pairs, true);
    std::size_t pair = 0;
    for (std::size_t i = 0; i < features; ++i)
        for (std::size_t j = 0; j < i; ++j) { (*li.values)[pair] = i; (*lj.values)[pair++] = j; }
    auto flattened = buffer<double>(allocation_id++, batch * pairs, false);
    enqueue(NodeKind::Consumer, {{gram.lease, AccessMode::Read}, {li.lease, AccessMode::Read},
        {lj.lease, AccessMode::Read}, {flattened.lease, AccessMode::Write}}, [&, gram, li, lj, flattened] {
            CHECK(gram_ran);
            for (std::size_t b = 0; b < batch; ++b)
                for (std::size_t p = 0; p < pairs; ++p)
                    (*flattened.values)[b * pairs + p] =
                        (*gram.values)[(b * features + (*li.values)[p]) * features + (*lj.values)[p]];
            index_ran = true;
        });
    auto result = buffer<double>(allocation_id++, batch * (dim + pairs), false);
    enqueue(NodeKind::Consumer, {{dense.lease, AccessMode::Read}, {flattened.lease, AccessMode::Read},
        {result.lease, AccessMode::Write}}, [&, flattened, result] {
            CHECK(index_ran);
            for (std::size_t b = 0; b < batch; ++b) {
                for (std::size_t d = 0; d < dim; ++d)
                    (*result.values)[b * (dim + pairs) + d] = (*dense.values)[b * dim + d];
                for (std::size_t p = 0; p < pairs; ++p)
                    (*result.values)[b * (dim + pairs) + dim + p] = (*flattened.values)[b * pairs + p];
            }
            final_ran = true;
        });

    CHECK(!cat_ran && !final_ran);
    CHECK(scheduler.snapshot().retained_bytes > 0);
    CHECK(scheduler.close() == CloseResult::Pending);
    // Deliberately complete table 1 before table 0; stream-local FIFO remains valid.
    CHECK(backend->step(2));
    CHECK(completion_order.front() == 1);
    std::mt19937 random(seed);
    for (std::size_t step = 0; !backend->empty(); ++step) {
        CHECK(step < 100000); // deadlock watchdog is a test failure, not normal closure
        backend->step(static_cast<StreamId>(random() % 3));
        scheduler.poll();
    }
    CHECK(final_ran && completion_order.size() == tables);
    CHECK(backend->kernels == tables + 4);
    CHECK(backend->waits == 2);
    CHECK(scheduler.close() == CloseResult::Closed);
    CHECK(scheduler.snapshot().active_nodes == 0 && scheduler.snapshot().retained_bytes == 0);
    CHECK(backend->retired == tables + 4);

    // Independent scalar oracle never reads packed T, Gram, flattened values or
    // embedding outputs: derive each feature directly from original input tables.
    auto feature = [&](std::size_t b, std::size_t f, std::size_t d) {
        if (f == 0) return static_cast<long double>((*dense.values)[b * dim + d]);
        const auto t = f - 1;
        long double sum = 0;
        for (std::size_t p = (*offsets[t].values)[b]; p < (*offsets[t].values)[b + 1]; ++p)
            sum += static_cast<long double>((*weights[t].values)[(*indices[t].values)[p] * dim + d]);
        return sum;
    };
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t d = 0; d < dim; ++d)
            CHECK((*result.values)[b * (dim + pairs) + d] == (*dense.values)[b * dim + d]);
        std::size_t p = 0;
        for (std::size_t i = 0; i < features; ++i)
            for (std::size_t j = 0; j < i; ++j) {
                long double oracle = 0;
                for (std::size_t d = 0; d < dim; ++d) oracle += feature(b, i, d) * feature(b, j, d);
                const auto actual = static_cast<long double>((*result.values)[b * (dim + pairs) + dim + p++]);
                CHECK(std::fabs(actual - oracle) <= 1e-12L * (1 + std::fabs(oracle)));
            }
    }
}

int main()
{
    try {
        for (unsigned seed : {7u, 53u, 771u, 2026u}) check_pipeline(seed);
        std::cout << "PASS R26 protocol-driven feature pipeline: 26 producers, 4 seeds, "
                     "two-stream fan-in, Gram, ordered index and final cat scalar oracle\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
