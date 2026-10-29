#pragma once
#include <tuple>
#include <utility>

// Off calls the complete original macro unchanged. Typed on-calls evaluate
// arguments once, and all temporaries live through the std::apply full
// expression, including the synchronous construction of owned snapshots or
// the original macro's own argument copies on an explicitly fenced fallback.
// No exception is caught here: an SDK/allocator failure cannot rerun an op.
#define HDB_EAGER_PREPARE_R24(aclnn_api, original_macro, v1, ...) \
    do { \
        if (!c10_npu::eager_prepare::Enabled()) { original_macro(aclnn_api,__VA_ARGS__); } \
        else if constexpr (hdb_opapi::HasEagerTypedSchema(#aclnn_api)) { \
            std::apply([&](auto&&... hdb_saved_args) { \
                const auto hdb_arguments=std::forward_as_tuple(hdb_saved_args...); \
                const bool hdb_qualified=hdb_opapi::EagerSdkQualified(#aclnn_api); \
                const bool hdb_domain=hdb_qualified && hdb_opapi::EagerArgumentDomain(#aclnn_api,hdb_arguments); \
                if (c10_npu::eager_prepare::Admit(#aclnn_api,hdb_qualified,"argument_contract_unadmitted",hdb_domain)) { \
                    hdb_opapi::EnqueueEagerArguments(#aclnn_api,v1,std::forward<decltype(hdb_saved_args)>(hdb_saved_args)...); \
                } else { original_macro(aclnn_api,hdb_saved_args...); } \
            },std::forward_as_tuple(__VA_ARGS__)); \
        } else { \
            (void)c10_npu::eager_prepare::Admit(#aclnn_api,false,"operator_unregistered",false); \
            original_macro(aclnn_api,__VA_ARGS__); \
        } \
    } while(false)
