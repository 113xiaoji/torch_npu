#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <istream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace hdb_opapi {
// Exact NN-provider EmbeddingBag/CANN9.2 lifecycle evidence only; no bypass.
// See sdk-nn-gate-20261026-10 and the independent NN pool/cache/alive receipts.
// The normalized gate binds this candidate, framework resolver and NN/base DSOs.
// Typed production, candidate-device and performance gates remain separate.
inline constexpr bool kOwnedSdkLifecycleValidated = true;
// Cold-only binary attestation. No external crypto library or interpreter call
// is introduced. The hot capability query caches the verified profile result.
inline std::string Sha256(std::istream& source)
{
    constexpr uint32_t k[] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    std::array<uint32_t,8> h{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    auto rotate=[](uint32_t x,unsigned n){return (x>>n)|(x<<(32-n));};
    auto compress=[&](const std::array<unsigned char,64>& block){
        uint32_t w[64];
        for(size_t i=0;i<16;++i)w[i]=(uint32_t(block[4*i])<<24)|(uint32_t(block[4*i+1])<<16)|
            (uint32_t(block[4*i+2])<<8)|uint32_t(block[4*i+3]);
        for(size_t i=16;i<64;++i){
            const auto s0=rotate(w[i-15],7)^rotate(w[i-15],18)^(w[i-15]>>3);
            const auto s1=rotate(w[i-2],17)^rotate(w[i-2],19)^(w[i-2]>>10);
            w[i]=w[i-16]+s0+w[i-7]+s1;
        }
        auto a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],q=h[7];
        for(size_t i=0;i<64;++i){
            const auto s1=rotate(e,6)^rotate(e,11)^rotate(e,25);
            const auto choice=(e&f)^((~e)&g);
            const auto t1=q+s1+choice+k[i]+w[i];
            const auto s0=rotate(a,2)^rotate(a,13)^rotate(a,22);
            const auto majority=(a&b)^(a&c)^(b&c);
            q=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+s0+majority;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=q;
    };
    std::array<unsigned char,64> block{};uint64_t bytes=0;
    for(;;){
        source.read(reinterpret_cast<char*>(block.data()),block.size());
        const auto count=static_cast<size_t>(source.gcount());
        if(source.bad() || (source.fail() && !source.eof()))throw std::runtime_error("SDK binary read failed");
        if(count>std::numeric_limits<uint64_t>::max()/8-bytes)throw std::overflow_error("SDK binary length overflow");
        bytes+=count;
        if(count==block.size()){compress(block);continue;}
        block[count]=0x80;std::fill(block.begin()+count+1,block.end(),static_cast<unsigned char>(0));
        if(count>=56){compress(block);block.fill(0);}
        const uint64_t bits=bytes*8;
        for(size_t i=0;i<8;++i)block[63-i]=static_cast<unsigned char>(bits>>(8*i));
        compress(block);break;
    }
    std::ostringstream result;result<<std::hex<<std::setfill('0');
    for(auto value:h)result<<std::setw(8)<<value;
    return result.str();
}
inline constexpr const char* kOpBase920Sha="72cd43b4af1c9c451b566ccb7b32071793402949578f838d8345e369b056477c";
inline constexpr const char* kOpApi920Sha="8107f8d0418490d438ee50037bc91dbf5a8658aca65508516d4b6f7ab12baf68";
} // namespace hdb_opapi
