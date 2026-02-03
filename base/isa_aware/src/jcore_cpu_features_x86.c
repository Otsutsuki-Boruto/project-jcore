/* CPU feature detection (x86/x64) with proper xgetbv/XCR0 checks */

#include "jcore_isa_dispatch.h"
#include <stdint.h>

#if defined(__x86_64__) || defined(__i386__)

static void jcore_cpuid(uint32_t eax_in, uint32_t ecx_in, uint32_t regs[4])
{
#if defined(__GNUC__) || defined(__clang__)
  uint32_t a = eax_in, c = ecx_in;
  uint32_t b, d;
  __asm__ volatile("cpuid"
                   : "=a"(a), "=b"(b), "=c"(c), "=d"(d)
                   : "0"(eax_in), "2"(ecx_in));
  regs[0] = a;
  regs[1] = b;
  regs[2] = c;
  regs[3] = d;
#else
  regs[0] = regs[1] = regs[2] = regs[3] = 0;
#endif
}

/* safe xgetbv wrapper; caller MUST ensure OSXSAVE is supported/enabled */
static uint64_t jcore_xgetbv_u64(void)
{
  uint32_t eax = 0, edx = 0;
#if defined(__GNUC__) || defined(__clang__)
  __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
#endif
  return ((uint64_t)edx << 32) | eax;
}

jcore_features_t jcore_detect_cpu_features_x86(void)
{
  jcore_features_t feats = JCORE_FEAT_NONE;
  uint32_t regs[4];

  /* Basic feature leaf */
  jcore_cpuid(1, 0, regs);
  uint32_t ecx1 = regs[2];
  uint32_t edx1 = regs[3];

  /* SSE2 (EDX bit 26) */
  if (edx1 & (1u << 26))
    feats |= JCORE_FEAT_SSE2;

  /* OSXSAVE + AVX (ECX bits 27=OSXSAVE, 28=AVX) */
  int osxsave = (ecx1 & (1u << 27)) != 0;
  int avx_hw = (ecx1 & (1u << 28)) != 0;

  uint64_t xcr0 = 0;
  if (osxsave)
  {
    /* safe to call xgetbv because OSXSAVE means OS enabled XGETBV usage */
    xcr0 = jcore_xgetbv_u64();
  }

  /* If CPU advertises AVX and OS saved YMM state (XCR0[2:1] == 11b) */
  if (avx_hw && osxsave && ((xcr0 & ((1ULL << 1) | (1ULL << 2))) == ((1ULL << 1) | (1ULL << 2))))
  {
    feats |= JCORE_FEAT_AVX;
  }

  /* Extended feature leaf for AVX2 / AVX512 / AMX */
  jcore_cpuid(7, 0, regs);
  uint32_t ebx7 = regs[1];
  /* AVX2 (EBX bit 5) */
  if (ebx7 & (1u << 5))
  {
    /* AVX2 requires AVX OS support as well (YMM state) */
    if (feats & JCORE_FEAT_AVX)
      feats |= JCORE_FEAT_AVX2;
  }

  /* AVX-512 foundation (EBX bit 16) - requires specific XCR0 bits set */
  if (ebx7 & (1u << 16))
  {
    /* AVX-512 needs XCR0 bits: 1 (XMM), 2 (YMM) and 5..7 (OPMASK/ZMM hi parts) */
    const uint64_t avx512_mask = (1ULL << 1) | (1ULL << 2) | (1ULL << 5) | (1ULL << 6) | (1ULL << 7);
    if (osxsave && ((xcr0 & avx512_mask) == avx512_mask))
    {
      feats |= JCORE_FEAT_AVX512;
    }
  }

  /* AMX-TILE (EBX bit 24) - OS must have XCR0 bits for tilecfg/tiledata set (bits 17/18) */
  if (ebx7 & (1u << 24))
  {
    const uint64_t amx_mask = (1ULL << 17) | (1ULL << 18); /* tilecfg & tiledata */
    if (osxsave && ((xcr0 & amx_mask) == amx_mask))
    {
      feats |= JCORE_FEAT_AMX;
    }
  }

  return feats;
}

#else

/* Non-x86 platforms: return none (extend later for ARM, etc). */
jcore_features_t jcore_detect_cpu_features_x86(void)
{
  (void)0;
  return JCORE_FEAT_NONE;
}

#endif
