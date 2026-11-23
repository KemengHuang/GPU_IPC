__device__ __forceinline__ void SYNC_THREADS()
{
    __syncthreads();
}

__device__ __forceinline__ void THREAD_FENCE()
{
    __threadfence();
}

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
// A CUDA warp is always 32 lanes, so a *_sync form acts on at most 32 lanes and its
// ballot is a 32-bit mask. An AMD wavefront is 32 lanes on RDNA and 64 on CDNA, and the
// unqualified HIP intrinsics take the wavefront width. The faithful translation
// therefore pins the shuffle width to 32 and shifts a wavefront-wide ballot down to the
// calling lane's own 32-lane half: that is the identity on a 32-lane wavefront and
// selects the correct half on a 64-lane one. Truncating the ballot instead would hand
// the upper half of a 64-lane wavefront the lower half's bits. The member masks are
// 32-bit lane masks of a 32-lane group; the callers already discard values read from
// lanes outside them, so they are not needed here.
__device__ __forceinline__ unsigned int WARP_MASK_FROM_COUNT(unsigned int count)
{
    return count >= 32 ? 0xffffffffU : (count == 0 ? 0U : ((1U << count) - 1U));
}

__device__ __forceinline__ void SYNC_WARP(unsigned int member_mask)
{
    (void)member_mask;
    __syncwarp();
}

__device__ __forceinline__ unsigned int WARP_BALLOT(int predicate,
                                                     unsigned int member_mask)
{
    (void)member_mask;
    unsigned long long mask = __ballot(predicate);
    return (unsigned int)(mask >> (__lane_id() & ~31u));
}

template <class Type>
__device__ __forceinline__ unsigned int WARP_MATCH_ANY(Type value,
                                                        unsigned int member_mask)
{
    (void)member_mask;
    unsigned long long mask = __match_any(value);
    return (unsigned int)(mask >> (__lane_id() & ~31u));
}

template <class Type>
__device__ __forceinline__ Type WARP_SHFL(Type var,
                                          int srcLane,
                                          unsigned int member_mask)
{
    (void)member_mask;
    return __shfl(var, srcLane, 32);
}

template <class Type>
__device__ __forceinline__ Type WARP_SHFL_DOWN(Type var,
                                               unsigned int delta,
                                               unsigned int member_mask)
{
    (void)member_mask;
    return __shfl_down(var, delta, 32);
}
#else
__device__ __forceinline__ unsigned int WARP_MASK_FROM_COUNT(unsigned int count)
{
    return count >= warpSize ? 0xffffffffU
                             : (count == 0 ? 0U : ((1U << count) - 1U));
}

__device__ __forceinline__ void SYNC_WARP(unsigned int member_mask)
{
    __syncwarp(member_mask);
}

__device__ __forceinline__ unsigned int WARP_BALLOT(int predicate,
                                                     unsigned int member_mask)
{
    return __ballot_sync(member_mask, predicate);
}

template <class Type>
__device__ __forceinline__ unsigned int WARP_MATCH_ANY(Type value,
                                                        unsigned int member_mask)
{
    return __match_any_sync(member_mask, value);
}

template <class Type>
__device__ __forceinline__ Type WARP_SHFL(Type var,
                                          int srcLane,
                                          unsigned int member_mask)
{
    return __shfl_sync(member_mask, var, srcLane);
}

template <class Type>
__device__ __forceinline__ Type WARP_SHFL_DOWN(Type var,
                                               unsigned int delta,
                                               unsigned int member_mask)
{
    return __shfl_down_sync(member_mask, var, delta);
}
#endif


template <class TypeA, class TypeB>
__device__ __forceinline__ TypeA ATOMIC_ADD(TypeA* dest, const TypeB& source)
{
    return atomicAdd(dest, source);
}
