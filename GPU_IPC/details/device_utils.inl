__device__ __forceinline__ void SYNC_THREADS()
{
    __syncthreads();
}

__device__ __forceinline__ void THREAD_FENCE()
{
    __threadfence();
}

__device__ __forceinline__ unsigned int WARP_MASK_FROM_COUNT(unsigned int count)
{
    return count >= warpSize ? 0xffffffffU
                             : (count == 0 ? 0U : ((1U << count) - 1U));
}

__device__ __forceinline__ unsigned int WARP_BALLOT(int predicate,
                                                     unsigned int member_mask)
{
    return __ballot_sync(member_mask, predicate);
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


template <class TypeA, class TypeB>
__device__ __forceinline__ TypeA ATOMIC_ADD(TypeA* dest, const TypeB& source)
{
    return atomicAdd(dest, source);
}
