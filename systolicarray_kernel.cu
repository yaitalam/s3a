

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdio.h>

template <typename scalar_t>
__device__ scalar_t setBit(scalar_t n, int bit_pos, int stuck_bit_value)
{
    uint16_t *val = reinterpret_cast<uint16_t *>(&n); // casting bits from float to int so that c/cuda can do bitwise oparations
    if (stuck_bit_value == 1)
    {
        *val |= (1 << bit_pos); // changes oposite bit to wanted bit leaves wanted bit
    }
    else
    {
        *val &= ~(1 << bit_pos); // changes oposite bit to wanted bit leaves wanted bit
    }
    return n;
}

__device__ double setBit(double n, int bit_pos, int stuck_bit_value)
{
    uint64_t *val = reinterpret_cast<uint64_t *>(&n);
    if (stuck_bit_value == 1)
    {
        *val |= (1 << bit_pos);
    }
    else
    {
        *val &= ~(1 << bit_pos);
    }
    return n;
}

__device__ float setBit(float n, int bit_pos, int stuck_bit_value)
{
    uint32_t *val = reinterpret_cast<uint32_t *>(&n); // casting bits from float to int so that c/cuda can do bitwise oparations
    if (stuck_bit_value == 1)
    {
        *val |= (1 << bit_pos); // changes oposite bit to wanted bit leaves wanted bit
    }
    else
    {
        *val &= ~(1 << bit_pos); // changes oposite bit to wanted bit leaves wanted bit
    }
    return n;
}

__device__ half setBit(half n, int bit_pos, int stuck_bit_value)
{
    uint16_t *val = reinterpret_cast<uint16_t *>(&n); // casting bits from float to int so that c/cuda can do bitwise oparations
    if (stuck_bit_value == 1)
    {
        *val |= (1 << bit_pos); // changes oposite bit to wanted bit leaves wanted bit
    }
    else
    {
        *val &= ~(1 << bit_pos); // changes oposite bit to wanted bit leaves wanted bit
    }
    return n;
}

template <typename scalar_t>
__global__ void systolicMatMulKernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> t1,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> t2,
    torch::PackedTensorAccessor<scalar_t, 5, torch::RestrictPtrTraits, size_t> res,
    int n, int m, int o, int d, const int fault)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < o)
    {
        int i_d = i / d;
        int j_d = j / d;
        int i_mod_d = i % d;
        int j_mod_d = j % d;

        for (int k = 0; k < m; ++k)
        {
            int k_d = k / d;
            auto input = t1[i][k];
            auto weight = t2[k][j];
            auto partial_sum = res[i_d][j_d][k_d][i_mod_d][j_mod_d];

            if (fault != 0)
            {

                // reads int valuse of bits to generate the fault
                int faut_type = (fault & (((1 << (30 - 28 + 1)) - 1) << 28)) >> 28;
                int x_pos = (fault & (((1 << (27 - 18 + 1)) - 1) << 18)) >> 18;
                int y_pos = (fault & (((1 << (17 - 8 + 1)) - 1) << 8)) >> 8;
                int stuck_bit_val = (fault & (((1 << (7 - 7 + 1)) - 1) << 7)) >> 7;
                int bit_pos = (fault & (((1 << (6 - 0 + 1)) - 1) << 0)) >> 0;

                // right link fault
                if (faut_type == 1)
                {
                    if (j_mod_d > y_pos && k % d == x_pos)
                    {
                        input = setBit(input, bit_pos, stuck_bit_val);
                    }
                }
                // DOWN LINKS
                else if (faut_type == 2)
                {
                    if (k % d == x_pos + 1 && j_mod_d == y_pos)
                    {
                        partial_sum = setBit(partial_sum, bit_pos, stuck_bit_val);
                    }
                }
                // WEIGHT REG
                else if (faut_type == 3)
                {
                    if (k % d == x_pos && j_mod_d == y_pos)
                    {
                        weight = setBit(weight, bit_pos, stuck_bit_val);
                    }
                }
            }

            res[i_d][j_d][k_d][i_mod_d][j_mod_d] = input * weight + partial_sum;

            // this for down link faults on the last row
            if (fault != 0)
            { // repace with fault check

                // reads int valuse of bits between 30 and 28 inclusive
                int faut_type = (fault & (((1 << (30 - 28 + 1)) - 1) << 28)) >> 28;
                //...you get the idea
                int x_pos = (fault & (((1 << (27 - 18 + 1)) - 1) << 18)) >> 18;
                int y_pos = (fault & (((1 << (17 - 8 + 1)) - 1) << 8)) >> 8;
                int stuck_bit_val = (fault & (((1 << (7 - 7 + 1)) - 1) << 7)) >> 7;
                int bit_pos = (fault & (((1 << (6 - 0 + 1)) - 1) << 0)) >> 0;

                if (faut_type == 2 && x_pos == d - 1)
                {
                    if (k % d == x_pos && j_mod_d == y_pos)
                    {
                        res[i_d][j_d][k_d][i_mod_d][j_mod_d] = setBit(res[i_d][j_d][k_d][i_mod_d][j_mod_d], bit_pos, stuck_bit_val);
                    }
                }
            }
        }
    }
}

torch::Tensor systolic_fw_cu(torch::Tensor t1, torch::Tensor t2, int d, const int fault_type)
{
    // hold the dimentions of the inputs
    const auto n = t1.size(0);
    const auto m = t1.size(1);
    const auto o = t2.size(1);

    // calcualte the dimentions of the 5d result matrix
    const auto res_n = (n + d - 1) / d;
    const auto res_m = (m + d - 1) / d;
    const auto res_o = (o + d - 1) / d;
    torch::Tensor res = torch::zeros({res_n, res_o, res_m, d, d}, t1.options()); //.options() copies over datatype and device of the tensor

    // reserving blocks and threads on the gpu
    const dim3 threadsPerBlock(16, 16);
    const dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                         (o + threadsPerBlock.y - 1) / threadsPerBlock.y);

    if (at::ScalarType::BFloat16 != t1.scalar_type())
    {

        // this is the call to pythorch that allows us to make the GPU call using out tensors
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(t1.scalar_type(), "systolic_fw_cu", ([&]
                                                                                 {

            // this the the call to the acultal __global__ GPU function scalar_t allows it to dynamicaly change between differnet floating types like float double and half
            systolicMatMulKernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
                // cuda cant handle tensors but it can handle packed acessors this method takes <datatype , dimention of tensor , **NOCLUE** , then the size>
                t1.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(), // 2D tensor
                t2.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(), // 2D tensor
                res.packed_accessor<scalar_t, 5, torch::RestrictPtrTraits, size_t>(), // 5D tensor
                //these are used to calculate what index of partal sums being accumulated the input*weight should go to
                n,
                m,
                o,
                d,
                //passing in the fault information
                fault_type
            ); }));
    }
    else
    {
        AT_DISPATCH_REDUCED_FLOATING_TYPES(t1.scalar_type(), "systolic_fw_cu", ([&]
                                                                                     {

            // this the the call to the acultal __global__ GPU function scalar_t allows it to dynamicaly change between differnet floating types like float double and half
            systolicMatMulKernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
                // cuda cant handle tensors but it can handle packed acessors this method takes <datatype , dimention of tensor , **NOCLUE** , then the size>
                t1.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(), // 2D tensor
                t2.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(), // 2D tensor
                res.packed_accessor<scalar_t, 5, torch::RestrictPtrTraits, size_t>(), // 5D tensor
                //these are used to calculate what index of partal sums being accumulated the input*weight should go to
                n,
                m,
                o,
                d,
                //passing in the fault information
                fault_type
            ); }));
    }
    return res;
}