#include "utils.h"

torch::Tensor systolicMatMul_fw(torch::Tensor t1, torch::Tensor t2, const int d , const int fault_type) {

    int n_1 = t1.size(0);
    int m_1 = t1.size(1);
    int o_1 = t2.size(1);

    int m_tmep = 0;
    int o_temp = 0;
    int n_temp = 0;

    if (m_1 % d != 0) {
        m_tmep = d - (m_1 % d);
    }
    if (o_1 % d != 0) {
        o_temp = d - (o_1 % d);
    }
    if (n_1 % d != 0) {
        n_temp = d - (n_1 % d);
    }

    t1 = torch::constant_pad_nd(t1, {0, m_tmep, 0, n_temp}, 0);
    t2 = torch::constant_pad_nd(t2, {0, o_temp, 0, m_tmep}, 0);

    int n = t1.size(0);
    int m = t1.size(1);
    int o = t2.size(1);

    t1 = t1.contiguous();
    t2 = t2.contiguous();


    //matrix multiplication
    CHECK_INPUT(t1);
    CHECK_INPUT(t2);
    //add call to new systolic array forward function
    torch::Tensor res = systolic_fw_cu(t1, t2, d , fault_type);
    
    // reformating
    torch::cuda::synchronize();
    res = res.sum(2).transpose(1, 2).reshape({n, o});
    res = res.slice(0, 0, n_1).slice(1, 0, o_1);

    return res;
}

int generate_fault_config(int fault_type, int x_pos, int y_pos, int stuck_bit_val, int bit_pos) {
    int fault_config = 0;
    fault_config |= ((1 & 0x1) << 31);
    fault_config |= ((fault_type & 0x7) << 28); // 3 bits for fault_type
    fault_config |= ((x_pos & 0x3FF) << 18); // 10 bits for x_pos (mask with 0x3FF to ensure 10 bits)
    fault_config |= ((y_pos & 0x3FF) << 8);  // 10 bits for y_pos (mask with 0x3FF to ensure 10 bits)
    fault_config |= ((stuck_bit_val & 0x1) << 7); // 1 bit for stuck_bit_val (mask with 0x1 to ensure 1 bit)
    fault_config |= (bit_pos & 0x7F);
    return fault_config;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("systolic_matmul_fw", &systolicMatMul_fw, "Systolic Matrix Multiplication (CUDA)");
    m.def("generate_fault_config", &generate_fault_config, "generate fault config");
}
