from typing import Any
import torch
from torch.autograd import Function
import torch_systolic_array

class Systolic_Array_CUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx,t1, t2, d):
            
        fault_config = torch_systolic_array.generate_fault_config(3,0,0,1,15)
        print(fault_config)
        ctx.save_for_backward(t1, t2)
        ctx.d = d
        ctx.fault_config =fault_config
        
        
        res = torch_systolic_array.systolic_matmul_fw(t1, t2, d,fault_config)
        
        return res

    @staticmethod
    def backward(ctx, grad_output):
        t1, t2 = ctx.saved_tensors
        
        grad_t1 = torch.matmul(grad_output, t2.t())
        grad_t2 = torch.matmul(t1.t(), grad_output)

        return grad_t1, grad_t2, None , None
    
