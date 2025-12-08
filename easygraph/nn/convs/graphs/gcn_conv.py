import torch
import torch.nn as nn
# class GCNConv(nn.Module):
#     def __init__(self, in_channels, out_channels, bias=True):
#         super(GCNConv, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)

#         self.reset_parameters()
        
#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.weight)
#         if self.bias is not None:
#             nn.init.zeros_(self.bias)

#     def forward(self, x, g):  

#         out = g.adj_t.matmul(x @ self.weight)

#         if self.bias is not None:
#             out += self.bias

#         return out

#     def __repr__(self):
#         return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
    

## The version of the GP 

# import torch
# import torch.nn as nn

# class GCNConv(nn.Module):
#     '''
#     GCN with graph partition version
#     '''
#     def __init__(self, in_channels, out_channels, bias=True):
#         super(GCNConv, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)

#         self.reset_parameters()
        
#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.weight)
#         if self.bias is not None:
#             nn.init.zeros_(self.bias)

#     def forward(self, x, g):  

#         out = g.cache['adj_gp'].matmul(x @ self.weight)

#         if self.bias is not None:
#             out += self.bias
#         return out

#     def __repr__(self):
#         return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)



## The version of the GP+backward 

# class FastGCNConvFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, weight, adj, bias=None):
#         AX = adj.matmul(x)
#         out = AX @ weight
#         # out = adj.matmul(x @ weight)
#         if bias is not None:
#             out += bias
#         # AX, x, weight, bias 是 Tensor，可以用 save_for_backward
#         ctx.save_for_backward(AX, weight, bias)
#         # adj 是稀疏矩阵，直接赋值给 ctx
#         ctx.adj = adj
#         return out

#     @staticmethod
#     def backward(ctx, grad_out):
#         AX, weight, bias = ctx.saved_tensors
#         adj = ctx.adj

#         grad_x = grad_w = grad_b = None
#         grad_w = AX.T @ grad_out
#         grad_x = adj.matmul(grad_out @ weight.T)
#         if bias is not None:
#             grad_b = grad_out.sum(0)
#         return grad_x, grad_w, None, grad_b
    
# class GCNConv(nn.Module):
#     def __init__(self, in_channels, out_channels, bias=True):
#         super().__init__()
#         self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
#         self.bias = nn.Parameter(torch.Tensor(out_channels)) if bias else None
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.weight)
#         if self.bias is not None:
#             nn.init.zeros_(self.bias)

#     def forward(self, x, g):
#         return FastGCNConvFn.apply(x, self.weight, g.cache['adj_gp'], self.bias)



### The version of C++ BK
# try:
#     import cpp_easygraph
#     HAS_CPP_BACKEND = True
# except ImportError:
#     print("Warning: cpp_easygraph module not found. Using slow Python fallback.")
#     HAS_CPP_BACKEND = False
    
# class GCNConv(nn.Module):
#     def __init__(self, in_channels, out_channels, bias=True):
#         super().__init__()
#         self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
#         self.bias = nn.Parameter(torch.Tensor(out_channels)) if bias else None
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.weight)
#         if self.bias is not None:
#             nn.init.zeros_(self.bias)

#     def forward(self, x, g):
#         return cpp_easygraph.upscale_gcn_forward(x, self.weight, g.cache['adj_torch'], self.bias)


###  The version of GP+BW upup
class FastGCNConvFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, adj, bias=None):
        AX = adj.matmul(x) 
        out = AX @ weight
        
        if bias is not None:
            out += bias
            
        ctx.save_for_backward(AX, weight, bias)
        ctx.adj = adj
        
        return out

    @staticmethod
    def backward(ctx, grad_out):
        AX, weight, bias = ctx.saved_tensors
        adj = ctx.adj

        grad_x = grad_w = grad_b = None
        
        if ctx.needs_input_grad[1]:
            grad_w = AX.t() @ grad_out

        if ctx.needs_input_grad[0]:
            grad_temp = grad_out @ weight.t()
            grad_x = adj.t().matmul(grad_temp)

        # 3. 计算 Bias 梯度
        if bias is not None and ctx.needs_input_grad[3]:
            grad_b = grad_out.sum(0)

        return grad_x, grad_w, None, grad_b

class GCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, g):
        # 1. 检查 adj_gp 是否存在
        if not hasattr(g, 'cache') or 'adj_gp' not in g.cache:
            raise RuntimeError("EasyGraph Error: 'adj_gp' not found in graph cache. Please run g.build_adj_gp() first.")


        return FastGCNConvFn.apply(x, self.weight, g.cache['adj_gp'], self.bias)