import decimal

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
from torch import nn
from torch.autograd import Function
#from ...utils import logging

#logger = logging.get_logger(__name__)

class round_ste(Function):
    """
    STE for torch.round()
    """

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

class floor_ste(Function):
    """
    STE for torch.floor()
    """
    
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

class QIL_weight_quantizer(Function):
    @staticmethod
    def forward(ctx, input, pruning_point, clipping_point, bitwidth):

        n = (2 ** (bitwidth - 1) - 1)

        s = n / ( clipping_point - pruning_point)
        input_q = torch.where(input.abs() < pruning_point, input * 0 , 
                (input.abs() - pruning_point) * s * torch.sign(input))

        input_q = torch.round(input_q.clamp(-n, n))
        input_dq = ( input_q.abs() / s + pruning_point ) * torch.sign(input_q)
        ctx.save_for_backward(input, input_q, pruning_point, clipping_point)
        ctx.n = n
        return input_dq

    @staticmethod
    def backward(ctx, grad_output):
        input, input_q, pruning_point, clipping_point = ctx.saved_tensors
        n = ctx.n
        distance = (clipping_point - pruning_point)

        input_mask = (input.abs() > pruning_point).float() * (input.abs() < clipping_point).float()
        clipping_mask = (input.abs() >= clipping_point).float()

        pruning_grad = torch.where(input_mask.bool(), (input.abs() - pruning_point) * torch.sign(input) / distance - input_q / n, input * 0.)
        clipping_grad = -pruning_grad
        #clipping_grad = torch.where(input_mask, -(input.abs() - pruning_point) * torch.sign(input) / distance + input_q / n, input * 0.)
        clipping_grad = clipping_grad + clipping_mask * torch.sign(input)
        
        pruning_grad = pruning_grad * grad_output
        clipping_grad = clipping_grad * grad_output
        return grad_output * input_mask, pruning_grad.sum((-1,-2)).reshape(pruning_point.shape) , clipping_grad.sum((-1,-2)).reshape(pruning_point.shape) ,None

class QIL_activation_quantizer(Function):
    @staticmethod
    def forward(ctx, input, pruning_point, clipping_point, bitwidth):

        n = (2 ** (bitwidth - 1) - 1)

        s = n / ( clipping_point - pruning_point)
        input_q = torch.where(input.abs() < pruning_point, input * 0 , 
                (input.abs() - pruning_point) * s * torch.sign(input))

        input_q = torch.round(input_q.clamp(-n, n))
        input_dq = ( input_q.abs() / s + pruning_point ) * torch.sign(input_q)
        ctx.save_for_backward(input, input_q, pruning_point, clipping_point)
        ctx.n = n
        return input_dq

    @staticmethod
    def backward(ctx, grad_output):
        input, input_q, pruning_point, clipping_point = ctx.saved_tensors
        n = ctx.n
        distance = (clipping_point - pruning_point)

        input_mask = (input.abs() > pruning_point).float() * (input.abs() < clipping_point).float()
        clipping_mask = (input.abs() >= clipping_point).float()

        pruning_grad = torch.where(input_mask.bool(), (input.abs() - pruning_point) * torch.sign(input) / distance - input_q / n, input * 0.)
        clipping_grad = -pruning_grad
        #clipping_grad = torch.where(input_mask, -(input.abs() - pruning_point) * torch.sign(input) / distance + input_q / n, input * 0.)
        clipping_grad = clipping_grad + clipping_mask * torch.sign(input)

        pruning_grad = pruning_grad * grad_output
        clipping_grad = clipping_grad * grad_output
        return grad_output * input_mask, pruning_grad.sum((0,1,-1)).reshape(pruning_point.shape) , clipping_grad.sum((0,1,-1)).reshape(pruning_point.shape) ,None

class QuantEmbedding(nn.Module):
    """
    Quantized version of 'torch.nn.Embedding'. Adds quantization-specific arguments on top of 'torch.nn.Embedding'.

    Args :
        weight_bit ('int', *optional*, defaults to '8'):
            Bitwidth for the quantized weight.
        momentum ('float', *optional*, defautls to '0.95'):
            Momentum for updating the activation quantization range ?
        quant_mode ('bool', *optional*, defaults to 'False'):
            Wheather or not the layer is quantized
    """
    def __init__(
            self, 
            num_embeddings,
            embedding_dim,
            padding_idx = None,
            max_norm = None,
            norm_type = 2.0,
            scale_grad_by_freq = False,
            sparse = False,
            _weight = None,
            weight_bit = 8,
            momentum = 0.95,
            quant_mode = False,
    ):
        super().__init__()
        self.num_ = num_embeddings
        self.dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.momentum = momentum
        if _weight is None:
            self.weight = nn.Parameter(torch.randn([num_embeddings, embedding_dim]))
        else:
            self.weight = _weight
        self.round_ste = round_ste.apply
        self.quant_mode = quant_mode
        #self.weight_function = QILQuantFunction.apply 
        self.weight_discretizer = QILWeightDiscretizer.apply

        self.bitW = weight_bit
        self.pruning_point = nn.Parameter(torch.zeros(1))
        self.clipping_point = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
        self.register_buffer("weight_quantize", torch.zeros_like(self.weight))
        self.register_buffer("weight_scaling_factor", torch.zeros_like(self.pruning_point))
        self.epsilon = 1e-12
        self.weight_module_init = True
        self.act_init_mode = False

        self.quantizer = QIL_weight_quantizer.apply

        if self.quant_mode == True:
            print('QIL Embedding Quantization Initialization with %d-bit.' %self.bitW)
        else:
            print('fine-tuning with full-precision model')

    def forward(self, x, positions = None, incremental_state = None):
        if not self.quant_mode :
            return nn.functional.embedding(
                    x,
                    self.weight,
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                ), None, None
        assert self.pruning_point < self.clipping_point, "pruning point need to be smaller than clipping_point"
        self.pruning_point.data[self.pruning_point < 0] = 0.
        #assert self.pruning_point >= 0, "center parameter not be zero"
        #transform
        data_type = torch.FloatTensor if x.device == torch.device("cpu")  else torch.cuda.FloatTensor
        
        self.weight_quantize = self.quantizer(self.weight, self.pruning_point, self.clipping_point, self.bitW)
        
        n = (2 ** (self.bitW - 1) - 1)

        self.weight_scaling_factor = n / ( self.clipping_point - self.pruning_point)
        '''
        self.weight_quantize = torch.where(self.weight.abs() < self.pruning_point, self.weight * 0, 
                (self.weight.abs() - self.pruning_point) * self.weight_scaling * torch.sign(self.weight))
        
        self.weight_quantize = self.round_ste(self.weight_quantize.clamp(-n, n))
        #self.weight_quantize = (torch.round(self.weight_quantize.clamp(-n, n)) - self.weight_quantize).detach() + \
                #self.weight_quantize
        self.weight_quantize = (self.weight_quantize.abs() / self.weight_scaling + self.pruning_point) * \
                torch.sign(self.weight_quantize)
        '''

        embed_quantize = nn.functional.embedding(
                input = x,
                weight = self.weight_quantize,
                padding_idx = self.padding_idx,
                max_norm = self.max_norm,
                norm_type =self.norm_type,
                scale_grad_by_freq = self.scale_grad_by_freq,
                sparse = self.sparse,
        )
        return embed_quantize, self.weight_scaling_factor, self.pruning_point
    
    def quantize_reset_parameter(self):
        data_type = torch.FloatTensor if self.weight.device == torch.device("cpu")  else torch.cuda.FloatTensor
        w_abs = self.weight.reshape(-1).abs().clone().detach()
        
        w_mean = w_abs.mean()
        w_std = w_abs.std()
        
        w_max = w_mean + 3 * w_std
        self.clipping_point.data = torch.Tensor([w_max]).to(self.weight.device).data

class QILActDiscretizer(Function):
    @staticmethod
    def forward(ctx, x, bit):
        n = float(2 ** (bit-1) -1)
        scaling_factor = 1 / n
        return torch.round(x / scaling_factor) * scaling_factor

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs.clone(), None

class QILWeightDiscretizer(Function):
    @staticmethod
    def forward(ctx, x, bit):
        n = float(2 ** (bit-1) -1)
        scaling_factor = 1 / n
        return torch.round(x / scaling_factor) * scaling_factor

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs.clone(), None

class DuQActDiscretizer(Function):
    @staticmethod
    def forward(ctx, x, bit):
        n = float(2**bit - 1)
        scaling_factor = 1 / n
        return torch.round(x / scaling_factor) * scaling_factor
    
    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs.clone(), None

class QILQuantAct(nn.Module):
    def __init__(self, bitA = 8, quant_mode = False, act_range_momentum = 0.95, per_group = False, num_groups = 12,channel_len = None):
        super().__init__()

        self.quant_mode = quant_mode
        self.act_range_momentum = act_range_momentum
        self.per_group = per_group
        self.num_groups = num_groups

        self.bitA = bitA
        self.act_discritizer = QILActDiscretizer.apply

        self.round_ste = round_ste.apply
        self.percentage = 0.01
        self.epsilon = 1e-12
 
        if self.per_group:
            self.pruning_point = nn.Parameter(torch.zeros([num_groups,1]))
            self.clipping_point = nn.Parameter(torch.zeros([num_groups,1]))

        else:
            self.pruning_point = nn.Parameter(torch.zeros(1))
            self.clipping_point = nn.Parameter(torch.zeros(1))
        
        self.register_buffer("act_scaling_factor", torch.zeros_like(self.clipping_point))

        if self.quant_mode == True:
            print('QIL Activation Quantization Initialization with %d-bit.' %self.bitA)
        else:
            print('fine-tuning with full-precision module')
        self.act_module_init = True
        self.act_init_mode = False
        self.quantizer = QIL_activation_quantizer.apply

    def forward(
        self,
        x,
        pre_act_scaling_factor = None,
        pre_act_offset = None,
        identity = None,
        identity_scaling_factor = None,
        identity_offset = None,
        specified_center = None,
        specified_distance = None,
        ):

        data_type = torch.FloatTensor if x.device == torch.device("cpu") else torch.cuda.FloatTensor
        x_act = x if identity is None else identity + x
        if self.per_group:
            group_shape = x_act.shape[:-1] + (self.num_groups, x_act.shape[-1] // self.num_groups)
        original_shape = x_act.shape

        if self.act_init_mode:
            x_abs = x_act.clone().detach().abs()
            if self.per_group:
                assert x_act.shape[-1] % self.num_groups == 0
                if len(x_act.shape) == 2:
                    argshape = (0, -1)
                elif len(x_act.shape) == 3:
                    argshape = (0, 1, -1)
                else :
                    raise NotImplementedError("input shape need to be 3-dimension or 2-dimension")
                x_abs = x_abs.reshape(group_shape)
                x_max = x_abs.amax(dim = argshape).reshape(self.num_groups, 1)
                #x_mean = x_abs.mean(dim = argshape).reshape(self.num_groups, 1)
                #x_std = x_abs.std(dim = argshape).reshape(self.num_groups, 1)
                #x_max = x_mean + 3 * x_std
            else:
                x_max = x_abs.max()
                #x_mean = x_abs.mean()
                #x_std = x_abs.std()
                #x_max = x_mean + 3 * x_std
            self.clipping_point.data = ( 1 - self.act_range_momentum ) * self.clipping_point.data \
                    + self.act_range_momentum * x_max.data
    
        if not self.quant_mode:
            return x_act, None, None
        assert torch.all(self.pruning_point < self.clipping_point) 
        self.pruning_point.data[self.pruning_point < 0] = 0.
        #assert torch.all(self.pruning_point >= 0), "pruning_point not be zero"
        
        n = (2 ** (self.bitA - 1) - 1)
        self.act_scaling_factor = n / (self.clipping_point - self.pruning_point)
        
        if self.per_group:
            x_act = x_act.reshape(group_shape)
        
        quantized_act = self.quantizer(x_act, self.pruning_point, self.clipping_point, self.bitA)
        '''
        quantized_act = torch.where(x_act.abs() < self.pruning_point, torch.zeros(1).type(data_type), 
                (x_act.abs() - self.pruning_point) * self.act_scaling_factor * torch.sign(x_act))
        quantized_act = self.round_ste(quantized_act.clamp(-n, n))
        #quantized_act = (torch.round(quantized_act.clamp(-n, n)) - quantized_act).detach() + quantized_act
        quantized_act = (quantized_act.abs() / self.act_scaling_factor + self.pruning_point) * \
                torch.sign(quantized_act) 
        '''
        if self.per_group:
            quantized_act = quantized_act.reshape(original_shape)
        return quantized_act, self.act_scaling_factor, self.pruning_point

class QILLinear(nn.Module):
    """
    QIL Quantized version of 'torch.nn.Linear'. Adds quantization-specific arguments on top of 'torch.nn.Linear'.

    Args:
        weight_bit ('int', *optional*, defaults to '8'):
            Bitwdith for the quantized weight.
        bias_bit ('int', *optional*, defaults to '32'):
            Bitwidth for the quantized bias.
        quant_mode ('bool', *optional*, defaults to 'False'):
            Whether or not the layer is quantized.
    """
    def __init__(
            self, in_features, out_features, bias = True, weight_bit = 8, bias_bit = 32, quant_mode = False, per_group = False, num_groups = 12
        ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.per_group = per_group
        self.num_groups = num_groups
        self.weight = nn.Parameter(torch.randn([out_features, in_features]))
        self.register_buffer("weight_quantize", torch.zeros_like(self.weight))
        self.round_ste = round_ste.apply
        if self.per_group:
            assert out_features % num_groups == 0
            self.pruning_point = nn.Parameter(torch.zeros([num_groups, 1, 1]))
            self.clipping_point = nn.Parameter(torch.ones([num_groups, 1, 1]))
        else:
            self.pruning_point = nn.Parameter(torch.zeros(1))
            self.clipping_point = nn.Parameter(torch.ones(1))

        self.register_buffer("weight_scaling_factor", torch.zeros_like(self.pruning_point))

        if bias:
            self.bias = nn.Parameter(torch.zeros([out_features]))
            self.register_buffer("bias_quantize", torch.zeros_like(self.bias))
            self.register_buffer("bias_scaling_factor", torch.zeros(1))
            self.register_buffer("bias_offset", torch.zeros(1))

        self.bitW = weight_bit
        self.quant_mode = quant_mode
        self.bias_bit = bias_bit
        self.weight_discretizer = QILWeightDiscretizer.apply
        self.epsilon = 1e-12
        self.weight_module_init = True
        self.act_init_mode = False
        
        self.quantizer = QIL_weight_quantizer.apply

    def forward(self, x, prev_act_scaling_factor = None, prev_act_offset = None):
        if not self.quant_mode:
            return nn.functional.linear(x, weight = self.weight, bias = self.bias), None, None

        data_type = torch.FloatTensor if x.device == torch.device("cpu") else torch.cuda.FloatTensor
        
        assert torch.all(self.pruning_point < self.clipping_point)
        #assert torch.all(self.pruning_point >= 0), "pruning point"
        self.pruning_point.data[self.pruning_point < 0] = 0.
        
        # 여기까지는 괜찮을듯 ..?
        n = ( 2 ** (self.bitW - 1) - 1)
        self.weight_scaling_factor = n / ( self.clipping_point - self.pruning_point)
        if self.per_group:
            original_shape = self.weight.shape
            group_shape = (self.num_groups, original_shape[0] // self.num_groups, original_shape[-1])
            self.weight_quantize = self.weight.reshape(group_shape)
        else:
            self.weight_quantize = self.weight * 1 
        self.weight_quantize = self.quantizer(self.weight_quantize, self.pruning_point, self.clipping_point, self.bitW)
        '''
        self.weight_quantize = torch.where(self.weight_quantize.abs() < self.pruning_point, self.weight_quantize * 0., \
                (self.weight_quantize.abs() - self.pruning_point) * self.weight_scaling_factor * \
                torch.sign(self.weight_quantize))
        #self.weight_quantize = (torch.round(self.weight_quantize.clamp(-n, n)) - self.weight_quantize).detach() + \
        #       self.weight_quantize
        self.weight_quantize = self.round_ste(self.weight_quantize.clamp(-n, n))

        self.weight_quantize = (self.weight_quantize.abs() / self.weight_scaling_factor + self.pruning_point) * \
                torch.sign(self.weight_quantize)
        '''
        if self.per_group:
            self.weight_quantize = self.weight_quantize.reshape(original_shape)
        return nn.functional.linear(x, weight = self.weight_quantize, bias = self.bias), None, None


    def quantize_reset_parameter(self):
        w_abs = self.weight.abs().clone().detach()
        if self.per_group:
            w_abs = w_abs.reshape(self.num_groups, -1)
            #w_max = w_abs.max(dim = -1, keepdim = True)[0]
            #w_max.unsqueeze_(-1)
            w_mean = w_abs.mean(dim = -1, keepdim = True)[0]
            w_std = w_abs.std(dim = -1, keepdim = True)[0]
            w_max = w_mean + 3 * w_std
            w_max.unsqueeze_(-1)
        else:
            #w_max = w_abs.max()
            #w_max = torch.Tensor([w_max]).to(self.weight.device)
            w_mean = w_abs.mean(dim = -1, keepdim = True)[0]
            w_std = w_abs.std(dim = -1, keepdim = True)[0]
            w_max = w_mean + 3 * w_std
        self.clipping_point.data = w_max.data

if __name__ == "__main__":
    '''
    Embedding_test = QuantEmbedding(1000, 80, weight_bit = 3, quant_mode = True)
    Embedding_test.quantize_reset_parameter()
    x = torch.randint(low = 0, high = 4, size = (3,))
    y, _, _=Embedding_test(x)
    #plt.hist(Embedding_test.weight.reshape(-1).detach(), bins = 100)
    #plt.hist(Embedding_test.weight_quantize.reshape(-1).detach(), bins = 100)
    #plt.show()
    loss = y.sum()
    loss.backward()
    for name,p in Embedding_test.named_parameters():
        print(name,p.grad)
    '''
    Linear_test = QILLinear(100, 80, weight_bit = 3,per_group = False, num_groups = 8,quant_mode = True)
    print(Linear_test.weight)
    Linear_test.quantize_reset_parameter()
    x = torch.rand((1, 100))
    print(x, "x")
    y, _, _ = Linear_test(x)
    print(Linear_test.weight_quantize)
    #plt.hist(Linear_test.weight.reshape(-1).detach(), bins = 100)
    #plt.hist(Linear_test.weight_quantize.reshape(-1).detach(), bins = 100)
    #plt.show()
    loss = y.sum()
    loss.backward()
    for name,p in Linear_test.named_parameters():
        print(name,p.grad)
    '''
    Act_test = QILQuantAct(bitA = 3, quant_mode = False, per_group =True, num_groups = 5)
    x = torch.randn((2, 2, 1000))
    print(x)
    print(Act_test(x))
    Act_test.act_init_mode = True
    y= Act_test(x)
    print(y)
    Act_test.quant_mode = True
    Act_test.act_init_mode = False
    y, _, _ = Act_test(x)
    print(y)
    plt.hist(y[0].reshape(-1).detach(), bins = 200)
    plt.hist(x[0].reshape(-1).detach(), bins = 200)
    plt.show()
    loss = y.sum()
    loss.backward()
    for name,p in Act_test.named_parameters():
        print(name,p.grad)
    '''
