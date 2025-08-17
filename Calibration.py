from __future__ import annotations
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import tqdm
import os
import pickle
import sympy as sp
import torch.optim.lr_scheduler as lr_scheduler

### sympytorch ###
import collections as co
import functools as ft
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Sequence,
    Tuple,
    Type,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

import sympy
import torch


ExprType = TypeVar("ExprType", bound=sympy.Expr)
T = TypeVar("T")

if TYPE_CHECKING:
    # Because there are methods of our class objects below called `sympy` that
    # implicitly override the `sympy` name in the global namespace while defining other
    # methods, our type checker needs to know that we're referring to the sympy module
    # in our type annotations.
    import sympy as sympy_


def _reduce(fn: Callable[..., T]) -> Callable[..., T]:
    def fn_(*args: Any) -> T:
        return ft.reduce(fn, args)

    return fn_


def _I(*args: Any) -> torch.Tensor:
    return torch.tensor(1j)


_global_func_lookup: Dict[
    Union[Type[sympy.Basic], Callable[..., Any]], Callable[..., torch.Tensor]
] = {
    sympy.Mul: _reduce(torch.mul),
    sympy.Add: _reduce(torch.add),
    sympy.div: torch.div,
    sympy.Abs: torch.abs,
    sympy.sign: torch.sign,
    # Note: May raise error for ints.
    sympy.ceiling: torch.ceil,
    sympy.floor: torch.floor,
    sympy.log: torch.log,
    sympy.exp: torch.exp,
    sympy.sqrt: torch.sqrt,
    sympy.cos: torch.cos,
    sympy.acos: torch.acos,
    sympy.sin: torch.sin,
    sympy.asin: torch.asin,
    sympy.tan: torch.tan,
    sympy.atan: torch.atan,
    sympy.atan2: torch.atan2,
    # Note: May give NaN for complex results.
    sympy.cosh: torch.cosh,
    sympy.acosh: torch.acosh,
    sympy.sinh: torch.sinh,
    sympy.asinh: torch.asinh,
    sympy.tanh: torch.tanh,
    sympy.atanh: torch.atanh,
    sympy.Pow: torch.pow,
    sympy.re: torch.real,
    sympy.im: torch.imag,
    sympy.arg: torch.angle,
    # Note: May raise error for ints and complexes
    sympy.erf: torch.erf,
    sympy.loggamma: torch.lgamma,
    sympy.Eq: torch.eq,
    sympy.Ne: torch.ne,
    sympy.StrictGreaterThan: torch.gt,
    sympy.StrictLessThan: torch.lt,
    sympy.LessThan: torch.le,
    sympy.GreaterThan: torch.ge,
    sympy.And: torch.logical_and,
    sympy.Or: torch.logical_or,
    sympy.Not: torch.logical_not,
    sympy.Max: torch.max,
    sympy.Min: torch.min,
    # Matrices
    sympy.MatAdd: torch.add,
    sympy.HadamardProduct: torch.mul,
    sympy.Trace: torch.trace,
    # Note: May raise error for integer matrices.
    sympy.Determinant: torch.det,
    sympy.core.numbers.ImaginaryUnit: _I,
    sympy.conjugate: torch.conj,
}

number_symbols = [cls for cls in sympy.NumberSymbol.__subclasses__()]


def number_symbol_to_torch(symbol: sympy.NumberSymbol, *args: Any) -> torch.Tensor:
    return torch.tensor(float(symbol))


_global_func_lookup.update(
    {s: ft.partial(number_symbol_to_torch, s()) for s in number_symbols}
)


class _Node(torch.nn.Module, Generic[ExprType]):
    def __init__(
        self,
        *,
        expr: ExprType,
        _memodict: Dict[sympy.Basic, torch.nn.Module],
        _func_lookup,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self._sympy_func: Type[ExprType] = expr.func

        self._torch_func: Callable[..., torch.Tensor]
        self._args: Union[
            torch.nn.ModuleList,
            Tuple[Callable[[Dict[str, torch.Tensor]], torch.Tensor], ...],
        ]
        self._value: Any

        if issubclass(expr.func, sympy.Float):
            self._value = torch.tensor(float(expr))
            self._torch_func = lambda: self._value
            self._args = ()
        elif issubclass(expr.func, sympy.Integer):
            self._value = torch.tensor(int(expr))
            self._torch_func = lambda: self._value
            self._args = ()
        elif issubclass(expr.func, sympy.Rational):
            self._numerator: torch.Tensor
            self._denominator: torch.Tensor
            assert isinstance(expr, sympy.Rational)
            self.register_buffer(
                "_numerator", torch.tensor(expr.p, dtype=torch.get_default_dtype(),requires_grad=False)
            )
            self.register_buffer(
                "_denominator", torch.tensor(expr.q, dtype=torch.get_default_dtype(),requires_grad=False)
            )
            self._torch_func = lambda: self._numerator / self._denominator
            self._args = ()
        elif issubclass(expr.func, sympy.UnevaluatedExpr):
            if len(expr.args) != 1 or not issubclass(expr.args[0].func, sympy.Float):
                raise ValueError("UnevaluatedExpr should only be used to wrap floats.")
            assert isinstance(expr.args[0], sympy.Float)
            self.register_buffer("_value", torch.tensor(float(expr.args[0])),requires_grad=False)
            self._torch_func = lambda: self._value
            self._args = ()
        elif issubclass(expr.func, sympy.Symbol):
            assert isinstance(expr, sympy.Symbol)
            self._name = expr.name
            self._torch_func = lambda value: value
            self._args = ((lambda memodict: memodict[expr.name]),)
        else:
            self._torch_func = _func_lookup[expr.func]
            args: List[torch.nn.Module] = []
            for arg in expr.args:
                try:
                    arg_ = _memodict[arg]
                except KeyError:
                    arg_ = type(self)(
                        expr=arg,  # type: ignore
                        _memodict=_memodict,
                        _func_lookup=_func_lookup,
                        **kwargs,
                    )
                    _memodict[arg] = arg_
                args.append(arg_)
            self._args = torch.nn.ModuleList(args)

    def sympy(self, _memodict: Dict[_Node, sympy_.Expr]) -> ExprType:
        if issubclass(self._sympy_func, sympy.Float):
            assert isinstance(self._value, torch.nn.Parameter)
            return self._sympy_func(self._value.item())
        elif issubclass(self._sympy_func, sympy.UnevaluatedExpr):
            assert isinstance(self._value, torch.Tensor)
            return self._sympy_func(self._value.item())
        elif issubclass(
            self._sympy_func,
            (type(sympy.S.NegativeOne), type(sympy.S.One), type(sympy.S.Zero)),
        ):
            return self._sympy_func()
        elif issubclass(self._sympy_func, sympy.Integer):
            return self._sympy_func(self._value)
        elif issubclass(self._sympy_func, sympy.Rational):
            if issubclass(self._sympy_func, type(sympy.S.Half)):
                return sympy.S.Half
            else:
                return self._sympy_func(
                    self._numerator.item(), self._denominator.item()
                )
        elif issubclass(self._sympy_func, sympy.Symbol):
            return self._sympy_func(self._name)
        elif issubclass(self._sympy_func, sympy.core.numbers.ImaginaryUnit):
            return sympy.I
        elif issubclass(self._sympy_func, sympy.core.numbers.NumberSymbol):
            return self._sympy_func()
        else:
            if issubclass(self._sympy_func, (sympy.Min, sympy.Max)):
                evaluate = False
            else:
                evaluate = True
            args = []
            for arg in self._args:
                assert isinstance(arg, _Node)
                try:
                    arg_ = _memodict[arg]
                except KeyError:
                    arg_ = arg.sympy(_memodict)
                    _memodict[arg] = arg_
                args.append(arg_)
            return self._sympy_func(*args, evaluate=evaluate)  # type: ignore

    def forward(self, memodict) -> torch.Tensor:
        args = []
        for arg in self._args:
            try:
                arg_ = memodict[arg]
            except KeyError:
                arg_ = arg(memodict)
                memodict[arg] = arg_
            args.append(arg_)
        return self._torch_func(*args)


class SymPyModule(torch.nn.Module):
    def __init__(self, *, expressions, extra_funcs=None, **kwargs):
        super().__init__(**kwargs)

        expressions = tuple(expressions)

        if extra_funcs is None:
            extra_funcs = {}
        _func_lookup = co.ChainMap(_global_func_lookup, extra_funcs)

        _memodict = {}
        self._nodes: Sequence[_Node] = torch.nn.ModuleList(  # type: ignore
            [
                _Node(expr=expr, _memodict=_memodict, _func_lookup=_func_lookup)
                for expr in expressions
            ]
        )
        self._expressions_string = str(expressions)

    def __repr__(self):
        return f"{type(self).__name__}(expressions={self._expressions_string})"

    def sympy(self) -> List[sympy.Expr]:
        _memodict: Dict[_Node, sympy.Expr] = {}
        return [node.sympy(_memodict) for node in self._nodes]

    def forward(self, **symbols: Any) -> torch.Tensor:
        out = [node(symbols) for node in self._nodes]
        out = torch.broadcast_tensors(*out)
        return torch.stack(out, dim=-1)

### sympytorch ###
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
def create_walker_quicker():
    path = f'11.pickle'
    with open(path,'rb') as file:
        f=pickle.load(file)
    mod=SymPyModule(expressions=[*f[2:-2]])
    return mod
def phi_num(layer):
    sum = 0
    for i in range(layer+1):
        sum += i
    return sum

def extract_train_data(num,batch_size,path):
    b = np.load(os.path.join(path,'data_prob_train.npy'))
    a = np.load(os.path.join(path,'data_voltage_train.npy'))

    a = torch.tensor(a[:num],dtype=torch.float64, device=device)
    b = torch.tensor(b[:num],dtype=torch.float64, device=device)
    train_data = TensorDataset(*(a, b))

    return DataLoader(train_data, batch_size, shuffle=True)

def extract_test_data(path):
    a = np.load(os.path.join(path,'data_voltage_test.npy'))
    b = np.load(os.path.join(path,'data_prob_test.npy'))

    a = torch.tensor(a,dtype=torch.float64, device=device)
    b = torch.tensor(b,dtype=torch.float64, device=device)
    return (a,b)

    
class MyLinear(nn.Module):
    def __init__(self, in_units):
        super().__init__()
        if False:
            self.a = torch.tensor([0.1202, 0.1204, 0.1202, 0.1200, 0.1199, 0.1203, 0.1203, 0.1198, 0.1198,
        0.1197, 0.1194, 0.1196, 0.1202, 0.1198, 0.1206, 0.1191, 0.1194, 0.1200,
        0.1199, 0.1205, 0.1211, 0.1193, 0.1200, 0.1197, 0.1199, 0.1204, 0.1194,
        0.1194, 0.1206, 0.1205, 0.1194, 0.1200, 0.1198, 0.1201, 0.1205, 0.1202,
        0.1207, 0.1186, 0.1211, 0.1198, 0.1201, 0.1201, 0.1205, 0.1203, 0.1202,
        0.1190, 0.1193, 0.1200, 0.1191, 0.1201, 0.1200, 0.1201, 0.1207, 0.1226,
        0.3114, 0.1166, 0.1188, 0.1202, 0.1203, 0.1194, 0.1200, 0.1201, 0.1198,
        0.1208, 0.1213],requires_grad=True,device=device,dtype=torch.float64,
             )
        else:
            self.a=torch.ones(65,device=device,dtype=torch.float64)*0.12

        if False:
            self.b = torch.tensor([ 6.4735e-01,  4.0863e-02,  9.0365e-01, -3.1057e-01,  1.3046e+00,
         5.2484e-01,  4.9945e-01,  9.5171e-03,  8.3274e-01, -1.2338e+00,
        -5.6380e-01, -1.0626e+00,  1.0361e+00,  6.0038e-01,  1.3028e+00,
         3.2624e+00,  1.1828e-03,  3.9835e-01, -2.4063e-01,  9.6431e-01,
         9.1401e-01, -2.2146e+00, -1.1623e+00,  7.0748e-01,  6.7849e-01,
         1.2569e+00,  4.3714e-01,  8.9281e-01,  2.4029e+00,  2.3094e+00,
         1.1855e+00,  1.7231e-01, -1.0658e-01,  1.9518e-01,  8.6280e-01,
         1.2138e-01, -1.8907e-01,  3.8424e+00,  8.3996e-01,  3.0178e-01,
         6.2261e-01,  1.2308e+00,  1.6269e-01,  3.1303e-01, -5.8488e-01,
         2.5459e-01, -2.3307e+00, -1.5257e+00,  1.1440e+00, -1.2264e-01,
         4.5227e-02,  7.6829e-02,  6.0232e-01,  3.5698e-01,  9.7223e-01,
        -3.5046e-01,  8.3127e-01,  4.5308e-01,  1.6988e+00, -2.9536e-01,
         5.4771e-01,  1.4278e+00,  9.7644e-01,  1.4116e+00,  8.5075e-02],requires_grad=True,device=device,dtype=torch.float64,
             )
        else:   
            self.b=torch.rand(65,device=device,dtype=torch.float64)

        self.a = nn.Parameter(self.a,requires_grad=True)
        self.b = nn.Parameter(self.b,requires_grad=True)
    def forward(self,X):
        return self.a *X**2+self.b
class chip(nn.Module):
    def __init__(self,mod):
        super().__init__()
        self.mod=mod
        if False:
            self.bs = torch.tensor(
                [0.8095, 0.7812, 0.7847, 0.7662, 0.7918, 0.8003, 0.7114, 0.7704, 0.7834,
                0.7906, 0.7111, 0.7995, 0.7713, 0.7808, 0.7999, 1.9529, 0.7768, 0.7806,
                0.7957, 0.7780, 0.8004, 0.8597, 0.7767, 0.7722, 0.7923, 0.7797, 0.7807,
                0.8036, 2.2161, 0.7844, 0.7786, 0.7982, 0.7696, 0.7785, 0.7755, 0.8386,
                0.9119, 0.7697, 0.7843, 0.7851, 0.7896, 0.7882, 0.7890, 0.7785, 0.8601,
                1.0598, 0.7898, 0.7823, 0.7690, 0.7825, 0.7809, 0.7869, 0.8073, 0.7957,
                0.9296, 1.0765, 0.8190, 0.7724, 0.8018, 0.7659, 0.8012, 0.7787, 0.7874,
                0.7759, 0.7754, 1.0007, 0.7854, 0.6777, 0.7770, 0.7790, 0.7712, 0.8028,
                0.7832, 0.7813, 0.7780, 0.7989, 0.8276, 0.7854],requires_grad=True,device=device,dtype=torch.float64)
        else:
            self.bs=torch.ones(78,device=device,dtype=torch.float64)*np.pi/4
        self.bs = nn.Parameter(self.bs,requires_grad=True)

        if False:
            self.alpha = torch.tensor(
                [0.8095, 0.7812, 0.7847, 0.7662, 0.7918, 0.8003, 0.7114, 0.7704, 0.7834,
                0.7906, 0.7111, 0.7995, 0.7713, 0.7808, 0.7999, 1.9529, 0.7768, 0.7806,
                0.7957, 0.7780, 0.8004, 0.8597, 0.7767, 0.7722, 0.7923, 0.7797, 0.7807,
                0.8036, 2.2161, 0.7844, 0.7786, 0.7982, 0.7696, 0.7785, 0.7755, 0.8386,
                0.9119, 0.7697, 0.7843, 0.7851, 0.7896, 0.7882, 0.7890, 0.7785, 0.8601,
                1.0598, 0.7898, 0.7823, 0.7690, 0.7825, 0.7809, 0.7869, 0.8073, 0.7957,
                0.9296, 1.0765, 0.8190, 0.7724, 0.8018, 0.7659, 0.8012, 0.7787, 0.7874,
                0.7759, 0.7754, 1.0007, 0.7854, 0.6777, 0.7770, 0.7790, 0.7712, 0.8028,
                0.7832, 0.7813, 0.7780, 0.7989, 0.8276, 0.7854],requires_grad=True,device=device,dtype=torch.float64)
        else:
            self.alpha=torch.ones(78,device=device,dtype=torch.float64)
        self.alpha = nn.Parameter(self.alpha,requires_grad=True)
    def forward(self,X):
        return self.mod(phi_1=X[:,0],phi_2=X[:,1],phi_3=X[:,2],phi_4=X[:,3],phi_5=X[:,4],phi_6=X[:,5],phi_7=X[:,6],phi_8=X[:,7],phi_9=X[:,8],phi_10=X[:,9],
                        phi_11=X[:,10],phi_12=X[:,11],phi_13=X[:,12],phi_14=X[:,13],phi_15=X[:,14],phi_16=X[:,15],phi_17=X[:,16],phi_18=X[:,17],phi_19=X[:,18],phi_20=X[:,19],
                        phi_21=X[:,20],phi_22=X[:,21],phi_23=X[:,22],phi_24=X[:,23],phi_25=X[:,24],phi_26=X[:,25],phi_27=X[:,26],phi_28=X[:,27],phi_29=X[:,28],phi_30=X[:,29],
                        phi_31=X[:,30],phi_32=X[:,31],phi_33=X[:,32],phi_34=X[:,33],phi_35=X[:,34],phi_36=X[:,35],phi_37=X[:,36],phi_38=X[:,37],phi_39=X[:,38],phi_40=X[:,39],
                        phi_41=X[:,40],phi_42=X[:,41],phi_43=X[:,42],phi_44=X[:,43],phi_45=X[:,44],phi_46=X[:,45],phi_47=X[:,46],phi_48=X[:,47],phi_49=X[:,48],phi_50=X[:,49],
                        phi_51=X[:,50],phi_52=X[:,51],phi_53=X[:,52],phi_54=X[:,53],phi_55=X[:,54],phi_56=X[:,55],phi_57=X[:,56],phi_58=X[:,57],phi_59=X[:,58],phi_60=X[:,59],
                        phi_61=X[:,60],phi_62=X[:,61],phi_63=X[:,62],phi_64=X[:,63],phi_65=X[:,64],phi_66=torch.zeros_like(X[:,64],requires_grad=True,dtype=torch.float64, device=device),
                        bs_1=self.bs[0],bs_2=self.bs[1],bs_3=self.bs[2],bs_4=self.bs[3],bs_5=self.bs[4],bs_6=self.bs[5],bs_7=self.bs[6],bs_8=self.bs[7],bs_9=self.bs[8],bs_10=self.bs[9],
                        bs_11=self.bs[10],bs_12=self.bs[11],bs_13=self.bs[12],bs_14=self.bs[13],bs_15=self.bs[14],bs_16=self.bs[15],bs_17=self.bs[16],bs_18=self.bs[17],bs_19=self.bs[18],bs_20=self.bs[19],
                        bs_21=self.bs[20],bs_22=self.bs[21],bs_23=self.bs[22],bs_24=self.bs[23],bs_25=self.bs[24],bs_26=self.bs[25],bs_27=self.bs[26],bs_28=self.bs[27],bs_29=self.bs[28],bs_30=self.bs[29],
                        bs_31=self.bs[30],bs_32=self.bs[31],bs_33=self.bs[32],bs_34=self.bs[33],bs_35=self.bs[34],bs_36=self.bs[35],bs_37=self.bs[36],bs_38=self.bs[37],bs_39=self.bs[38],bs_40=self.bs[39],
                        bs_41=self.bs[40],bs_42=self.bs[41],bs_43=self.bs[42],bs_44=self.bs[43],bs_45=self.bs[44],bs_46=self.bs[45],bs_47=self.bs[46],bs_48=self.bs[47],bs_49=self.bs[48],bs_50=self.bs[49],
                        bs_51=self.bs[50],bs_52=self.bs[51],bs_53=self.bs[52],bs_54=self.bs[53],bs_55=self.bs[54],bs_56=self.bs[55],bs_57=self.bs[56],bs_58=self.bs[57],bs_59=self.bs[58],bs_60=self.bs[59],
                        bs_61=self.bs[60],bs_62=self.bs[61],bs_63=self.bs[62],bs_64=self.bs[63],bs_65=self.bs[64],bs_66=self.bs[65],bs_67=self.bs[66],bs_68=self.bs[67],bs_69=self.bs[68],bs_70=self.bs[69],
                        bs_71=self.bs[70],bs_72=self.bs[71],bs_73=self.bs[72],bs_74=self.bs[73],bs_75=self.bs[74],bs_76=self.bs[75],bs_77=self.bs[76],bs_78=self.bs[77],
                        alpha_1=self.alpha[0],alpha_2=self.alpha[1],alpha_3=self.alpha[2],alpha_4=self.alpha[3],alpha_5=self.alpha[4],alpha_6=self.alpha[5],alpha_7=self.alpha[6],alpha_8=self.alpha[7],alpha_9=self.alpha[8],alpha_10=self.alpha[9],
                        alpha_11=self.alpha[10],alpha_12=self.alpha[11],alpha_13=self.alpha[12],alpha_14=self.alpha[13],alpha_15=self.alpha[14],alpha_16=self.alpha[15],alpha_17=self.alpha[16],alpha_18=self.alpha[17],alpha_19=self.alpha[18],alpha_20=self.alpha[19],
                        alpha_21=self.alpha[20],alpha_22=self.alpha[21],alpha_23=self.alpha[22],alpha_24=self.alpha[23],alpha_25=self.alpha[24],alpha_26=self.alpha[25],alpha_27=self.alpha[26],alpha_28=self.alpha[27],alpha_29=self.alpha[28],alpha_30=self.alpha[29],
                        alpha_31=self.alpha[30],alpha_32=self.alpha[31],alpha_33=self.alpha[32],alpha_34=self.alpha[33],alpha_35=self.alpha[34],alpha_36=self.alpha[35],alpha_37=self.alpha[36],alpha_38=self.alpha[37],alpha_39=self.alpha[38],alpha_40=self.alpha[39],
                        alpha_41=self.alpha[40],alpha_42=self.alpha[41],alpha_43=self.alpha[42],alpha_44=self.alpha[43],alpha_45=self.alpha[44],alpha_46=self.alpha[45],alpha_47=self.alpha[46],alpha_48=self.alpha[47],alpha_49=self.alpha[48],alpha_50=self.alpha[49],
                        alpha_51=self.alpha[50],alpha_52=self.alpha[51],alpha_53=self.alpha[52],alpha_54=self.alpha[53],alpha_55=self.alpha[54],alpha_56=self.alpha[55],alpha_57=self.alpha[56],alpha_58=self.alpha[57],alpha_59=self.alpha[58],alpha_60=self.alpha[59],
                        alpha_61=self.alpha[60],alpha_62=self.alpha[61],alpha_63=self.alpha[62],alpha_64=self.alpha[63],alpha_65=self.alpha[64],alpha_66=self.alpha[65],alpha_67=self.alpha[66],alpha_68=self.alpha[67],alpha_69=self.alpha[68],alpha_70=self.alpha[69],
                        alpha_71=self.alpha[70],alpha_72=self.alpha[71],alpha_73=self.alpha[72],alpha_74=self.alpha[73],alpha_75=self.alpha[74],alpha_76=self.alpha[75],alpha_77=self.alpha[76],alpha_78=self.alpha[77])

class coupling(nn.Module):
    def __init__(self):
        super().__init__()
        if False:
            self.coupling = nn.Parameter(torch.tensor(
              [1.2637, 1.5910, 1.8614, 1.4693, 1.4670, 0.7401, 1.3729, 1.4273, 0.3025,
        0.8245, 1.3381, 0.3702, 0.8329, 1.2684, 1.4453, 1.0038, 0.9259, 1.0960,
        1.4548, 2.0235],requires_grad=True,dtype=torch.float64, device=device))
        else:
            self.coupling = nn.Parameter(torch.ones(20,dtype=torch.float64, device=device))
        self.coupling = nn.Parameter(self.coupling,requires_grad=True)
    def forward(self,X):
        #coupler = torch.cat([torch.tensor([1],requires_grad=True,dtype=torch.float64, device=device),self.coupling],dim=0)
        #p=coupler*X
        p=self.coupling*torch.abs(X)**2
        return (p/torch.sum(p,dim=1).reshape(-1,1))
    

def train_function(net, trainer_fn, loss, hyperparams, data_iter, num_epochs=4):

    #Initialization
    #def init_weights(m):
        #if type(m) == nn.Linear:
            #torch.nn.init.normal(m.weight)
            #torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    #net.apply(init_weights)
    if continue_train:
        net.load_state_dict(torch.load(params_path))

    optimizer = trainer_fn(net.parameters(), **hyperparams)
    scheduler2 = lr_scheduler.MultiStepLR(optimizer, milestones=[230],gamma=0.1)
    scheduler1 = lr_scheduler.StepLR(optimizer, step_size=1,gamma=0.995)
    #记录loss函数
    test_out = net(data_test[0])
    test_out = test_out.reshape(data_test[1].shape)
    loss_test = loss(test_out,data_test[1])
    dist_test = torch.mean(torch.sum(torch.abs(test_out-data_test[1]),dim=-1))
    print('test loss',loss_test.cpu().detach().numpy())
    print('test dist',dist_test.cpu().detach().numpy())
    dist_list = []
    loss_list = []
    test_dist_list = []
    test_loss_list = []
    lr_list=[]
    flag = 0
    epoch = 0
    temp=0
    th=5
    for name,param in net.named_parameters():
        print(name)
    while True:
        loss_train, l2 = 0, 0
        dist = 0
        
        if False:
            if np.abs(dist_list[-1]-dist_list[-2])<0.00001 and flag<=8:
                print('******************************')
                print(flag)
                temp+=1
                if temp <th:
                    continue
                elif temp == th:
                    for name,param in net.named_parameters():
                        if flag%2==0:
                            if '0.a' in name:
                                param.requires_grad=True
                            if '0.b' in name:
                                param.requires_grad= False
                            print(f'111111111111111111111111111{flag}')
                        elif flag%2==1:
                            if '0.a' in name:
                                param.requires_grad=False
                            if '0.b' in name:
                                param.requires_grad=True
                            print(f'222222222222222222222222222{flag}')
                    flag+=1
                    temp=0
                if flag == 6:
                    lr_list.append(0.001)
                optimizer = trainer_fn(net.parameters(), lr=lr_list[-1])
            elif np.abs(dist_list[-1]-dist_list[-2])<0.00001 and flag==9:
                temp+=1
                if temp <th:
                    continue
                elif temp == th:
                    for name,param in net.named_parameters():
                        if '0.a' in name:
                            param.requires_grad=False
                        if '0.b' in name:
                            param.requires_grad= False
                        if '1.bs' in name:
                            param.requires_grad=True
                        if '1.alpha' in name:
                            param.requires_grad=True
                    optimizer = trainer_fn(net.parameters(), lr=lr_list[-1])
                    print(f'33333333333333333333333333333333333333333333{flag}')
                    flag+=1
                    temp=0
            elif np.abs(dist_list[-1]-dist_list[-2])<0.00001 and flag==10:
                temp+=1
                if temp <th:
                    continue
                elif temp == th:
                    for name,param in net.named_parameters():
                        if '0.a' in name:
                            param.requires_grad=True
                        if '0.b' in name:
                            param.requires_grad= True
                        if '1.bs' in name:
                            param.requires_grad=False
                        if '1.alpha' in name:
                            param.requires_grad=False
                    optimizer = trainer_fn(net.parameters(), lr=lr_list[-1])
                    print(f'3333333333333333333333333333333{flag}')
                    flag+=1
                    temp=0
            elif np.abs(dist_list[-1]-dist_list[-2])<0.00001 and flag==11:
                temp+=1
                if temp <th:
                    continue
                elif temp == th:
                    for name,param in net.named_parameters():
                        if '0.a' in name:
                            param.requires_grad=True
                        if '0.b' in name:
                            param.requires_grad= True
                        if '1.bs' in name:
                            param.requires_grad=True
                        if '1.alpha' in name:
                            param.requires_grad=True
                    optimizer = trainer_fn(net.parameters(), lr=lr_list[-1])
                    print(f'bsbsbsbsbsbsbsbsbsbsbsbsbsbsbsb{flag}')
                    flag+=1    
                    temp=0
            elif np.abs(dist_list[-1]-dist_list[-2])<0.00001 and flag==12:
                temp+=1
                if temp <th:
                    continue
                elif temp == th:
                    break
            elif np.abs(dist_list[-1]-dist_list[-2])>=0.0001:
                temp=0
        for X, y in data_iter:
            optimizer.zero_grad()
            out = net(X)
            y = y.reshape(out.shape)
            l_t = loss(out,y).mean()
            loss_train += l_t.item()*batch_size
            l_t.backward(retain_graph=True)
            optimizer.step()
            dist += torch.sum(torch.sum(torch.abs(out-y),dim=-1))
        scheduler1.step()
        #scheduler2.step()
        #记录测试集的loss
        dist = dist/ data_num
        loss_train = loss_train / data_num
        loss_list.append(loss_train)
        dist_list.append(dist.cpu().detach().numpy())

        test_out = net(data_test[0])
        test_out = test_out.reshape(data_test[1].shape)
        
        loss_test = loss(test_out,data_test[1])
        dist_test = torch.mean(torch.sum(torch.abs(test_out-data_test[1]),dim=-1))
        test_loss_list.append(loss_test)
        test_dist_list.append(dist_test)
        print(f'***********{epoch}*************')
        print('train loss:', loss_train)
        print('test loss',loss_test.cpu().detach().numpy())
        print('train dist:', dist.cpu().detach().numpy())
        print('test dist',dist_test.cpu().detach().numpy())
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
        epoch+=1

        if epoch==num_epochs:break




    torch.save(net.state_dict(),params_path)

    np.savetxt(os.path.join(path,f'loss({data_num}).txt'),np.array(loss_list))
    np.savetxt(os.path.join(path,f'dist({data_num}).txt'),np.array(dist_list))

    plt.figure()
    plt.plot(np.arange(len(loss_list)),loss_list)
    plt.savefig(os.path.join(path,'loss_plt.png'))
    plt.figure()
    plt.plot(np.arange(len(dist_list)),dist_list)
    plt.savefig(os.path.join(path,'dist_plt.png'))
    
batch_size= 128
layer =11
data_num = 1225

num=phi_num(layer)-layer
lr, num_epochs, betas = 0.0001 ,100, (0.9, 0.999)
trainer = torch.optim.Adam
loss=nn.L1Loss(reduction='mean')
mod1 = create_walker_quicker()
print('transfer accomplished')
net = nn.Sequential(
    MyLinear(3),chip(mod1),coupling()
)
net.to(device)


if __name__=='__main__':
    add_noise = False
    path = f'result'
    if os.path.exists(path) == False:
        os.makedirs(path)

    continue_train = True
    params_path = f'{layer}layer({data_num}).pth'
    train_path = f'data'
    test_path = f'data_test'
    
    data= extract_train_data(data_num,batch_size,train_path)
    data_test = extract_test_data(test_path)
    time1 = time.time()
    train_function(net, trainer, loss, {'lr': lr},data, num_epochs=num_epochs)
    print(time.time()-time1)