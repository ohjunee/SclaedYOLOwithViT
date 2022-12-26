# This file contains modules common to various models
from utils.utils import *
from utils.activations import *
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor

try:
    from mish_cuda import MishCuda as Mish
except:
    class Mish(nn.Module):  # https://github.com/digantamisra98/Mish
        def forward(self, x):
            return x * torch.nn.functional.softplus(x).tanh()



def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p




def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Mish(nn.Module):  # https://github.com/digantamisra98/Mish
    def forward(self, x):
        return x * torch.nn.functional.softplus(x).tanh()

class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        x1 = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        return self.conv(x1)



class Conv(nn.Module):
    # Standard convolution: conv -> BatchNorm -> Mish()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = Mish() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions

        return x

class TransformerLayer_ViT(nn.Module):
    def __init__(self, c, num_heads):
        super().__init__()

        self.ln1 = nn.LayerNorm(c)
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.ln2 = nn.LayerNorm(c)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)
        self.dropout = nn.Dropout(0.1)
        self.act = SiLU()
        self.pe = PatchEmbedding()

    def forward(self, x):
        x_ = self.ln1(x)
        x = self.dropout(self.ma(self.q(x_), self.k(x_), self.v(x_))[0]) + x
        x_ = self.ln2(x)
        x_ = self.fc2(self.dropout(self.act(self.fc1(x_))))
        x = x + self.dropout(x_)
        return x

class TransformerLayer_light(nn.Module):
    def __init__(self, c, num_heads):
        super().__init__()

#        self.ln1 = nn.LayerNorm(c)
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
#        self.ln2 = nn.LayerNorm(c)
#        self.fc1 = nn.Linear(c, c, bias=False)
#        self.fc2 = nn.Linear(c, c, bias=False)
#        self.dropout = nn.Dropout(0.1)
#        self.act = SiLU()

    def forward(self, x):
#        x_ = self.ln1(x)
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
#        x = self.dropout(self.ma(self.q(x), self.k(x), self.v(x))[0]) + x
#        x_ = self.ln2(x)
#        x_ = self.fc2(self.dropout(self.act(self.fc1(x_))))
#        x = x + self.dropout(x_)
        return x

class TransformerLayer_ViT_(nn.Module):
    def __init__(self, c, num_heads, use_HLN=False):
        super().__init__()
        self.ln = nn.LayerNorm(c)

        # HINet https://github.com/megvii-model/HINet/blob/main/basicsr/models/archs/hinet_arch.py
        if use_HLN:
            self.ln = nn.LayerNorm(c // 2)
        self.use_HLN = use_HLN

        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)
        self.dropout = nn.Dropout(0.1)
        self.act = SiLU()

    def forward(self, x):
        if self.use_HLN:
            out_1, out_2 = torch.chunk(x, 2, dim=2)
            x_ = torch.cat([self.ln(out_1), out_2], dim=2)
        else:
            x_ = self.ln(x)

        x = self.dropout(self.ma(self.q(x_), self.k(x_), self.v(x_))[0]) + x

        if self.use_HLN:
            out_1, out_2 = torch.chunk(x, 2, dim=2)
            x_ = torch.cat([self.ln(out_1), out_2], dim=2)
        else:
            x_ = self.ln(x)

        x_ = self.fc2(self.dropout(self.act(self.fc1(x_))))
        x = x + self.dropout(x_)
        return x

class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads): # c = 256(v4s), num_heads = 4

        super().__init__()

        # query key value
        '''
              # y = ax1 + bx2 + cx3 + k
              # 모델을 선언 및 초기화. 다중 선형 회귀이므로 input_dim=3, output_dim=1.
              model = nn.Linear(3, 1, bias=True) ... (입력차원, 출력차원, bias)
        '''

        '''
        if c = 3인 경우
        y1   | w1 w2 w3 | | x1 |
        y2 = | w4 w5 w6 | | x2 |
        y3   | w7 w8 w9 | | x3 |
        '''
        self.q = nn.Linear(c, c, bias=False) # small (256, 256)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)

        # multi-head attention
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)

        # mlp
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x

# class TransformerBlock(nn.Module):
#     # Vision Transformer https://arxiv.org/abs/2010.11929
#     def __init__(self, c1, c2, num_heads, num_layers):
#         super().__init__()
#         self.conv = None
#
#         if c1 != c2:
#             self.conv = Conv(c1, c2)
#         self.linear = nn.Linear(c2, c2)  # learnable position embedding
#         self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
#         # self.tr = nn.Sequential(*[TransformerLayer_ViT(c2, num_heads) for _ in range(num_layers)])
#
#
#         self.c2 = c2
#
#     def forward(self, x):
#         # 입출력 채널 수가 다른 경우 초기 입력 값은 Conv 연산
#         if self.conv is not None:
#             x = self.conv(x)
#
#         # Flatten & Positional Embedding
#         # 입력 피쳐맵의 shape 언패킹
#         b, _, w, h = x.shape  # torch.Size([1, 128, 20, 20])
#
#         # 3차원 형태의 텐서로 flatten, *flatten(input, start_dim = 0, end_dim = -1)
#         p = x.flatten(2)      # torch.Size([1, 128, 400])
#         p = p.unsqueeze(0)    # torch.Size([1, 1, 128, 400])
#         p = p.transpose(0, 3) # torch.Size([400, 1, 512, 1])
#         p = p.squeeze(3)      # torch.Size([400, 1, 512])
#         e = self.linear(p)    # learnable position embedding
#         # rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size)
#         x = p + e
#
#         x = self.tr(x)
#         x = x.unsqueeze(3)
#         x = x.transpose(0, 3)
#         x = x.reshape(b, self.c2, w, h)
#         return x

class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None

        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        # self.tr = nn.Sequential(*[TransformerLayer_ViT_(c2, num_heads, use_HLN=False) for _ in range(num_layers)])
        # self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.tr = nn.Sequential(*[TransformerLayer_ViT(c2, num_heads) for _ in range(num_layers)])
        # self.tr = nn.Sequential(*[TransformerLayer_light(c2, num_heads) for _ in range(num_layers)])
        self.spp = None
        self.c2 = c2

        # # patch Embedding
        # self.pe = PatchEmbedding(in_channels=c1)
        #
        # self.patch_size = 20
        # self.emb_size = 20 * 20 * c1
        #
        # # Method 1: Flatten and FC layer
        # self.projection = nn.Sequential(
        #     Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=20, s2=20),
        #     nn.Linear(20 * 20 * c1, 20 * 20 * c1)
        # )
        #
        # # Method 2: Conv
        # self.projection = nn.Sequential(
        #     # using a conv layer instead of a linear one -> performance gains
        #     nn.Conv2d(c1, 20 * 20 * c1, 20, stride=20),
        #     Rearrange('b e (h) (w) -> b (h w) e')
        # )
        #
        # self.cls_token = nn.Parameter(torch.randn(1, 1, 20 * 20 * c1))
        # self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x):
        # 입출력 채널 수가 다른 경우 초기 입력 값은 Conv 연산
        if self.conv is not None:
            x = self.conv(x)

        # Flatten & Positional Embedding
        # 입력 피쳐맵의 shape 언패킹
        b, _, w, h = x.shape  # torch.Size([1, 256, 20, 20])

        # 3차원 형태의 텐서로 flatten, *flatten(input, start_dim = 0, end_dim = -1)
        p = x.flatten(2)      # torch.Size([1, 256, 400])
        p = p.unsqueeze(0)    # torch.Size([1, 1, 256, 400])
        p = p.transpose(0, 3) # torch.Size([400, 1, 256, 1])
        p = p.squeeze(3)      # torch.Size([400, 1, 256])
        e = self.linear(p)    # learnable position embedding
        # rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size)
        x = p + e

        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)

        # SPP
        if self.spp is not None:
            x = self.spp(x)
        return x

class Bottleneck(nn.Module):
    # Standard bottleneck: 채널수 조절, 깊은 신경망 구현 시 gradient vanishing 문제를 개선
    # c1 => Conv1(conv(1x1),bn,act) => c1 / 2 => Conv2(conv(3x3),bn,act) => c2
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)       # input: c1,    output: c1/2
        self.cv2 = Conv(c_, c2, 3, 1, g=g)  # input: c1/2,  output: c2
        self.add = shortcut and c1 == c2    # 모듈 입력채널과 출력채널 수가 같으며 shortcut 옵션이 true인 경우에만 short conn.

    def forward(self, x):
        # Short-Cut
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)                   # bn->act
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)  # no bn, act # input: c1,   output: c1/2
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)               # input: c1/2, output: c1/2
        self.cv4 = Conv(2 * c_, c2, 1, 1)                            # input: c1,   output: c2
        self.bn = nn.BatchNorm2d(2 * c_)                             # applied to cat(cv2, cv3)
        self.act = Mish()
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)]) # module unpacking
        print(n)

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))

class BottleneckCSP2(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP2, self).__init__()
        c_ = int(c2)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = Mish()
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)]) # module unpacking
        # self.m = TransformerBlock_SPPCSP(c_, c_, 4, n)

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class FCSP2(nn.Module):
    # Focus CSP2 wh information into c-space
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(FCSP2, self).__init__()
        c_ = int(c2)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = Mish()
        self.m = Focus(c_, c_)
        self.up = nn.Upsample(scale_factor=2, mode = 'nearest')

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.up(self.m(x1))
        y2 = self.cv2(x1)
        return self.cv3(self.act(self.bn(torch.cat((y1, y2), dim=1))))

# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
class TransformerBlock_SPPCSP(TransformerBlock):
    # Bottleneck module with TransformerBlock()
    def __init__(self, c1, c2, num_heads=4, num_layers=1):
        super().__init__(c1, c2, num_heads, num_layers)  # c1=512, c2=512, num_layer=3, False, 1, 0.5)
        self.spp = SPPCSP(c2, c2, k=(5, 9, 13))

class TR_SPPCSP_CSP(BottleneckCSP):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)  # c1=512, c2=512, num_layer=3, False, 1, 0.5)
        c_ = int(c2 * e)
        self.m = TransformerBlock_SPPCSP(c_, c_, 4, n)
        print(self.m)

class TR_SPPCSP_CSP2(BottleneckCSP2):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)  # c1=512, c2=512, num_layer=3, False, 1, 0.5)
        c_ = int(c2)
        self.m = TransformerBlock_SPPCSP(c_, c_, 4, n)
        print(self.m)

# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

class TR_CSP(BottleneckCSP):
    # Bottleneck module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e) # c1=512, c2=512, num_layer=3, False, 1, 0.5)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)
        print(self.m)



class BoTSPPCSP_CSP2(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BoTSPPCSP_CSP2, self).__init__()
        c_ = int(c2)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = Mish()
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)]) # module unpacking
        self.spp = SPPCSP(c_, c_, k=(5, 9, 13))

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.spp(self.m(x1))
        y2 = self.cv2(x1)
        return self.cv3(self.act(self.bn(torch.cat((y1, y2), dim=1))))

class SPPCSP_CSP(BottleneckCSP):
    # Bottleneck module with TransformerBlock()
    def __init__(self, c1, c2, n=1, k=(5, 9, 13), shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e) # c1=512, c2=512, num_layer=3, False, 1, 0.5)
        c_ = int(c2 * e)
        self.m = SPPCSP(c_, c_, k)

class SPPCSP_CSP2(BottleneckCSP2):
    # Bottleneck module with TransformerBlock()
    def __init__(self, c1, c2, n=1, k=(5, 9, 13), shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e) # c1=512, c2=512, num_layer=3, False, 1, 0.5)
        c_ = int(c2)
        self.m = SPPCSP(c_, c_, k)

class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        # 채널을 절반으로 축소
        self.cv1 = Conv(c1, c_, 1, 1)
        # 채널수가 인풋대비 4배가 된 것을 다시 c2로 변환
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        # DN-CNN SPP module 사용, yolov3-spp참고
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        # 3가지 output size = 19로 동일
    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

# 대충 spp를 csp구조로 사용한다는 의미인것 같음
class SPPCSP(nn.Module):
    # CSP SPP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSP, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1) # len(k) + 1
        self.cv6 = Conv(c_, c_, 3, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = Mish()
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(self.act(self.bn(torch.cat((y1, y2), dim=1))))

class SPPCSP2(nn.Module):
    # CSP SPP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSP2, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1) # len(k) + 1
        self.cv6 = Conv(c_, c_, 3, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = Mish()
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(self.act(self.bn(torch.cat((y1, y2), dim=1))))

# MaxPooling
class MP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Flatten(nn.Module):
    # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)  # to x(b,c2,1,1)
        self.flat = Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)
