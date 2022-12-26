import argparse
from copy import deepcopy

from models.experimental import *


class Detect(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.stride = None  # strides computed during build
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        # 각 Conv layer를 list로 전달하고 layer의 iterator를 만든다. forward 연산처리를 간단하게 할 수 있음
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        # Modulelist = [Conv1[256,255], Conv2[512,255], Conv3[1024,255]
        # in = [256, 512, 1024], out =[255]
        self.export = False  # onnx export

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export  # self.training은 nn.Module 클래스의 생성자 정의시 True로 설정됨

        # detection layer(nl=3) 수만큼 반복
        for i in range(self.nl):
            # Conv layer로 이루어진 ModuleList m[i]에 input x[i] 입력
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        # cfg 형식의 경우
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        # yaml 형식의 경우
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name

            # 입력된 모델 구조를 불러옴
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        # Define model
        if nc and nc != self.yaml['nc']:
            print('Overriding %s nc=%g with nc=%g' % (cfg, self.yaml['nc'], nc))
            self.yaml['nc'] = nc  # override yaml value
        # parse_model로 yaml을 읽어서 모델과 저장목록을 저장
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist, ch_out
        # copy.deepcopy: 내부에 객체들까지 모두 새롭게 copy.
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        # 모델의 마지막 레이어는 Detect 모듈
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 640  # 2x min stride
            m.stride = torch.tensor(
                [s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward(x) x.shape: (1,3,128,128)
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        torch_utils.initialize_weights(self)
        self.info()

    def forward(self, x, augment=False, profile=False):
        # multi-scale
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [0.83, 0.67]  # scales
            y = []

            for i, xi in enumerate((x,
                                    torch_utils.scale_img(x.flip(3), s[0]),  # flip-lr and scale
                                    torch_utils.scale_img(x, s[1]),  # scale
                                    )):
                # cv2.imwrite('img%g.jpg' % i, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])
                y.append(self.forward_once(xi)[0])

            y[1][..., :4] /= s[0]  # scale
            y[1][..., 0] = img_size[1] - y[1][..., 0]  # flip lr
            y[2][..., :4] /= s[1]  # scale
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            # m.i, m.f, m.t, m.np == module index, 'from' index, type, number params
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                try:
                    import thop
                    o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # FLOPS
                except:
                    o = 0
                t = torch_utils.time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((torch_utils.time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run, for 문을 통해 선택된 module
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  #  from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  #  from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ', end='')
        for m in self.model.modules():
            if type(m) is Conv:
                m.conv = torch_utils.fuse_conv_and_bn(m.conv, m.bn)  # update conv
                m.bn = None  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def info(self):  # print model information
        torch_utils.model_info(self)


def parse_model(d, ch):  # model_dict, input_channels(3)
    print('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings, ex) m = 'Conv', eval(m): class 'models.common.Conv'

        # args에 string 값이 들어가 있는 경우를 판단하기 위함
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        # gd: depth_multiple에 따라 모델의 깊이가 결정됨
        n = max(round(n * gd), 1) if n > 1 else n  # depth gain

        if m in [nn.Conv2d, Conv, Bottleneck, SPP, BottleneckCSP, BottleneckCSP2,
                 TR_CSP, SPPCSP, SPPCSP_CSP, BoTSPPCSP_CSP2, TR_SPPCSP_CSP, TR_SPPCSP_CSP2, FCSP2]:
            # channel c1: input, c2: output
            c1, c2 = ch[f], args[0]

            # Normal
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1.75  # exponential (default 2.0)
            #     e = math.log(c2 / ch[1]) / math.log(2)
            #     c2 = int(ch[1] * ex ** e)
            # if m != Focus:

            # output channel 값이 최종출력 3*(80+5) = 255 와 다르다면 8로 나누고 다시 곱하는 과정을 수행(왜하는거지), 255이면 나누지않고 그대로 저장, 나누게되면 값이 달라짐
            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            # Experimental
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1 + gw  # exponential (default 2.0)
            #     ch1 = 32  # ch[1]
            #     e = math.log(c2 / ch1) / math.log(2)  # level 1-n
            #     c2 = int(ch1 * ex ** e)
            # if m != Focus:
            #     c2 = make_divisible(c2, 8) if c2 != no else c2
            # 즉, Width_Multiple 값이 증가할 수록 해당 레이어의 conv 필터 수가 증가
            # 변경된 c2값이 out_channel값으로 들어간다.

            args = [c1, c2, *args[1:]]  # args = [input, output, kernel_size, stride]

            if m in [BottleneckCSP, BottleneckCSP2, TR_CSP, SPPCSP, SPPCSP_CSP, SPPCSP_CSP2,
                     BoTSPPCSP_CSP2, TR_SPPCSP_CSP, TR_SPPCSP_CSP2, FCSP2]:
                args.insert(2, n)  # list의 2번째 위치에 n값(target_layer)을 삽입
                #n = 1

        elif m is nn.BatchNorm2d:
            args = [ch[f]]

        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])

        elif m is Detect:
            args.append([ch[x + 1] for x in f])

            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]

        # 모델의 깊이(n)에 따라 nn.Sequential로 층을 쌓을지 아니면 단일 모듈형태로 저장할지 결정, (*args)로 초기화
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module unpacking -> m_

        # module명을 추출해서 t라는 변수에 저장
        t = str(m)[8:-2].replace('__main__.', '')  # module type

        # tensor.numel(): 텐서의 원소 갯수
        # model.parameters(): 모델의 파라미터 정보 load
        np = sum([x.numel() for x in m_.parameters()])  # number params: 언패킹된 각 모듈의 파라미터 수의 총 합을 계산
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params

        print('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print

        # from layer가 -1이 아닌 경우
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        # extend(): list 끝에 iterable의 각 항목을 원소로 추가함
        layers.append(m_)
        ch.append(c2)

    # layers unpacking, save element sorting
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='L-yolov4s-TR1-SPPCSP-CSP2+BoTSPPCSP-CSP2.yaml', help='model.yaml')
    # parser.add_argument('--cfg', type=str, default='yolov4s-TR3.yaml', help='model.yaml')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    device = torch_utils.select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    # y = model(img, profile=True)

    # ONNX export
    # model.model[-1].export = True
    # torch.onnx.export(model, img, opt.cfg.replace('.yaml', '.onnx'), verbose=True, opset_version=11)

    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
