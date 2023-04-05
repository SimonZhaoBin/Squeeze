import torch
import torch.nn as nn
import torch.nn.functional as F
use_cuda = torch.cuda.is_available()

filter_sizes = [32, 16, 8, 8, 16, 32]
kernel_sizes = [9, 5, 3, 3, 5, 9]
strides = [3, 2, 2, 2, 2, 3]

class Fire(nn.Module):

    def __init__(self, in_channel, out_channel, squzee_channel):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channel, squzee_channel, 1),
            nn.BatchNorm2d(squzee_channel),
            nn.ReLU(inplace=True)
        )

        self.expand_1x1 = nn.Sequential(
            nn.Conv2d(squzee_channel, int(out_channel / 2), 1),
            nn.BatchNorm2d(int(out_channel / 2)),
            nn.ReLU(inplace=True)
        )

        self.expand_3x3 = nn.Sequential(
            nn.Conv2d(squzee_channel, int(out_channel / 2), 3, padding=1),
            nn.BatchNorm2d(int(out_channel / 2)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.squeeze(x)
        x = torch.cat([
            self.expand_1x1(x),
            self.expand_3x3(x)
        ], 1)
        return x
class SQUEEZENET(nn.Module):

    def __init__(self, input_channels=1,class_num=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            # padding=(kernel_size-1)/2 if stride=1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.fire2 = Fire(48, 64, 8)
        self.fire3 = Fire(64, 64, 8)
        self.fire4 = Fire(64, 128, 32)
        self.fire5 = Fire(128, 128, 32)
        self.fire6 = Fire(128, 192, 24)
        self.fire7 = Fire(192, 192, 24)
        self.fire8 = Fire(192, 256, 32)
        self.fire9 = Fire(256, 256, 32)
        self.conv10 = nn.Conv2d(256, class_num, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.upsample = nn.Upsample(size=300, mode='nearest')  #torch.nn.Upsample(size=None, scale_factor=None, mode='nearest', align_corners=None)

        self.pos_output = nn.Conv2d(class_num, 1, kernel_size=1)
        self.cos_output = nn.Conv2d(class_num, 1, kernel_size=1)
        self.sin_output = nn.Conv2d(class_num, 1, kernel_size=1)
        self.width_output = nn.Conv2d(class_num, 1, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.fire2(x)
        x = self.fire3(x) + x
        x = self.fire4(x)
        x = self.maxpool(x)

        x = self.fire5(x) + x
        x = self.fire6(x)
        x = self.fire7(x) + x
        x = self.fire8(x)
        x = self.maxpool(x)

        x = self.fire9(x)
        x = self.conv10(x)
        x = self.upsample(x)
        #x = self.avg(x)

        #print("1zbtest output = ", x.shape)
        #x = x.view(x.size(0), -1)
        #print("4zhaobin=", pos_output.shape, cos_output.shape, sin_output.shape, width_output.shape)
        #print("2zbtest output = ",x.shape,len(x))
        #return x

        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)
        return pos_output, cos_output, sin_output, width_output

        #x = self.avg(x)
        #print("1zbtest output = ", x.shape)
        #x = x.view(x.size(0), -1)
        #print("4zhaobin=", pos_output.shape, cos_output.shape, sin_output.shape, width_output.shape)
        #print("2zbtest output = ",x.shape,len(x))
        #return x
#  这个函数赋值有点蒙，没太懂
    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin, y_width = yc
        #print("4zhaobin=",y_pos.shape,y_cos.shape, y_sin.shape, y_width.shape,len(yc),len(self(xc)))
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)
        # if len(xc)==8:
        #     pos_pred, cos_pred, sin_pred, width_pred,_,_,_,_ = self(xc)
        # elif len(xc)==2:
        #     pos_pred, cos_pred = self(xc)
        #     sin_pred, width_pred = self(xc)
        # else:
        #     pos_pred= self(xc)
        #     cos_pred = self(xc)
        #     sin_pred = self(xc)
        #     width_pred = self(xc)
        #print("5zhaobin=",pos_pred.shape,cos_pred.shape, sin_pred.shape, width_pred.shape)
        p_loss = F.mse_loss(pos_pred, y_pos)
        cos_loss = F.mse_loss(cos_pred, y_cos)
        sin_loss = F.mse_loss(sin_pred, y_sin)
        width_loss = F.mse_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }


#def squeezenet(class_num=300):
    #model = SqueezeNet(class_num=class_num)

    #return model



