import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['ConvBNReluBlock', 'GenericRSUBlock', 'GenericRSUFBlock']


def _upsample_like(src, dst):
    return F.upsample(src, size=dst.shape[2:], mode='bilinear')


class ConvBNReluBlock(nn.Module):
    r"""
    Conv, BatchNorm, ReLU operations encapsulated in a block.
    """
    def __init__(self, input_ch, output_ch, dirate):
        super(ConvBNReluBlock, self).__init__()
        self.conv = nn.Conv2d(input_ch, output_ch, kernel_size=3, padding=1*dirate, dilation=1 * dirate)
        self.bn = nn.BatchNorm2d(output_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, image):
        return self.relu(self.bn(self.conv(image)))


class GenericRSUBlock(nn.Module):
    r"""
    Generic RSU Block, parametrized by `L`, the parameter defines depth.
    RSU block as proposed in <TODO: add link>, which has a U-net like architecture
    """
    def __init__(self, input_ch=3, mid_ch=12, output_ch=3, L=None):
        super(GenericRSUBlock, self).__init__()

        self._L = L

        # encoder
        encoder = [ConvBNReluBlock(input_ch=input_ch, output_ch=output_ch, dirate=1)]

        for i in range(L-1):
            dirate = 1  # if i < L-1 else 2
            inp = output_ch if i == 0 else mid_ch
            encoder.append(ConvBNReluBlock(input_ch=inp, output_ch=mid_ch, dirate=dirate))
            if i < L-2:
                encoder.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))

        # TODO: go without defining last layer - use nn.Sequential but carefully and slice
        self.last_layer_enc = ConvBNReluBlock(input_ch=mid_ch, output_ch=mid_ch, dirate=2)

        # decoder
        decoder = []
        for i in range(L - 2):  # decoder has -1 channel, -1 for the last channel out
            decoder.append(ConvBNReluBlock(input_ch=mid_ch*2, output_ch=mid_ch, dirate=1))  # TODO: verify mid_ch * 2

        decoder.append(ConvBNReluBlock(input_ch=mid_ch*2, output_ch=output_ch, dirate=1))  # TODO: verify mid_ch * 2

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def __repr__(self):
        return "Block RSU: " + str(self._L)

    def forward(self, x):
        outputs = []
        downward = x

        # TODO: Replace downward with downstream
        # encoder
        for layer in self.encoder:
            downward = layer(downward)
            if isinstance(layer, ConvBNReluBlock):
                outputs.append(downward)

        hx_in = outputs.pop(0).clone()
        assert len(self.decoder) == len(outputs)
        # decoder
        upward = self.last_layer_enc.forward(downward)
        for layer in self.decoder[:-1]:
            upward = layer(torch.cat((upward, outputs.pop()), 1))
            upward = _upsample_like(upward, outputs[-1])

        return hx_in + self.decoder[-1](torch.cat((upward, outputs.pop()), 1))


class GenericRSUFBlock(nn.Module):
    r"""
    Generic RSUF block. A modification of the original RSU block, with dilated convolutions.
    The parameter `dirate` controls the sparsity/density of the dilatations.
    """
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, L=4):
        if L < 3:
            raise ValueError("Depth (L) of GenericRSUFBlock must be >=3.")

        super(GenericRSUFBlock, self).__init__()
        encoder = [ConvBNReluBlock(input_ch=in_ch, output_ch=out_ch, dirate=1)]
        decoder = []

        for i in range(L):
            encoder.append(
                ConvBNReluBlock(input_ch=out_ch if i == 0 else mid_ch, output_ch=mid_ch, dirate=2 ** i)
            )

        for i in range(L-2):
            decoder.append(
                ConvBNReluBlock(input_ch=mid_ch * 2, output_ch=mid_ch, dirate=2 ** (L-2-i))
            )

        i += 1
        decoder.append(ConvBNReluBlock(input_ch=mid_ch * 2, output_ch=out_ch, dirate=2 ** (L-2-i)))
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        outputs = []
        downward = x
        for layer in self.encoder:
            downward = layer(downward)
            outputs.append(downward)

        hx_in = outputs.pop(0).clone()
        upward = outputs.pop()
        for layer in self.decoder:
            upward = layer(torch.cat((outputs.pop(), upward), 1))

        return hx_in + upward


if __name__ == '__main__':
    print()
    # Test blocks
    c1 = GenericRSUBlock(3, 32, 64, L=4)
    c1f = GenericRSUFBlock(3, 32, 64, L=4)
    c1f.forward(torch.Tensor(torch.rand((1, 3, 256, 256))))
    c1f.forward(torch.Tensor(torch.rand((1, 3, 256, 256))))
