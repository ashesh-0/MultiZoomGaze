from static_model import GazeStaticModel


class GazeStaticWarpModel(GazeStaticModel):
    def forward(self, input):
        batch, seq, C, h, w = input.shape
        assert C == 3
        input = input.view(batch * seq, C, h, w)
        angular_output, var = super().forward(input)
        angular_output = angular_output.view(batch, seq, -1)
        var = var.view(batch, seq, -1)
        return angular_output, var
