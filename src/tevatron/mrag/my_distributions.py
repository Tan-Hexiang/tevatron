import torch


class BinaryConcrete(torch.distributions.relaxed_bernoulli.RelaxedBernoulli):
    def __init__(self, temperature, logits):
        super().__init__(temperature=temperature, logits=logits)
        self.device = self.temperature.device

    def cdf(self, value):
        return torch.sigmoid(
            (torch.log(value) - torch.log(1.0 - value)) * self.temperature - self.logits
        )

    def log_prob(self, value):
        return torch.where(
            (value > 0) & (value < 1),
            super().log_prob(value),
            torch.full_like(value, -float("inf")),
        )



class Streched(torch.distributions.TransformedDistribution):
    def __init__(self, base_dist, l=-0.1, r=1.1):
        self.l = l
        self.r = r
        super().__init__(
            base_dist, torch.distributions.AffineTransform(loc=l, scale=r - l)
        )

    def expected_L0(self):
        return 1-self.base_dist.cdf(torch.tensor(-self.l/(self.r-self.l)))


class RectifiedStreched(Streched):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample(self, sample_shape=torch.Size([])):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape=torch.Size([])):
        x = super().rsample(sample_shape)
        # 采样变到[0,1]之间
        return x.clamp(0, 1)
