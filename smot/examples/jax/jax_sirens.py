import math
from typing import Any, Callable, Optional, Sequence, Type, Union

from flax import linen as nn
import jax
from jax.core import NamedShape
import jax.numpy as jnp


class SirenLayer(nn.Module):
    dim: int = 128
    w0: float = 1.0
    c: float = 6.0
    is_first: bool = False
    use_bias: bool = True
    activation: Callable = jnp.sin

    def _init_weights(
        self,
        key: Any,
        shape: Union[Sequence[int], NamedShape],
        dtype: Type,
    ) -> jnp.ndarray:
        if self.is_first:
            w_std = 1 / self.dim
        else:
            w_std = math.sqrt(self.c / self.dim) / self.w0
        return jax.random.uniform(
            key=key,
            shape=shape,
            dtype=dtype,
            minval=-w_std,
            maxval=w_std,
        )

    @nn.compact
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        x = nn.Dense(
            features=self.dim,
            use_bias=self.use_bias,
            kernel_init=self._init_weights,
            bias_init=self._init_weights,
        )(*args)
        return self.activation(x)


class SirenNet(nn.Module):
    num_channels: int = 3
    d_hidden: int = 64
    depth: int = 8
    w0: float = 1.0
    w0_initial: float = 30.0
    use_bias: bool = True
    activation: Callable = jnp.sin
    final_activation: Optional[Callable] = jax.nn.sigmoid

    @nn.compact
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        for i in range(self.depth):
            is_first = i == 0
            if is_first:
                w0 = self.w0_initial
            else:
                w0 = self.w0
            x = SirenLayer(
                dim=self.d_hidden,
                is_first=is_first,
                w0=w0,
                use_bias=self.use_bias,
                activation=self.activation,
            )(*args)
        if self.final_activation is None:
            final_activation = lambda x: x
        else:
            final_activation = self.final_activation
        return SirenLayer(
            dim=self.num_channels,
            is_first=False,
            w0=w0,
            use_bias=self.use_bias,
            activation=final_activation,
        )(x)
