"""
Elementary cellular automata simulator in numpy.
"""


import itertools
import pathlib
import time
from typing import Literal

import einops
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import tqdm


def main(
    rule: int = 110,
    width: int = 32,
    height: int = 32,
    init: Literal["random", "middle"] = "middle",
    seed: int = 42,
    animate: bool = True,
    fps: None | float = None,
    save_image: None | pathlib.Path = None,
    upscale: int = 1,
):
    print(f"rule: {rule}")
    print(f"bits: {rule:08b}")
    print("Wolfram table:")
    print(" 1 1 1   1 1 0   1 0 1   1 0 0   0 1 1   0 1 0   0 0 1   0 0 0")
    print("   " + "       ".join(f'{rule:08b}'))

    print("initialising state...")
    match init:
        case "middle":
            state = jnp.zeros(width, dtype=np.uint8)
            state = state.at[width//2].set(1)
        case "random":
            key = jax.random.PRNGKey(seed)
            key, _key = jax.random.split(key)
            state = jnp.random.randint(
                minval=0,
                maxval=2,  # not included
                shape=(width,),
                dtype=np.uint8,
                key=_key,
            )
    print("initial state:", state)

    print("simulating automaton...")
    start_time = time.perf_counter()
    histories = jax.jit(
        jax.vmap(
            simulate,
            in_axes=(0,None,None),
            out_axes=0, # the default
        ),
        static_argnames=('height',),
    )(
        jnp.arange(256),
        state,
        height,
    )
    end_time = time.perf_counter()
    print("simulation complete!")
    # print("result shape", history.shape)
    print(f"time taken {end_time - start_time:.4f} seconds")

    if animate:
        print("rendering...")
        for i, history in enumerate(histories):
            print("rule", i)
            for row in history:
                print(''.join(["█░"[s]*2 for s in row]))
                if fps is not None: time.sleep(1/fps)
            if fps is not None: time.sleep(1/fps)

    if save_image is not None:
        print("rendering to", save_image, "...")
        histories_arranged = einops.rearrange(
            histories,
            '(r1 r2) h w -> (r1 h) (r2 w)',
            r1=16,
            r2=16,
        )
        histories_greyscale = 255 * (1-histories_arranged)
        histories_upscaled = (histories_greyscale
            .repeat(upscale, axis=0)
            .repeat(upscale, axis=1)
        )
        Image.fromarray(np.asarray(histories_upscaled)).save(save_image)


def simulate(
    rule: int,
    init_state: jax.Array,    # uint8[width]
    height: int,
) -> jax.Array:                 # uint8[height, width]
    # parse rule
    rule_uint8 = jnp.uint8(rule)
    rule_bits = jnp.unpackbits(rule_uint8, bitorder='little')
    rule_table = rule_bits.reshape(2, 2, 2)

    init_state = jnp.pad(init_state, 1, mode='wrap') 

    def step(state, _):
        next_state = jnp.pad(rule_table[
            state[0:-2],
            state[1:-1],
            state[2:],
        ], 1, mode='wrap')

        return next_state, next_state

    _, history_except_first = jax.lax.scan(
        step,
        init_state,
        jnp.zeros(height-1),
    )
    history = jnp.concatenate(
        [init_state[jnp.newaxis, :], history_except_first],
        axis=0,
    )

    # return a view of the array without the width padding
    history = jnp.stack(history)

    return history[:, 1:-1]


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
