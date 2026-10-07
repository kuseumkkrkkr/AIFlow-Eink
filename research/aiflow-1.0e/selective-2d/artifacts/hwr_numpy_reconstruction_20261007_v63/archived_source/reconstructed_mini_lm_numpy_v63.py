"""New NumPy inference reconstruction; not the lost V42-V62 implementation.

Reads only the narrow float32 tensor format in the hash-verified local archive.
Unknown pickle globals and storage types are rejected. No PyTorch dependency.
"""
import collections
import io
import pickle
import zipfile

import numpy as np
from scipy.special import erf, softmax


def load_checkpoint(path):
    with zipfile.ZipFile(path) as archive:
        root = next(n[:-8] for n in archive.namelist() if n.endswith('data.pkl'))
        if archive.read(root + 'byteorder') != b'little':
            raise ValueError('only little-endian checkpoints supported')

        def rebuild(storage, offset, shape, stride, requires_grad, hooks):
            if requires_grad or offset < 0 or any(x < 0 for x in shape + stride):
                raise ValueError('unsupported tensor metadata')
            end = offset + sum((n - 1) * s for n, s in zip(shape, stride)) + 1
            if end > len(storage):
                raise ValueError('tensor exceeds storage')
            return np.ndarray(shape, dtype='<f4', buffer=storage,
                              offset=offset * 4, strides=tuple(s * 4 for s in stride)).copy()

        class Reader(pickle.Unpickler):
            def find_class(self, module, name):
                allowed = {('torch._utils', '_rebuild_tensor_v2'): rebuild,
                           ('torch', 'FloatStorage'): 'float32',
                           ('collections', 'OrderedDict'): collections.OrderedDict}
                if (module, name) not in allowed:
                    raise ValueError(f'unsupported checkpoint global: {module}.{name}')
                return allowed[module, name]

            def persistent_load(self, identity):
                kind, dtype, key, location, count = identity
                if kind != 'storage' or dtype != 'float32' or not str(key).isdigit():
                    raise ValueError('unsupported storage')
                data = archive.read(root + 'data/' + str(key))
                if len(data) != count * 4:
                    raise ValueError('storage size mismatch')
                return np.frombuffer(data, dtype='<f4')

        return Reader(io.BytesIO(archive.read(root + 'data.pkl'))).load()


class NumpyLM:
    def __init__(self, payload):
        self.payload = payload
        self.w = payload['state_dict']
        self.labels = payload['labels']
        self.relations = payload['relations']
        self.pad = len(self.labels) + len(self.relations)
        self.cls, self.sep, self.mask = self.pad + 1, self.pad + 2, self.pad + 3

    def linear(self, x, key):
        return x @ self.w[key + '.weight'].T + self.w[key + '.bias']

    def norm(self, x, key):
        centered = x - x.mean(axis=-1, keepdims=True)
        return centered / np.sqrt((centered ** 2).mean(axis=-1, keepdims=True) + 1e-5) * self.w[key + '.weight'] + self.w[key + '.bias']

    def forward(self, ids, target, target_only=False):
        ids = np.asarray(ids, dtype=np.int64)
        if ids.ndim != 1 or len(ids) > self.payload['max_positions'] or not 0 <= target < len(ids):
            raise ValueError('invalid request shape')
        if np.any(ids < 0) or np.any(ids >= len(self.w['token_embedding.weight'])) or ids[target] != self.mask:
            raise ValueError('invalid request vocabulary or target')
        if target_only and self.payload['layers'] != 1:
            raise ValueError('target-only path requires exactly one layer')
        x = self.w['token_embedding.weight'][ids] + self.w['position_embedding.weight'][:len(ids)]
        for index in range(self.payload['layers']):
            key = f'encoder.layers.{index}'
            n = self.norm(x, key + '.norm1')
            qkv = n @ self.w[key + '.self_attn.in_proj_weight'].T + self.w[key + '.self_attn.in_proj_bias']
            q, k, v = np.split(qkv, 3, axis=-1)
            heads = self.payload['heads']
            dim = q.shape[-1] // heads
            q = q.reshape(len(ids), heads, dim).transpose(1, 0, 2)
            k = k.reshape(len(ids), heads, dim).transpose(1, 0, 2)
            v = v.reshape(len(ids), heads, dim).transpose(1, 0, 2)
            if target_only:
                q = q[:, target:target+1]
                x = x[target:target+1]
            scores = q @ k.transpose(0, 2, 1) * np.float32(dim ** -0.5)
            scores[:, :, ids == self.pad] = -1e4
            a = (softmax(scores, axis=-1) @ v).transpose(1, 0, 2).reshape(len(x), -1)
            x = x + self.linear(a, key + '.self_attn.out_proj')
            f = self.linear(self.norm(x, key + '.norm2'), key + '.linear1')
            f = f * np.float32(.5) * (1 + erf(f / np.float32(2 ** .5)))
            x = x + self.linear(f, key + '.linear2')
        return self.linear(self.norm(x[0 if target_only else target], 'output_norm'), 'classifier')
