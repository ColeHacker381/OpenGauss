import numpy as np

class Transform:

    def __init__(self):
        pass

    def translate(self, image, tx, ty):
        output = np.zeros_like(image)
        h, w = image.shape

        src_y1, src_y2 = max(0, -ty), min(h, h - ty)
        src_x1, src_x2 = max(0, -tx), min(w, w - tx)

        dst_y1, dst_y2 = max(0, ty), min(h, h + ty)
        dst_x1, dst_x2 = max(0, tx), min(w, w + tx)

        output[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]

        return output

    def inverse_map(self, target_shape, source_center, target_center, scale):
        height, width = target_shape
        csx, csy = source_center
        ctx, cty = target_center

        xt = np.arange(width)
        yt = np.arange(height)

        xs = (xt - ctx) / scale + csx
        ys =  (yt - cty) / scale + csy

        return xs, ys

    def create_validity_mask(self, xs, ys, source_shape):
        height, width = source_shape
        vx = (xs >= 0) & (xs <= width - 1)
        vy = (ys >= 0) & (ys <= height - 1)
        vmask = vy[:, None] & vx[None, :]

        return vmask

    def bilinear_sample(self, image, xs, ys):
        h, w = image.shape

        ys = np.clip(ys, 0, h - 1)
        xs = np.clip(xs, 0, w - 1)

        x0 = np.floor(xs).astype(int)
        x1 = np.minimum(x0 + 1, w - 1)
        y0 = np.floor(ys).astype(int)
        y1 = np.minimum(y0 + 1, h - 1)

        wy = (ys - y0)[:, None]
        wx = (xs - x0)[None, :]

        top = (
            (1 - wx) * image[y0[:, None], x0[None, :]]
            + wx * image[y0[:, None], x1[None, :]]
        )

        bottom = (
            (1 - wx) * image[y1[:, None], x0[None, :]]
            + wx * image[y1[:, None], x1[None, :]]
        )

        return (1 - wy) * top + wy * bottom


    def align(self, source, target, source_center, target_center, scale):

        xs, ys = self.inverse_map(target.shape, source_center, target_center, scale)
        vmask = self.create_validity_mask(xs, ys, source.shape)
        sample_source = self.bilinear_sample(source, xs, ys)

        aligned_source = target.copy()
        aligned_source[vmask] = sample_source[vmask]
        return aligned_source, vmask

        

