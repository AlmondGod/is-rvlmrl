/* Exact uint8 color predicates; no floating point, image resizing or morphology. */
#include <stdint.h>
#include <stddef.h>

void wood_mask(const uint8_t *rgb, uint8_t *out, size_t n,
               int threshold, int rg_ratio) {
    for (size_t i = 0; i < n; ++i) {
        int r = rgb[3*i], g = rgb[3*i+1], b = rgb[3*i+2];
        out[i] = (r > threshold) & (100*r > rg_ratio*g) & (100*g > 118*b);
    }
}
