"""Microbenchmark exact native color predicates and sparse-mask coordinates.

This does not measure simulation throughput or physical task success.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import cv2
import numpy as np
from experiments.torso_assembly import image_ops, native_pixels


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True);parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--iterations',type=int,default=200)
    args=parser.parse_args()
    if min(args.repeats,args.iterations)<1:parser.error('Positive repetition counts required')
    out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    cv2.setNumThreads(1);native_pixels.enable()
    image=np.random.default_rng(42).integers(0,256,(960,1280,3),dtype=np.uint8)
    mask=np.zeros(image.shape[:2],np.uint8)
    cv2.fillConvexPoly(mask,np.array([[200,200],[1000,220],[1000,260],[200,240]],np.int32),1)
    original_flag=image_ops._NATIVE
    try:
        image_ops._NATIVE=False
        color_reference=image_ops.wood_mask(image,70,1.02);coordinate_reference=image_ops.mask_pixels(mask)
        image_ops._NATIVE=True
        np.testing.assert_array_equal(color_reference,image_ops.wood_mask(image,70,1.02))
        for expected,actual in zip(coordinate_reference,image_ops.mask_pixels(mask)):
            np.testing.assert_array_equal(expected,actual)
        rows=[]
        for repeat in range(args.repeats):
            row={}
            for enabled in ([False,True] if repeat%2==0 else [True,False]):
                image_ops._NATIVE=enabled
                for label,function in [('color',lambda:image_ops.wood_mask(image,70,1.02)),
                                       ('coordinates',lambda:image_ops.mask_pixels(mask))]:
                    start=time.perf_counter()
                    for _ in range(args.iterations):function()
                    row[label+('_native_s' if enabled else '_baseline_s')]=time.perf_counter()-start
            rows.append(row)
    finally:image_ops._NATIVE=original_flag
    sources=[Path(__file__),Path(image_ops.__file__),Path(native_pixels.__file__),
             Path(native_pixels.__file__).with_name('rgb_mask.c')]
    result=dict(scope='Synthetic image-operation microbenchmark; excludes compilation and does not measure full simulation speed.',
        iterations=args.iterations,repeats=args.repeats,image_shape=list(image.shape),rng_seed=42,
        sparse_nonzero_pixels=int(np.count_nonzero(mask)),opencv_threads=1,
        exact_color_and_coordinate_check=True,native_runtime=native_pixels.receipt(),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},measurements=rows,
        median_speedup={label:statistics.median(r[label+'_baseline_s'] for r in rows)/
                              statistics.median(r[label+'_native_s'] for r in rows)
                        for label in ('color','coordinates')})
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
