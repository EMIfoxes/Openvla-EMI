import h5py
file = '/media/lxx/Elements/project/openvla-oft/datasets/libero_goal_no_noops/open_the_middle_drawer_of_the_cabinet_demo.hdf5'
# f = h5py.File(file, 'r')

#!/usr/bin/env python3
"""
Universal HDF5 inspector — no GUI, no extra deps except h5py.
Usage:
    python h5ls.py file.h5 [-d max_depth] [-p path]
"""
import sys, argparse, h5py

def _desc(obj):
    """返回一行人类可读的摘要"""
    if isinstance(obj, h5py.Dataset):
        extra = []
        if obj.chunks:           extra.append(f"chunks={obj.chunks}")
        if obj.compression:      extra.append(f"compression={obj.compression}")
        if obj.scaleoffset:      extra.append(f"scaleoffset={obj.scaleoffset}")
        if obj.shuffle:          extra.append("shuffle")
        if obj.fletcher32:       extra.append("fletcher32")
        if obj.fillvalue is not None and obj.fillvalue != 0:
            extra.append(f"fill={obj.fillvalue}")
        return (f"Dataset {obj.shape} {obj.dtype}"
                + (f"  [{', '.join(extra)}]" if extra else ""))
    if isinstance(obj, h5py.Group):
        return f"Group ({len(obj)} members)"
    if isinstance(obj, h5py.Datatype):
        return f"Named datatype {obj.dtype}"
    return str(type(obj).__name__)

def _walk(name, obj, max_depth, prefix="", depth=0):
    """递归打印树状结构"""
    if max_depth is not None and depth > max_depth:
        return
    indent = "│  " * depth
    connector = "└─ " if depth else ""
    link_info = ""
    if isinstance(obj, h5py.SoftLink):
        link_info = f" -> {obj.path}  (softlink)"
    elif isinstance(obj, h5py.ExternalLink):
        link_info = f" -> {obj.filename}::{obj.path}  (external)"
    print(f"{indent}{connector}{name}  {_desc(obj)}{link_info}")

def main():
    ap = argparse.ArgumentParser(description="Generic HDF5 file inspector")
    ap.add_argument("--file",  default="/media/lxx/Elements/project/openvla-oft/datasets/libero_goal_no_noops/open_the_middle_drawer_of_the_cabinet_demo.hdf5", help="HDF5 file to inspect")
    ap.add_argument("--depth", default=1, type=int, help="max depth to descend")
    ap.add_argument("--path",  default="/", help="start path inside file")
    args = ap.parse_args()

    try:
        with h5py.File(args.file, "r") as f:
            start = f[args.path]
            # 如果给定的是根，则遍历整个文件；否则只打印该节点及其子树
            if args.path == "/":
                start.visititems(lambda n, o: _walk(n, o, args.depth))
            else:
                _walk(args.path.strip("/"), start, args.depth, depth=0)
    except FileNotFoundError:
        sys.exit(f"文件不存在: {args.file}")
    except KeyError as e:
        sys.exit(f"路径不存在: {e}")

if __name__ == "__main__":
    main()