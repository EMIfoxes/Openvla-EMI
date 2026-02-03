import h5py
import matplotlib.pyplot as plt
file = '/media/lxx/Elements/project/openvla-oft/datasets/libero_goal_no_noops/open_the_middle_drawer_of_the_cabinet_demo.hdf5'
f = h5py.File(file, 'r')      # 只读模式
# print(list(f['data/demo_0/obs/'].keys()))        # 快速看有哪些 dataset
idx = 0
data = f['/action']   # 拿到 dataset 对象
# data = f['action'][idx]    # 拿到 dataset 对象
print(data)


# print(img.shape, img.dtype)      # 看看 shape 和数据类型
# print(type(img))                       # 直接打印数据（numpy 数组）
# plt.imshow(img)
# plt.axis('off')
# plt.title(f'frame {idx}')
# plt.show()


