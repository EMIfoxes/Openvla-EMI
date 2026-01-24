import h5py
import matplotlib.pyplot as plt
file = '/media/lxx/Elements/project/openvla-oft/datasets/libero_goal_no_noops/open_the_middle_drawer_of_the_cabinet_demo.hdf5'
f = h5py.File(file, 'r')      # 只读模式
# print(list(f['data/demo_0/obs/'].keys()))        # 快速看有哪些 dataset
idx = 137

ee_ori = f['data/demo_0/obs/ee_ori'][idx]    
ee_pos = f['data/demo_0/obs/ee_pos'][idx]    
ee_states = f['data/demo_0/obs/ee_states'][idx]    
gripper_states = f['data/demo_0/obs/gripper_states'][idx]
joint_states = f['data/demo_9/obs/joint_states'][idx]
rewards = f['data/demo_9/rewards'][idx]
actions = f['data/demo_9/actions'][idx]
dones = f['data/demo_9/dones'][idx]
robot_states = f['data/demo_9/robot_states'][idx]
states = f['data/demo_9/states'][idx]

# list 保留两位小数
print('ee_pos =', [round(x, 2) for x in ee_pos])  
print('ee_ori =', [round(x, 2) for x in ee_ori])     
print('ee_states =', [round(x, 2) for x in ee_states])     
print('gripper_states =', [round(x, 2) for x in gripper_states])     
print('joint_states =', [round(x, 2) for x in joint_states])
print('rewards =', rewards)
print('actions =', [round(x, 2) for x in actions])
print('dones =', dones)
print('robot_states =', [round(x, 2) for x in robot_states])
print('states =', [round(x, 2) for x in states])