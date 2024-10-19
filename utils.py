import matplotlib.pyplot as plt
import numpy as np
import torch
import imageio
import io

def random_rotation_matrix():
    # Random quaternion
    q = torch.randn(4)
    q = q / torch.norm(q)
    
    # Quaternion to rotation matrix
    R = torch.tensor([
        [1 - 2*q[2]**2 - 2*q[3]**2, 2*q[1]*q[2] - 2*q[3]*q[0], 2*q[1]*q[3] + 2*q[2]*q[0]],
        [2*q[1]*q[2] + 2*q[3]*q[0], 1 - 2*q[1]**2 - 2*q[3]**2, 2*q[2]*q[3] - 2*q[1]*q[0]],
        [2*q[1]*q[3] - 2*q[2]*q[0], 2*q[2]*q[3] + 2*q[1]*q[0], 1 - 2*q[1]**2 - 2*q[2]**2]
    ])
    return R

def augment_data(data):
    B, T, M = data.shape
    augmented_data = torch.zeros_like(data)
    
    for i in range(B):
        for c in range(0, M, 6):
            R = random_rotation_matrix().cuda()
            acc = data[i, :, c:c+3].transpose(0, 1)  # Shape (3, T)
            gyro = data[i, :, c+3:c+6].transpose(0, 1)  # Shape (3, T)
            
            # Apply rotation
            rotated_acc = torch.matmul(R, acc)
            rotated_gyro = torch.matmul(R, gyro)
            
            # Concatenate and assign to augmented_data
            augmented_data[i, :, c:c+3] = rotated_acc.transpose(0, 1)
            augmented_data[i, :, c+3:c+6] = rotated_gyro.transpose(0, 1)
    
    return augmented_data

def update_limits(data):
    # Get global min and max for each axis
    min_x, max_x = np.min(data[:, :, 0]), np.max(data[:, :, 0])
    min_y, max_y = np.min(data[:, :, 2]), np.max(data[:, :, 2])
    min_z, max_z = np.min(data[:, :, 1]), np.max(data[:, :, 1])

    # Add some padding to ensure the skeleton doesn't touch the plot edges
    padding = 0.1
    x_range = max_x - min_x
    y_range = max_y - min_y
    z_range = max_z - min_z

    return (min_x - padding * x_range, max_x + padding * x_range), \
           (min_y - padding * y_range, max_y + padding * y_range), \
           (min_z - padding * z_range, max_z + padding * z_range)

def plot_skeleton(frame_data, xlims, ylims, zlims, dataset):
    """
    Plot a single frame of skeleton data.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(frame_data[:, 0], frame_data[:, 2], frame_data[:, 1])

    # Add code here to connect the joints as per your skeleton structure
    if dataset == 't2m':
        connections = [
            [0, 2, 5, 8, 11],
            [0, 1, 4, 7, 10],
            [0, 3, 6, 9, 12, 15],
            [9, 14, 17, 19, 21],
            [9, 13, 16, 18, 20]
        ]

    if dataset == 'kit':
        connections = [
            [0, 11, 12, 13, 14, 15], 
            [0, 16, 17, 18, 19, 20], 
            [0, 1, 2, 3, 4], 
            [3, 5, 6, 7], 
            [3, 8, 9, 10]
        ]
    
    if dataset == 'ntu':
        connections = [
            [0, 12, 13, 14, 15],
            [0, 16, 17, 18, 19],
            [0, 1, 20, 2, 3],
            [20, 4, 5, 6, 7, 21],
            [7, 22],
            [20, 8, 9, 10, 11, 23],
            [11, 24],
        ]

    # Plot the lines for each sequence
    for connection in connections:
        for i in range(len(connection)-1):
            start_joint = connection[i]
            end_joint = connection[i+1]
            ax.plot([frame_data[start_joint, 0], frame_data[end_joint, 0]],
                    [frame_data[start_joint, 2], frame_data[end_joint, 2]],
                    [frame_data[start_joint, 1], frame_data[end_joint, 1]])

    ax.view_init(elev=10, azim=90)
    ax.set_box_aspect((np.ptp(xlims), np.ptp(ylims), np.ptp(zlims))) 
    
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_zlim(zlims)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = imageio.imread(buf)
    buf.close()

    plt.close(fig)  # Close the figure to prevent display
    return img

def plot_skeleton_gif(data, dataset):
    xlims, ylims, zlims = update_limits(data)
    images = [plot_skeleton(frame, xlims, ylims, zlims, dataset) for frame in data]
    imageio.mimsave('./skeleton_animation.gif', images, fps=20)
    return

def plot_single_skeleton(data, dataset, frame=0):

    xlims, ylims, zlims = update_limits(data)
    frame_data = data[frame]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(frame_data[:, 0], frame_data[:, 2], frame_data[:, 1])

    # Add code here to connect the joints as per your skeleton structure
    if dataset == 't2m':
        connections = [
            [0, 2, 5, 8, 11],
            [0, 1, 4, 7, 10],
            [0, 3, 6, 9, 12, 15],
            [9, 14, 17, 19, 21],
            [9, 13, 16, 18, 20]
        ]

    if dataset == 'kit':
        connections = [
            [0, 11, 12, 13, 14, 15], 
            [0, 16, 17, 18, 19, 20], 
            [0, 1, 2, 3, 4], 
            [3, 5, 6, 7], 
            [3, 8, 9, 10]
        ]
    
    if dataset == 'ntu':
        connections = [
            [0, 12, 13, 14, 15],
            [0, 16, 17, 18, 19],
            [0, 1, 20, 2, 3],
            [20, 4, 5, 6, 7, 21],
            [7, 22],
            [20, 8, 9, 10, 11, 23],
            [11, 24],
        ]

    # Plot the lines for each sequence
    for connection in connections:
        for i in range(len(connection)-1):
            start_joint = connection[i]
            end_joint = connection[i+1]
            ax.plot([frame_data[start_joint, 0], frame_data[end_joint, 0]],
                    [frame_data[start_joint, 2], frame_data[end_joint, 2]],
                    [frame_data[start_joint, 1], frame_data[end_joint, 1]])

    #ax.view_init(elev=10, azim=90)
    ax.set_box_aspect((np.ptp(xlims), np.ptp(ylims), np.ptp(zlims))) 
    
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_zlim(zlims)

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')

    plt.savefig('skeleton.pdf', bbox_inches='tight')

def compute_height(joints, head_index, l_foot_index, r_foot_index):
    joints = torch.from_numpy(joints)
    left = (joints[:,head_index,1] - joints[:,l_foot_index,1])[0]
    right = (joints[:,head_index,1] - joints[:,r_foot_index,1])[0]
    height = (left + right) / 2
    return height

def compute_metrics_np(similarity_matrix, correct_labels):

    B, _ = similarity_matrix.shape
   
    ranked_indices = np.argsort(-similarity_matrix, axis=1)
    
    correct_label_ranks = np.array([np.where(ranked_indices[i] == correct_labels[i])[0][0] for i in range(B)]) + 1
    
    # Compute R@K
    R_at_1 = np.mean(correct_label_ranks <= 1)
    R_at_2 = np.mean(correct_label_ranks <= 2)
    R_at_3 = np.mean(correct_label_ranks <= 3)
    R_at_4 = np.mean(correct_label_ranks <= 4)
    R_at_5 = np.mean(correct_label_ranks <= 5)
    
    # Compute MRR
    MRR = np.mean(1.0 / correct_label_ranks)
    
    return R_at_1, R_at_2, R_at_3, R_at_4, R_at_5, MRR
