import numpy as np
# from shapely.geometry import Point
# from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
import math
import cv2 as cv

ax = plt.axes(projection='3d')

def equation_plane(para1, para2, para3):  
    x1, y1, z1 = para1
    x2, y2, z2 = para2
    x3, y3, z3 = para3
    a1 = x2 - x1 
    b1 = y2 - y1 
    c1 = z2 - z1 
    a2 = x3 - x1 
    b2 = y3 - y1 
    c2 = z3 - z1 
    a = b1 * c2 - b2 * c1 
    b = a2 * c1 - a1 * c2 
    c = a1 * b2 - b1 * a2 
    d = (- a * x1 - b * y1 - c * z1) 
    return a, b, c, d

def plane_equation_2(p1, p2, p3):

    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    v1 = p3 - p1
    v2 = p2 - p1
    cp = np.cross(v1, v2)
    a, b, c = cp
    d = np.dot(cp, p3)
    return a, b, c, d

def plane_equation_from_point_normal_vector(normal_vector, point):
    x,y,z = normal_vector
    A, B, C = point
    return x, y, z, -(x*A + y*B + z*C)

def get_angle(para1, para2):
    a1, b1, c1, _ = para1
    a2, b2, c2, _ = para2
    d = (a1 * a2) + (b1 * b2) + (c1 * c2) 
    e1 = math.sqrt( a1 * a1 + b1 * b1 + c1 * c1) 
    e2 = math.sqrt( a2 * a2 + b2 * b2 + c2 * c2) 
    d = d / (e1 * e2)
    d = round(d,1) 
    A = math.degrees(math.acos(d)) 
    return A

def norm2(X):
	return np.sqrt(np.sum(X ** 2))

def normalized(X):
	return X / norm2(X)

def get_planes_intersection(A, B):
	U = normalized(np.cross(A[:-1], B[:-1]))
	M = np.array((A[:-1], B[:-1], U))
	X = np.array((-A[-1], -B[-1], 0.))
	return U, np.linalg.solve(M, X)

def get_two_planes_intersection_vector(A, B):
    a,b,c,_ = A
    x,y,z,_ = B
    return [y*c - z*b, z*a - x*c, x*b - y*a]

def find_2d_corner(image, mask,object_depth,point_cloud,axis, start_loop, end_loop, direct):

    for i in range(start_loop, end_loop, direct):
        if axis == 'x':
            mask_collumn = mask[:,i]
        else:
            mask_collumn = mask[i]
        if mask_collumn.max() == 1:
            position_arr = np.where(mask_collumn == 1)
            temp_arr = []
            for temp in position_arr[0]:
                if axis == 'x':
                    if np.isnan(point_cloud[temp][i]).any():
                        continue
                    temp_arr.append(temp)
                else:
                    if np.isnan(point_cloud[i][temp]).any() or object_depth[i][temp] > 1:
                        continue
                    temp_arr.append(temp)

            if not temp_arr:
                continue

            position = temp_arr[int(len(temp_arr)/2)]
            if axis == 'x':
                box_corner = [i, position]
            else:
                box_corner = [position, i]
            break
    image = cv.circle(image,tuple(box_corner), 5, (0,0,255), -1)
    return box_corner, image

def distance_point_plane(M, alpha):
    a1, b1, c1, d1 = alpha
    a2, b2, c2, _ = M
    num = (a1 * a2) + (b1 * b2) + (c1 * c2) + d1
    denom = math.sqrt( a1 * a1 + b1 * b1 + c1 * c1) 
    return num/denom

def distance_two_point(A, B):
    a1, b1, c1, _ = A
    a2, b2, c2, _ = B
    denom = math.sqrt( (a1 - a2)*(a1 - a2) + (b1 - b2)*(b1 - b2) + (c1 - c2)*(c1 - c2))
    return denom

def xyz_cluster(point_1, point_2):
    return[[point_1[0],point_2[0]],[point_1[1],point_2[1]],[point_1[2],point_2[2]]]


# Test all function
# find plane equation function
# xy_plane = equation_plane([0,0,0],[1,0,0],[0,1,0])
# xz_plane = equation_plane([0,0,0],[1,0,0],[0,0,1])
# yz_plane = equation_plane([0,0,0],[0,0,1],[0,1,0])
# print(xy_plane, xz_plane, yz_plane)

# intersection_vector = get_two_planes_intersection_vector(xy_plane, xz_plane)
# print(intersection_vector)





f =  open('dummy_data/depth_map.npy', 'rb') 
object_depth = np.load(f, allow_pickle=True)
point_cloud = np.load(f, allow_pickle=True)
mask = np.load(f, allow_pickle=True)
h, w = mask.shape
image = cv.imread('dummy_data/image_seg.jpg')

left_box_corner, image = find_2d_corner(image, mask, object_depth, point_cloud, 'x', 0, w, 1)
right_box_corner, image = find_2d_corner(image, mask, object_depth, point_cloud, 'x', w-1, 0, -1)
top_box_corner, _ = find_2d_corner(image, mask, object_depth, point_cloud, 'y', 0, h, 1)
top_box_corner = [580,430]
image = cv.circle(image,tuple(top_box_corner), 5, (0,0,255), -1)
button_box_corner, image = find_2d_corner(image, mask, object_depth, point_cloud, 'y', h-1, 0, -1)

box_cloud_data = []
for y in range(top_box_corner[1], button_box_corner[1],2):
    for x in range(left_box_corner[0], right_box_corner[0],2):
        depth = object_depth[y][x]
        if depth > 0 and depth < 2:
            box_cloud_data.append(point_cloud[y][x])
            try:
                if temp_depth > depth:
                    temp_depth = depth
                    nearest_point = [x,y]
            except:
                temp_depth = depth
nearest_point = [595,443]
image = cv.circle(image,tuple(nearest_point), 5, (0,0,255), -1)
# cv.imshow('image', image)
# cv.waitKey(0)
left = point_cloud[left_box_corner[1]][left_box_corner[0]]
right = point_cloud[right_box_corner[1]][right_box_corner[0]]
nearest = point_cloud[nearest_point[1]][nearest_point[0]]
button = point_cloud[button_box_corner[1]][button_box_corner[0]]
top = point_cloud[top_box_corner[1]][top_box_corner[0]]

plane_left = equation_plane([left[0],left[1],left[2]], 
                                [button[0],button[1],button[2]],
                                [nearest[0],nearest[1],nearest[2]])
plane_right = equation_plane([right[0],right[1],right[2]], 
                                [button[0],button[1],button[2]],
                                [nearest[0],nearest[1],nearest[2]])

print(get_angle(plane_left, plane_right))
print(nearest[:3])
# points_accumulate = get_planes_intersection(plane_left, plane_right)
# points_accumulate = [np.array(points_accumulate[0]), np.array(points_accumulate[0]) + np.array(points_accumulate[1])]
# ax.plot(np.array(points_accumulate)[:,0], 
#             np.array(points_accumulate)[:,1],
#             np.array(points_accumulate)[:,2])

# inetersection_line_vector = get_two_planes_intersection_vector(plane_left, plane_right)
# point_in_inetersection_line = np.array([np.array(inetersection_line_vector) + nearest[:3], button[:3]])

# plane vector (a,b,c)
plan_top_normal_vector = nearest[:3] - button[:3]
# plane_top = point_in_inetersection_line[1] - point_in_inetersection_line[0]

# plane equation (a,b,c,d)
plane_top = plane_equation_from_point_normal_vector(plan_top_normal_vector, nearest[:3])
# plane_top = np.append(plane_top, np.sum(nearest[:3]*plane_top))

print(get_angle(plane_left, plane_top))
print(get_angle(plane_left, plane_left))

print(distance_point_plane(top, plane_top))

print('distance from nearest to button = ', distance_two_point(button, top))

x,y,z = xyz_cluster(plane_top,plane_left)
ax.plot(x, y, z, c='r')

ax.plot(point_in_inetersection_line[:,0], 
            point_in_inetersection_line[:,1],
            point_in_inetersection_line[:,2])

ax.scatter3D(np.array(box_cloud_data)[:,0], np.array(box_cloud_data)[:,1], np.array(box_cloud_data)[:,2], cmap='Greens', s=0.5)
plt.show()