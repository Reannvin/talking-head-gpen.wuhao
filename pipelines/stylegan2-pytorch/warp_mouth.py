import cv2
import numpy as np
from skimage import transform as trans
import os
from tqdm import tqdm


def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

    return dst

# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = [] 
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    mask=cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)
    #mask=np.zeros(r2.shape[:2],dtype=np.uint8)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    
    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    #img2Rect=cv2.resize(img2Rect,(r2[2],r2[3]))
    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    # print(r2[3], r2[2])
    # print(r2[1], r2[1]+r2[3], r2[0], r2[0]+r2[2])
    # print(img2.shape)
    # print(img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]].shape)
    # print(img2Rect.shape)
    # print(mask.shape)
    if r2[1]+r2[3]>img2.shape[0] :
        height = img2.shape[0] - r2[1]
        mask=cv2.resize(mask,(mask.shape[1],height))
        img2Rect=cv2.resize(img2Rect,(mask.shape[1],height))
    else:
        height = r2[3]
    if r2[0]+r2[2]>img2.shape[1]:
        width = img2.shape[1] - r2[0]
        mask=cv2.resize(mask,(width,mask.shape[0]))
        img2Rect=cv2.resize(img2Rect,(width,mask.shape[0]))
    else:
        width = r2[2]
    # print(img2[r2[1]:r2[1]+height, r2[0]:r2[0]+width].shape)
    # print(img2Rect.shape)
    # print(mask.shape)
    img2[r2[1]:r2[1]+height, r2[0]:r2[0]+width] = img2[r2[1]:r2[1]+height, r2[0]:r2[0]+width] * ( (1.0, 1.0, 1.0) - mask )
     
    img2[r2[1]:r2[1]+height, r2[0]:r2[0]+width] = img2[r2[1]:r2[1]+height, r2[0]:r2[0]+width] + img2Rect
   # return img2

def get_triangle_indices(points):
    rect = cv2.boundingRect(np.float32([points]))
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(points.tolist())
    triangle_list = subdiv.getTriangleList()
    indices = []
    for t in triangle_list:
        indices.append([np.where((points == t[i:i+2]).all(axis=1))[0][0] for i in range(0, 6, 2)])
    return indices

def warp_mouth(image1,image2,landmark1,landmark2):
# path1 = '39.jpg'
# path2 = '51.jpg'
    

    #mouth = landmark[48:68] get mouth mask
    src_image = image1
    dst_image = image2
    src_points=landmark1
    dst_points=landmark2
    #mouth_points = list(range(60))
    mouth_points=range(48,60)
    #mouth_points=[48,51,54,57]
    src_triangles = get_triangle_indices(src_points[mouth_points])
    dst_triangles = get_triangle_indices(dst_points[mouth_points])

    warped_mouth = np.zeros_like(dst_image)

    # 对每个三角形进行仿射变换并贴合到目标图像上
    for tri_indices in src_triangles:
        # 获取三角形的三个顶点
        src_tri = np.float32([src_points[mouth_points[i]] for i in tri_indices])
        dst_tri = np.float32([dst_points[mouth_points[i]] for i in tri_indices])
        warpTriangle(src_image, warped_mouth, src_tri, dst_tri)
        #print(warped_mouth)
        # 计算仿射变换矩阵
    #     tform=trans.SimilarityTransform()
    #     tform.estimate(src_tri, dst_tri)
    #     M=tform.params[0:2,:]
    #    # M = cv2.getAffineTransform(src_tri, dst_tri)
    #     warped_mouth = cv2.warpAffine(src_image, M, (dst_image.shape[1], dst_image.shape[0]), dst=warped_mouth, borderMode=cv2.BORDER_TRANSPARENT)
   # warped_mouth[:warped_mouth.shape[0]//2,:]=0
    cv2.imwrite('warped_mouth.jpg', warped_mouth)
    half_face_point=list(range(0,17))
    mask = np.zeros_like(dst_image[:, :, 0])
    mask=cv2.fillConvexPoly(mask, np.int32(dst_points[mouth_points]), (255))

    # 混合来源的嘴部区域到目标图像
    output = cv2.seamlessClone(warped_mouth, dst_image, mask, tuple(np.mean(dst_points[mouth_points], axis=0).astype(int)), cv2.NORMAL_CLONE)
    return output
    #cv2.imwrite('output.jpg', output)

def get_face_dir(img_dir):
    img_dir="crop_FFHQ"
    img_list=os.listdir(img_dir)
    img_list=[os.path.join(img_dir,img) for img in img_list if img.endswith('.jpg')]
    os.makedirs('warp_faces',exist_ok=True)
    gen_image_num=200

    for i in tqdm(range(gen_image_num)):
        save_path=os.path.join('warp_faces',str(i)+'.jpg')
        idx=np.random.choice(len(img_list),2)
        img1_path=img_list[idx[0]]
        img2_path=img_list[idx[1]]
        output=warp_mouth(img1_path,img2_path)
        cv2.imwrite(save_path,output)

