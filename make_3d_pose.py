import cv2
import numpy as np

import sys
sys.path.append('..')

from settings import ARGS
from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from lifting.prob_model import Prob3dPose

def make_pos(pose_3d):

    '''
    :param pose_3d: array, shape = (the number of humans, 3, 17)
    :return: string
    '''

    s = ''
    person_num, dim, point_num = pose_3d.shape
    person_num =  1
    for i in range(person_num):
        for j in range(point_num):
            s += str(j)
            for k in range(dim):
                s += ' '+ str(pose_3d[i][k][j])
            s += ', '
        s += '\n'
    return s


# hyper_params
args = ARGS

# size of images
w, h = model_wh(args['RESOLUTION'])

# make tf_pose_estimator
e = TfPoseEstimator(get_graph_path(args['MODEL']), target_size=(w, h))

# caption of movie
cap = cv2.VideoCapture(args['INPUT_PATH'])

# load 3d estimator's weight
poseLifting = Prob3dPose('./lifting/models/prob_model_params.mat')


counter = 0
while cap.isOpened():
    # read frames
    ret_val, image = cap.read()
    counter += 1

    # skip frames according to skip_frame
    if counter % args['FRAME_SKIP'] != 0:
        continue

    # inference humans born
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
    if not args['SHOWBG']:
        image = np.zeros(image.shape)

    # draw humans born
    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

    # size
    image_h, image_w = image.shape[:2]
    standard_w = 640
    standard_h = 480

    pose_2d_mpiis = []
    visibilities = []
    for human in humans:
        pose_2d_mpii, visibility = common.MPIIPart.from_coco(human)
        pose_2d_mpiis.append([(int(x * standard_w + 0.5), int(y * standard_h + 0.5)) for x, y in pose_2d_mpii])
        visibilities.append(visibility)

    pose_2d_mpiis = np.array(pose_2d_mpiis)
    visibilities = np.array(visibilities)
    transformed_pose2d, weights = poseLifting.transform_joints(pose_2d_mpiis, visibilities)

    # make 3d born model
    pose_3d = poseLifting.compute_3d(transformed_pose2d, weights)

    # make pos.txt
    s = make_pos(pose_3d)

    with open(args['OUTPUT_PATH'], 'a') as f:
        f.write(s)

