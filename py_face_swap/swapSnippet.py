import pyfaceswap
import argparse
import cv2
import sys
import time

def main():
    landmarks = '/root/face_swap/data/models/shape_predictor_68_face_landmarks.dat'       # path to landmarks model file
    model_3dmm_h5 = '/root/face_swap/data/models/BaselFaceModel_mod_wForehead_noEars.h5'  # path to 3DMM file (.h5)
    model_3dmm_dat = '/root/face_swap/data/models/BaselFace.dat'                          # path to 3DMM file (.dat)
    reg_model = '/root/face_swap/data/models/3dmm_cnn_resnet_101.caffemodel'              # path to 3DMM regression CNN model file (.caffemodel)
    reg_deploy = '/root/face_swap/data/models/3dmm_cnn_resnet_101_deploy.prototxt'        # path to 3DMM regression CNN deploy file (.prototxt)
    reg_mean = '/root/face_swap/data/models/3dmm_cnn_resnet_101_mean.binaryproto'         # path to 3DMM regression CNN mean file (.binaryproto)
    seg_model = '/root/face_swap/data/models/face_seg_fcn8s.caffemodel'                   # path to face segmentation CNN model file (.caffemodel)
    seg_deploy = '/root/face_swap/data/models/face_seg_fcn8s_deploy.prototxt'             # path to face segmentation CNN deploy file (.prototxt)
    source = '/root/face_swap/data/images/brad_pitt_01.jpg'     # source image

    parser = argparse.ArgumentParser(description='Draw annotation on video')
    parser.add_argument('-i', help='Input video')
    parser.add_argument('-o', help='Output video')
    args = parser.parse_args()

    fs = pyfaceswap.PyFaceSwap()
    if( fs.initCtx(len(sys.argv), sys.argv) ):
        print 'Initialization failed!'
        return
    gpuId = 1
    expReg = 1
    genericFace = 0
    highQual = 0
    fs.loadModels(landmarks, model_3dmm_h5, model_3dmm_dat, reg_model, reg_deploy,\
            reg_mean, seg_model, seg_deploy, genericFace, expReg, highQual, gpuId)

    if highQual:
        targetHeight = 240.0
    else:
        targetHeight = 200.0

    cap = cv2.VideoCapture(args.i)

    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    fps = 24
    size = ( int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) )

    sourceImg = cv2.imread(source)
    srcRszRatio = targetHeight / sourceImg.shape[0]
    rszSrcImg = cv2.resize(sourceImg, None, None, fx=srcRszRatio, fy=srcRszRatio,\
            interpolation=cv2.INTER_LINEAR)
    if ( fs.setSourceImg(rszSrcImg) ):
        print 'Set Source Image Failed!'
        return

    result = []
    frame_id = 0
    max_frames = 8 * fps


    print 'Input: {}, FPS: {}, HighQuality: {} Size: ({},{})'.format(args.i, fps, highQual, size[0], size[1]) 
    start = time.time()
    while True:
        if frame_id > max_frames:
            break
        ret, targetImg = cap.read()
        print 'Progress: {:d}%...'.format(int((frame_id+1)/float(max_frames)*100))
        if frame_id % 2:
            frame_id = frame_id + 1
            result.append(result[-1])
            continue
        if ret:
            tgtRszRatio = targetHeight / targetImg.shape[0]
            rszTgtImg = cv2.resize(targetImg, None, None, fx=tgtRszRatio, fy=tgtRszRatio,\
                    interpolation=cv2.INTER_LINEAR)
            if ( fs.setTargetImg(rszTgtImg) ):
                print 'Set Target Image Failed! Use last result'
                if frame_id == 0:
                    result.append(targetImg)
                else:
                    result.append(result[-1])
                frame_id = frame_id + 1
                continue
            tmp = fs.swap()
            rszRes = cv2.resize(tmp, size, interpolation=cv2.INTER_LINEAR)
            result.append(rszRes)
        else:
            break
        frame_id = frame_id + 1
    swap = time.time() - start
    
    if result:
        video = cv2.VideoWriter(args.o,fourcc,fps,size)
        for img in result:
            video.write(img)

    print 'Avg. FPS: {}'.format(float(len(result))/swap)
    video.release()
    del fs

if __name__ == '__main__':
    main()

