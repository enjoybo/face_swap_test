import pyfaceswap
import argparse
import cv2
import sys
import time

def main():
    parser = argparse.ArgumentParser(description='Draw annotation on video')
    parser.add_argument('-i', help='Input video', required=True)
    parser.add_argument('-o', help='Output video', required=True)
    parser.add_argument('--idol', help='Source Idol', type=int, required=True, default=0)
    parser.add_argument('--gpu', help='GPU ID', type=int, required=True, default=0)
    parser.add_argument('--rotate', help='Source Idol', type=int,required=True)
    parser.add_argument('--highQual', help='ResNet101 (or else VGG16)', type=int, required=True, default=0)
    parser.add_argument('--imgH', help='Image height', type=int, required=True)
    args = parser.parse_args()

    landmarks = '/root/face_swap/data/models/shape_predictor_68_face_landmarks.dat'       # path to landmarks model file
    model_3dmm_h5 = '/root/face_swap/data/models/BaselFaceModel_mod_wForehead_noEars.h5'  # path to 3DMM file (.h5)
    model_3dmm_dat = '/root/face_swap/data/models/BaselFace.dat'                          # path to 3DMM file (.dat)
    reg_mean = '/root/face_swap/data/models/dfm_resnet_101_mean.binaryproto'         # path to 3DMM regression CNN mean file (.binaryproto)
    seg_model = '/root/face_swap/data/models/face_seg_fcn8s.caffemodel'                   # path to face segmentation CNN model file (.caffemodel)
    seg_deploy = '/root/face_swap/data/models/face_seg_fcn8s_deploy.prototxt'             # path to face segmentation CNN deploy file (.prototxt)
    sourceDir = '/root/face_swap/data/images/'

    sources = ['brad_pitt_01.jpg', 'emma-stone.jpg', 'emma-watson.jpg', 'donald-trump.jpg',\
            'chenwu.jpg', 'nick-young.jpg']

    source = '%s/%s'%(sourceDir, sources[args.idol])
    print 'Using source %s'%(sources[args.idol])

    gpuId = args.gpu
    generic = 0

    highQual = args.highQual
    if highQual:
        print 'High Quality Enabled! (ResNet-101)'
        reg_model = '/root/face_swap/data/models/dfm_resnet_101.caffemodel'              # path to 3DMM regression CNN model file (.caffemodel)
        reg_deploy = '/root/face_swap/data/models/dfm_resnet_101_deploy.prototxt'        # path to 3DMM regression CNN deploy file (.prototxt)
    else:
        print 'Low Quality Enabled! (VGG16)'
        reg_model = '/root/face_swap/data/models/dfm_vgg16.caffemodel'              # path to 3DMM regression CNN model file (.caffemodel)
        reg_deploy = '/root/face_swap/data/models/dfm_vgg16_deploy.prototxt'        # path to 3DMM regression CNN deploy file (.prototxt)

    targetHeight = float(args.imgH)
    pfs = pyfaceswap.PyFaceSwap()

    if( pfs.createCtx(len(sys.argv), sys.argv) ):
        print 'Initialization failed!'
        return

    pfs.loadModels(landmarks, model_3dmm_h5, model_3dmm_dat, reg_model, reg_deploy,\
            reg_mean, seg_model, seg_deploy, generic, highQual, gpuId)

    sourceImg = cv2.imread(source)
    if ( pfs.setSourceImg(sourceImg) ):
        print 'Set Source Image Failed!'
    print '>>> Source set!'

    cap = cv2.VideoCapture(args.i)
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    fps = 30
    if args.rotate:
        size = ( int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) )
    else:
        size = ( int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) )

    result = []
    frame_id = 0
    print 'Input: {}, FPS: {}, Size: ({},{})'.format(args.i, fps, size[0], size[1]) 
    failed = True
    start = time.time()
    while True:
        ret, targetImg = cap.read()
        if frame_id == 0:
            lastImg = targetImg
        if args.rotate:
            targetImg = cv2.rotate(targetImg, cv2.ROTATE_90_CLOCKWISE)
        print frame_id
        if ret:
            tgtRszRatio = float(targetHeight) / targetImg.shape[0]
            rszTgtImg = cv2.resize(targetImg, None, None, fx=tgtRszRatio, fy=tgtRszRatio,\
                    interpolation=cv2.INTER_LINEAR)
            if ( not pfs.setTargetImg(rszTgtImg, False, failed) ):
                tmp = pfs.swap()
                if type(tmp) == type(None):
                    failed = True
                else:
                    rszRes = cv2.resize(tmp, size, interpolation=cv2.INTER_LINEAR)
                    lastImg = rszRes
                    failed = False
            else:
                failed = True
            result.append(lastImg)
            frame_id = frame_id + 1
        else:
            break
    swap = time.time() - start
    
    if result:
        video = cv2.VideoWriter(args.o,fourcc,fps,size)
        for img in result:
            video.write(img)

    print 'Avg. FPS: {}'.format(float(len(result))/swap)
    video.release()
    del pfs

if __name__ == '__main__':
    main()

