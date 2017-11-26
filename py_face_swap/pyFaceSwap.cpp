// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// C++ code
#include <string>
#include <boost/python.hpp>

#include <iostream>

#include <boost/shared_ptr.hpp>
#include "pyFaceSwap.hpp"
#include "face_swap/face_swap.h"

#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API
#include "pyboostcvconverter.hpp"

using namespace boost::python;
using std::cout;
using std::string;

PyFaceSwap::PyFaceSwap() {
    import_array();
}

PyFaceSwap::~PyFaceSwap() {
    if (surface) delete surface;
    if (openGLContext) delete openGLContext;
    if (surfaceFormat) delete surfaceFormat;
    if (a) delete a;
}

void PyFaceSwap::loadModels(string landmarks_path, string model_3dmm_h5_path, string model_3dmm_dat_path,
        string reg_model_path, string reg_deploy_path, string reg_mean_path,
        string seg_model_path, string seg_deploy_path, bool generic, bool highQual,
        int gpu_device_id) {

    const bool with_gpu = 1;

    fs = boost::shared_ptr<face_swap::FaceSwap>( new face_swap::FaceSwap(landmarks_path, model_3dmm_h5_path, model_3dmm_dat_path,
            reg_model_path, reg_deploy_path, reg_mean_path, generic, highQual,
            with_gpu, (int)gpu_device_id) );

    fs->setSegmentationModel(seg_model_path, seg_deploy_path);
}


int PyFaceSwap::setSourceImg(PyObject* pyImg) {
    cv::Mat image, source_seg;
    image = pbcvt::fromNDArrayToMat(pyImg);
    //cv::imshow("Source", image);
    //cv::waitKey(0);
    int ret = fs->setSource(image, source_seg);
    if (!ret) return -1;
    return 0;
}

int PyFaceSwap::setTargetImg(PyObject* pyImg, bool bypass, bool init_tracker) {
    cv::Mat image, target_seg;
    image = pbcvt::fromNDArrayToMat(pyImg);
    //cv::imshow("Target", image);
    //cv::waitKey(0);
    int ret = fs->setTarget(image, target_seg, bypass, init_tracker);
    if (!ret) return -1;
    return 0;
}

int PyFaceSwap::createCtx(int argc, PyObject *arglst) {

    size_t cnt = PyList_GET_SIZE(arglst);
    char **argv = new char*[cnt + 1];
    for (size_t i = 0; i < cnt; i++) {
        PyObject *s = PyList_GET_ITEM(arglst, i);
        assert (PyString_Check(s));     // likewise
        size_t len = PyString_GET_SIZE(s);
        char *copy = new char[len + 1];
        memcpy(copy, PyString_AS_STRING(s), len + 1);
        argv[i] = copy;
    }
    argv[cnt] = NULL;

    // Intialize OpenGL context
    a = new QApplication(argc, argv);
    for (size_t i = 0; i < cnt; i++)
        delete [] argv[i];
    delete [] argv;

    surfaceFormat = new QSurfaceFormat;
    surfaceFormat->setMajorVersion(1);
    surfaceFormat->setMinorVersion(5);

    openGLContext = new QOpenGLContext;
    openGLContext->setFormat(*surfaceFormat);
    openGLContext->create();
    if (!openGLContext->isValid()) return -1;

    surface = new QOffscreenSurface;
    surface->setFormat(*surfaceFormat);
    surface->create();
    if (!surface->isValid()) return -2;

    openGLContext->makeCurrent(surface);

    // Initialize GLEW
    GLenum err = glewInit();
    if (GLEW_OK != err) return -3;

    return 0;
}

PyObject* PyFaceSwap::swap() {
    cv::Mat rendered_img;
    rendered_img = fs->swap();
    PyObject *ret = pbcvt::fromMatToNDArray(rendered_img);
    return ret;
}

static void init_ar(){
    Py_Initialize();
    import_array();
    return NUMPY_IMPORT_ARRAY_RETVAL;
}


BOOST_PYTHON_MODULE(pyfaceswap)
{
    init_ar();
    class_<PyFaceSwap>("PyFaceSwap", init<>())
        .def(init<>())
        .def("loadModels", &PyFaceSwap::loadModels)
        .def("setSourceImg", &PyFaceSwap::setSourceImg)
        .def("setTargetImg", &PyFaceSwap::setTargetImg)
        .def("createCtx", &PyFaceSwap::createCtx)
        .def("swap", &PyFaceSwap::swap)
        ;
}
