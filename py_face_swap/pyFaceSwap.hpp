#include <string>
#include <Python.h>
#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include "face_swap/face_swap.h"

// OpenGL
#include <GL/glew.h>
// Qt
#include <QApplication>
#include <QOpenGLContext>
#include <QOffscreenSurface>

//using namespace boost::python;
using std::string;

class PyFaceSwap {

    public:
        PyFaceSwap();
        ~PyFaceSwap();

        /* 
         * Create OpenGL context. Notice that only one context per process and only
         * the thread that initializes the context can render
         */
        int createCtx(int argc, PyObject *arglst);

        int setSourceImg(PyObject *pyImg);

        /*
         * bypass: if true, use the last stored 3DMM coefficients, pose, and segmentation
         * init_tracker: if true, use dlib detection and initialize the tracker,
         *               if false, use KCF tracking.
         *     (One should always init_tracker before using the tracker. That is, one must
         *     at least make init_tracker = true once.)
         */
        int setTargetImg(PyObject *pyImg, bool bypass = false, bool init_tracker = true);
        PyObject* swap();

        void loadModels(string landmarks_path, string model_3dmm_h5_path, string model_3dmm_dat_path,
                string reg_model_path, string reg_deploy_path, string reg_mean_path,
                string seg_model_path, string seg_deploy_path, bool generic, bool highQual,
                int gpu_device_id);

    private:
        boost::shared_ptr<face_swap::FaceSwap> fs;
        QApplication *a = NULL;
        QSurfaceFormat *surfaceFormat = NULL;
        QOpenGLContext *openGLContext = NULL;
        QOffscreenSurface *surface = NULL;
};
