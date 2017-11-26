#include "face_swap/face_swap.h"
#include "face_swap/utilities.h"

// std
#include <iostream>
#include <limits>
#include <chrono>

// OpenCV
#include <opencv2/photo.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>  // Debug

// dlib
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing/shape_predictor.h>

#define DEBUG 0

using namespace std::chrono;

namespace face_swap
{
    FaceSwap::FaceSwap(const std::string& landmarks_path, const std::string& model_3dmm_h5_path,
        const std::string& model_3dmm_dat_path, const std::string& reg_model_path,
        const std::string& reg_deploy_path, const std::string& reg_mean_path,
        bool generic, bool highQual, bool with_gpu, int gpu_device_id) :
		m_with_gpu(with_gpu),
		m_gpu_device_id(gpu_device_id)
    {
        m_tracker = KCFTracker(true, false, true, false);
        m_detector = dlib::get_frontal_face_detector();
        dlib::deserialize(landmarks_path) >> m_pose_model;

        // Initialize CNN 3DMM with exression
        m_cnn_3dmm_expr = std::make_unique<CNN3DMMExpr>(
			reg_deploy_path, reg_model_path, reg_mean_path, model_3dmm_dat_path,
			generic, highQual, with_gpu, gpu_device_id);

        // Load Basel 3DMM
        m_basel_3dmm = std::make_unique<Basel3DMM>();
        *m_basel_3dmm = Basel3DMM::load(model_3dmm_h5_path);

        // Create renderer
        m_face_renderer = std::make_unique<FaceRenderer>();
    }

	void FaceSwap::setSegmentationModel(const std::string& seg_model_path,
		const std::string& seg_deploy_path)
	{
		m_face_seg = std::make_unique<face_seg::FaceSeg>(seg_deploy_path,
			seg_model_path, m_with_gpu, m_gpu_device_id);
	}

	void FaceSwap::clearSegmentationModel()
	{
		m_face_seg = nullptr;
	}

	bool FaceSwap::isSegmentationModelInit()
	{
		return m_face_seg != nullptr;
	}

	bool FaceSwap::setSource(const cv::Mat& img, const cv::Mat& seg)
    {
        m_source_img = img;

        // Preprocess image
        std::vector<cv::Point> cropped_landmarks;
        cv::Mat cropped_img, cropped_seg;
        if (!preprocessImages(img, seg, m_src_landmarks, cropped_landmarks,
			cropped_img, cropped_seg))
            return false;

#if DEBUG
        int start_ms, end_ms;
        start_ms = duration_cast< milliseconds >(
            system_clock::now().time_since_epoch()
        ).count();
#endif
		// If segmentation was not specified and we have a segmentation model then
		// calculate the segmentation
		if (cropped_seg.empty() && m_face_seg != nullptr)
			cropped_seg = m_face_seg->process(cropped_img);
#if DEBUG
        end_ms = duration_cast< milliseconds >(
            system_clock::now().time_since_epoch()
        ).count();
        std::cout << "Segmentation: " << (end_ms-start_ms) << " ms" << std::endl;
#endif
        // Calculate coefficients and pose
        cv::Mat shape_coefficients, tex_coefficients, expr_coefficients;
        cv::Mat vecR, vecT, K;
        m_cnn_3dmm_expr->process(cropped_img, cropped_landmarks, shape_coefficients,
            tex_coefficients, expr_coefficients, vecR, vecT, K);
#if DEBUG
        start_ms = duration_cast< milliseconds >(
            system_clock::now().time_since_epoch()
        ).count();
#endif
        // Create mesh
        m_src_mesh = m_basel_3dmm->sample(shape_coefficients, tex_coefficients, 
            expr_coefficients);
#if DEBUG
        end_ms = duration_cast< milliseconds >(
            system_clock::now().time_since_epoch()
        ).count();
        std::cout << "Create mesh: " << (end_ms-start_ms) << " ms" << std::endl;

        start_ms = duration_cast< milliseconds >(
            system_clock::now().time_since_epoch()
        ).count();
#endif
        // Texture mesh
        generateTexture(m_src_mesh, cropped_img, cropped_seg, vecR, vecT, K, 
            m_tex, m_uv);
#if DEBUG
        end_ms = duration_cast< milliseconds >(
            system_clock::now().time_since_epoch()
        ).count();
        std::cout << "Generate texture mesh: " << (end_ms-start_ms) << " ms" << std::endl;
#endif

        /// Debug ///
        m_src_cropped_img = cropped_img;
        m_src_cropped_seg = cropped_seg;
        m_src_cropped_landmarks = cropped_landmarks;
        m_src_vecR = vecR;
        m_src_vecT = vecT;
        m_src_K = K;
        /////////////

        return true;
    }

    bool FaceSwap::setTarget(const cv::Mat& img, const cv::Mat& seg, bool bypass, bool init_track)
    {
        m_target_img = img;
        //m_target_seg = seg;

        // Preprocess image
        std::vector<cv::Point> cropped_landmarks;
        cv::Mat cropped_img, cropped_seg;
        if (!bypass) {
            if (!preprocessImages(img, seg, m_tgt_landmarks, cropped_landmarks,
                cropped_img, cropped_seg, m_target_bbox, init_track))
                return false;
            m_tgt_cropped_img = cropped_img;
        }

		// If segmentation was not specified and we have a segmentation model then
		// calculate the segmentation
#if DEBUG
        int start_ms, end_ms;
        start_ms = duration_cast< milliseconds >(
            system_clock::now().time_since_epoch()
        ).count();
#endif
        if (!bypass) {
            if (cropped_seg.empty() && m_face_seg != nullptr)
            {
                cropped_seg = m_face_seg->process(cropped_img);
                m_target_seg = cv::Mat::zeros(img.size(), CV_8U);
                cropped_seg.copyTo(m_target_seg(m_target_bbox));
            }
            m_tgt_cropped_seg = cropped_seg;
        }
#if DEBUG
        end_ms = duration_cast< milliseconds >(
            system_clock::now().time_since_epoch()
        ).count();
        std::cout << "Segmentation: " << (end_ms-start_ms) << " ms" << std::endl;
#endif
        
        /// Debug ///
        m_tgt_cropped_landmarks = cropped_landmarks;
        /////////////
#if DEBUG
        start_ms = duration_cast< milliseconds >(
            system_clock::now().time_since_epoch()
        ).count();
#endif

        // Calculate coefficients and pose
        if (!bypass) {
            m_cnn_3dmm_expr->process(cropped_img, cropped_landmarks, m_shape_coefficients,
                m_tex_coefficients, m_expr_coefficients, m_vecR, m_vecT, m_K, bypass);
        }
#if DEBUG
        end_ms = duration_cast< milliseconds >(
            system_clock::now().time_since_epoch()
        ).count();
        std::cout << "C&P: " << (end_ms-start_ms) << " ms" << std::endl;
#endif
    
#if DEBUG
        start_ms = duration_cast< milliseconds >(
            system_clock::now().time_since_epoch()
        ).count();
#endif
        // Create mesh
        m_dst_mesh = m_basel_3dmm->sample(m_shape_coefficients, m_tex_coefficients,
            m_expr_coefficients);
        m_dst_mesh.tex = m_tex;
        m_dst_mesh.uv = m_uv;
#if DEBUG
        end_ms = duration_cast< milliseconds >(
            system_clock::now().time_since_epoch()
        ).count();
        std::cout << "sampleMesh: " << (end_ms-start_ms) << " ms" << std::endl;
#endif

        return true;
    }

    cv::Mat FaceSwap::swap()
    {
#if DEBUG
        int start_ms, end_ms;
        start_ms = duration_cast< milliseconds >(
            system_clock::now().time_since_epoch()
        ).count();
#endif
        // Initialize renderer
        m_face_renderer->init(m_tgt_cropped_img.cols, m_tgt_cropped_img.rows);
        m_face_renderer->setProjection(m_K.at<float>(4));
        m_face_renderer->setMesh(m_dst_mesh);

        // Render
        cv::Mat rendered_img;
        m_face_renderer->render(m_vecR, m_vecT);
        m_face_renderer->getFrameBuffer(rendered_img);

        // Blend images
        cv::Mat tgt_rendered_img = cv::Mat::zeros(m_target_img.size(), CV_8UC3);
        rendered_img.copyTo(tgt_rendered_img(m_target_bbox));

        m_tgt_rendered_img = tgt_rendered_img;  // For debug

        cv::Mat res = blend(tgt_rendered_img, m_target_img, m_target_seg);
#if DEBUG
        end_ms = duration_cast< milliseconds >(
            system_clock::now().time_since_epoch()
        ).count();
        std::cout << "swap: " << (end_ms-start_ms) << " ms" << std::endl;
#endif
        return res;
    }

    const Mesh & FaceSwap::getSourceMesh() const
    {
        return m_src_mesh;
    }

    const Mesh & FaceSwap::getTargetMesh() const
    {
        return m_dst_mesh;
    }


    void dlib_obj_to_points(const dlib::full_object_detection& obj,
        std::vector<cv::Point>& points)
    {
        points.resize(obj.num_parts());
        for (unsigned long i = 0; i < obj.num_parts(); ++i)
        {
            cv::Point& p = points[i];
            const dlib::point& obj_p = obj.part(i);
            p.x = (float)obj_p.x();
            p.y = (float)obj_p.y();
        }
    }

    void FaceSwap::extract_landmarks(const cv::Mat& frame, std::vector<cv::Point>& landmarks, bool init)
    {
        // Convert OpenCV's mat to dlib format 
        dlib::cv_image<dlib::bgr_pixel> dlib_frame(frame);
        cv::Rect result;
        dlib::rectangle face;
        bool hasFace = false;

#if DEBUG
        int start_ms, end_ms;
        start_ms = duration_cast< milliseconds >(
            system_clock::now().time_since_epoch()
        ).count();
#endif
        if (!init) {
			result = m_tracker.update(frame);
            face = dlib::rectangle(result.tl().x, result.tl().y,
                                result.br().x, result.br().y);
            hasFace = true;
        }
        else {
            // Detect bounding boxes around all the faces in the image.
            std::vector<dlib::rectangle> faces = m_detector(dlib_frame);
            if (faces.size()) {
                face = faces[0];
                hasFace = true;
            }
        }
#if DEBUG
        end_ms = duration_cast< milliseconds >(
            system_clock::now().time_since_epoch()
        ).count();
        std::cout << "Face detection: " << (end_ms-start_ms) << " ms" << std::endl;
#endif

#if DEBUG
        start_ms = duration_cast< milliseconds >(
            system_clock::now().time_since_epoch()
        ).count();
#endif
        //frame_landmarks.faces.resize(faces.size());
        if (hasFace) {
            if (init) {
                m_tracker.init( cv::Rect(face.left(), face.top(),
                            face.width(), face.height()), frame );
            }
            // Set landmarks
            dlib::full_object_detection shape = m_pose_model(dlib_frame, face);
            dlib_obj_to_points(shape, landmarks);
        }
#if DEBUG
        end_ms = duration_cast< milliseconds >(
            system_clock::now().time_since_epoch()
        ).count();
        std::cout << "Landmark detection: " << (end_ms-start_ms) << " ms" << std::endl;
#endif
    }

    cv::Rect getFaceBBoxFromLandmarks(const std::vector<cv::Point>& landmarks,
        const cv::Size& frameSize, bool square)
    {
        int xmin(std::numeric_limits<int>::max()), ymin(std::numeric_limits<int>::max()),
            xmax(-1), ymax(-1), sumx(0), sumy(0);
        for (const cv::Point& p : landmarks)
        {
            xmin = std::min(xmin, p.x);
            ymin = std::min(ymin, p.y);
            xmax = std::max(xmax, p.x);
            ymax = std::max(ymax, p.y);
            sumx += p.x;
            sumy += p.y;
        }

        int width = xmax - xmin + 1;
        int height = ymax - ymin + 1;
        int centerx = (xmin + xmax) / 2;
        int centery = (ymin + ymax) / 2;
        int avgx = (int)std::round(sumx / landmarks.size());
        int avgy = (int)std::round(sumy / landmarks.size());
        int devx = centerx - avgx;
        int devy = centery - avgy;
        int dleft = (int)std::round(0.1*width) + abs(devx < 0 ? devx : 0);
        int dtop = (int)std::round(height*(std::max(float(width) / height, 1.0f) * 2 - 1)) + abs(devy < 0 ? devy : 0);
        int dright = (int)std::round(0.1*width) + abs(devx > 0 ? devx : 0);
        int dbottom = (int)std::round(0.1*height) + abs(devy > 0 ? devy : 0);

        // Limit to frame boundaries
        xmin = std::max(0, xmin - dleft);
        ymin = std::max(0, ymin - dtop);
        xmax = std::min((int)frameSize.width - 1, xmax + dright);
        ymax = std::min((int)frameSize.height - 1, ymax + dbottom);

        // Make square
        if (square)
        {
            int sq_width = std::max(xmax - xmin + 1, ymax - ymin + 1);
            centerx = (xmin + xmax) / 2;
            centery = (ymin + ymax) / 2;
            xmin = centerx - ((sq_width - 1) / 2);
            ymin = centery - ((sq_width - 1) / 2);
            xmax = xmin + sq_width - 1;
            ymax = ymin + sq_width - 1;

            // Limit to frame boundaries
            xmin = xmin >= 0? xmin : 0;
            ymin = ymin >= 0? ymin : 0;
            xmax = xmax <= (int)frameSize.width - 1? xmax : (int)frameSize.width - 1;
            ymax = ymax <= (int)frameSize.height - 1? ymax : (int)frameSize.height - 1;
        }

        return cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax));
    }

    bool FaceSwap::preprocessImages(const cv::Mat& img, const cv::Mat& seg,
        std::vector<cv::Point>& landmarks, std::vector<cv::Point>& cropped_landmarks,
        cv::Mat& cropped_img, cv::Mat& cropped_seg, cv::Rect& bbox, bool init_track)
    {
#if DEBUG
        int start_ms, end_ms;
#endif
        // Calculate landmarks
        extract_landmarks(img, cropped_landmarks, init_track);
        if (cropped_landmarks.empty()) return false;


        // Calculate crop bounding box
        bbox = getFaceBBoxFromLandmarks(cropped_landmarks, img.size(), true);
        bbox.width = bbox.width / 4 * 4;    // Make sure cropped image is dividable by 4
        bbox.height = bbox.height / 4 * 4;

        // Crop landmarks
        for (cv::Point& p : cropped_landmarks)
        {
            p.x -= bbox.x;
            p.y -= bbox.y;
        }

        // Crop images
        cropped_img = img(bbox);
        if(!seg.empty()) cropped_seg = seg(bbox);

        return true;
    }

    bool FaceSwap::preprocessImages(const cv::Mat& img, const cv::Mat& seg,
        std::vector<cv::Point>& landmarks, std::vector<cv::Point>& cropped_landmarks,
        cv::Mat& cropped_img, cv::Mat& cropped_seg)
    {
        cv::Rect bbox;
        return preprocessImages(img, seg, landmarks, cropped_landmarks,
            cropped_img, cropped_seg, bbox);
    }

    void FaceSwap::generateTexture(const Mesh& mesh, const cv::Mat& img, 
        const cv::Mat& seg, const cv::Mat& vecR, const cv::Mat& vecT,
        const cv::Mat& K, cv::Mat& tex, cv::Mat& uv)
    {
        // Resize images to power of 2 size
        cv::Size tex_size(nextPow2(img.cols), nextPow2(img.rows));
        cv::Mat img_scaled, seg_scaled;
        cv::resize(img, img_scaled, tex_size, 0.0, 0.0, cv::INTER_CUBIC);
        if(!seg.empty())
            cv::resize(seg, seg_scaled, tex_size, 0.0, 0.0, cv::INTER_NEAREST);

        // Combine image and segmentation into one 4 channel texture
        if (!seg.empty())
        {
            std::vector<cv::Mat> channels;
            cv::split(img, channels);
            channels.push_back(seg);
            cv::merge(channels, tex);
        }
        else tex = img_scaled; 

        uv = generateTextureCoordinates(m_src_mesh, img.size(), vecR, vecT, K);
    }

    cv::Mat FaceSwap::generateTextureCoordinates(
        const Mesh& mesh,const cv::Size& img_size,
        const cv::Mat & vecR, const cv::Mat & vecT, const cv::Mat & K)
    {
        cv::Mat P = createPerspectiveProj3x4(vecR, vecT, K);
        cv::Mat pts_3d;
        cv::vconcat(mesh.vertices.t(), cv::Mat::ones(1, mesh.vertices.rows, CV_32F), pts_3d);
        cv::Mat proj = P * pts_3d;

        // Normalize projected points
        cv::Mat uv(mesh.vertices.rows, 2, CV_32F);
        float* uv_data = (float*)uv.data;
        float z;
        for (int i = 0; i < uv.rows; ++i)
        {
            z = proj.at<float>(2, i);
            *uv_data++ = proj.at<float>(0, i) / (z * img_size.width);
            *uv_data++ = proj.at<float>(1, i) / (z * img_size.height);
        }

        return uv;
    }

    cv::Mat FaceSwap::blend(const cv::Mat& src, const cv::Mat& dst, const cv::Mat& dst_seg)
    {
        // Calculate mask
        cv::Mat mask(src.size(), CV_8U);
        unsigned char* src_data = src.data;
        unsigned char* dst_seg_data = dst_seg.data;
        unsigned char* mask_data = mask.data;
        for (int i = 0; i < src.total(); ++i)
        {
            unsigned char cb = *src_data++;
            unsigned char cg = *src_data++;
            unsigned char cr = *src_data++;
            if (!(cb == 0 && cg == 0 && cr == 0))  *mask_data++ = 255;
            else *mask_data++ = 0;
        }

        // Combine the segmentation with the mask
        if (!dst_seg.empty())
            cv::bitwise_and(mask, dst_seg, mask);

        // Find center point
        int minc = std::numeric_limits<int>::max(), minr = std::numeric_limits<int>::max();
        int maxc = 0, maxr = 0;
        mask_data = mask.data;
        for (int r = 0; r < src.rows; ++r)
            for (int c = 0; c < src.cols; ++c)
            {
                if (*mask_data++ < 255) continue;
                minc = std::min(c, minc);
                minr = std::min(r, minr);
                maxc = std::max(c, maxc);
                maxr = std::max(r, maxr);
            }
        if (minc >= maxc || minr >= maxr) return cv::Mat();
        cv::Point center((minc + maxc) / 2, (minr + maxr) / 2);

        // Do blending
        cv::Mat blend;
        cv::seamlessClone(src, dst, mask, center, blend, cv::NORMAL_CLONE);

        return blend;
    }

    Mesh FaceSwap::getDstMesh() {
        return m_dst_mesh;
    }

    float FaceSwap::getK4() {
        return m_K.at<float>(4);
    }

    cv::Mat FaceSwap::getVecT() {
        return m_vecT;
    }

    cv::Mat FaceSwap::getVecR() {
        return m_vecR;
    }

    cv::Mat FaceSwap::getTgtCroppedImg() {
        return m_tgt_cropped_img;
    }

    cv::Mat FaceSwap::getTargetImg() {
        return m_target_img;
    }

    cv::Mat FaceSwap::getTargetSeg() {
        return m_target_seg;
    }

    cv::Rect FaceSwap::getTargetBbox() {
        return m_target_bbox;
    }

}   // namespace face_swap
