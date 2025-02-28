#ifndef CGCV_ALGORITHMS_H
#define CGCV_ALGORITHMS_H

#include <numeric>
#include <limits>
#include <opencv2/opencv.hpp>

class algorithms
{
   public:
    static void compute_grayscale(const cv::Mat &input_image, cv::Mat &grayscale_image);

    static void gaussian_blur(const cv::Mat &input_image, const int &kernel_size, 
                              const float &sigma, cv::Mat &blurred_image);

    static void compute_log_transform(const cv::Mat &blurred_image, cv::Mat &log_transform);

    static void apply_bilateral_filter(const cv::Mat &log_transform, cv::Mat &filtered_image);

    static void canny(const cv::Mat &filtered_image, const int threshold1, const int threshold2, cv::Mat &edges);

    static void apply_morph_operation(const cv::Mat &filled_image, const int kernel_size, 
                                      const cv::MorphTypes mode, cv::Mat &morphed_image);

    static void L2_distance_transform(const cv::Mat &source_image, cv::Mat &transformed_image);

    static void match_cracks(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &matching_frame, const int step, 
                             cv::Mat &minimal_distances, cv::Mat &mask_img1, cv::Mat &mask_img2);

    static void blend_originals(const cv::Mat &original_img1, const cv::Mat &original_img2, 
                                const cv::Mat &mask_img1, const cv::Mat &mask_img2, cv::Mat &blended_original);

    static void compute_gradient(const cv::Mat &source_image, cv::Mat &gradient_x, cv::Mat &gradient_y, cv::Mat &gradient_abs);

    static void non_maxima_suppression(const cv::Mat &gradient_image, const cv::Mat &gradient_x,
                                       const cv::Mat &gradient_y, cv::Mat &non_maxima);

    static void hysteresis(const cv::Mat &non_max_sup, const int threshold_min, const int threshold_max, cv::Mat &output_image);
};

#endif  // CGCV_ALGORITHMS_H
