#include "algorithms.h"

//========================================================================================
// compute_grayscale()
//----------------------------------------------------------------------------------------
// TODO:  - iterate over all pixels in the image.
//        - multiply [R,G,B] with [0.2989, 0.5870, 0.1140] and accumulate.
//
// hints: - use cv::Vec3b type to access the color values of a 3-channel image.
//        - be aware that OpenCV treats matrix accesses in row-major order;
//          iterate through rows, then columns!
//
// parameters:
//  - input_image: [CV_8UC3] input image for the grayscale calculation
//  - grayscale_image: [CV_8UC1] output grayscale image
//
// return: void
//========================================================================================
void algorithms::compute_grayscale(const cv::Mat &input_image, cv::Mat &grayscale_image) {
    using namespace cv;

    constexpr double R_FAC = 0.2989;
    constexpr double G_FAC = 0.5870;
    constexpr double B_FAC = 0.1140;

    grayscale_image = Mat::zeros(input_image.size(), CV_8UC1);

    for (int row = 0; row < input_image.rows; row++) {
        for (int col = 0; col < input_image.cols; col++) {
            uchar const b = input_image.at<Vec3b>(row, col)[0];
            uchar const g = input_image.at<Vec3b>(row, col)[1];
            uchar const r = input_image.at<Vec3b>(row, col)[2];

            const auto grey = r * R_FAC + g * G_FAC + b * B_FAC;

            grayscale_image.at<uchar>(row, col) = static_cast<uchar>(grey);
        }
    }
}


//========================================================================================
// gaussian_blur()
// ---------------------------------------------------------------------------------------
// TODO:  - get a [k_s x 1] gaussian filter
//        - obtain a [k_s, k_s] with an outer product
//        - convolve the image with this kernel
//
// hints: - use -1 for the ddepth parameter to keep the depth the same
//
// parameters:
//  - input_image: [CV_8UC1] input grayscale image
//  - kernel_size: size (x- and y-direction) of the filter kernel
//  - sigma: variance of the filter kernel, controlling the smoothness
//  - blurred_image: [CV_8UC1] output blurred grayscale image
//
// return: void
//========================================================================================
void algorithms::gaussian_blur(const cv::Mat &input_image, const int &kernel_size,
                               const float &sigma, cv::Mat &blurred_image) {
    using namespace cv;

    const Mat one_d_kernel = getGaussianKernel(kernel_size, sigma);
    Mat kernel;
    mulTransposed(one_d_kernel, kernel, false);

    filter2D(input_image, blurred_image, -1, kernel);
}


//========================================================================================
// compute_log_transform()
//----------------------------------------------------------------------------------------
// TODO:  - apply the log function to blurred_image
//        - scale the output to respective format
//
// hints: - use cv::log() to apply the logarithm transformation
//        - use cv::minMaxIdx() to determine normalization factor
//        - apply 1 to the input image for numerical stability
//
// parameters:
//  - blurred_image: [CV_8UC1] Input grayscale image (blurred image)
//  - log_transform: [CV_8UC1] Output log-transformed grayscale image
//
// return: void
//========================================================================================
void algorithms::compute_log_transform(const cv::Mat &blurred_image, cv::Mat &log_transform) {
    using namespace cv;

    const Mat x = blurred_image.clone();

    Mat log_x;
    log(x + 1, log_x);

    double min;
    double max;

    minMaxIdx(blurred_image, &min, &max);

    const auto log_max = log(1 + max);

    const Mat y_of_x = (log_x / log_max) * 255;
    log_transform = y_of_x.clone();
}


//========================================================================================
// apply_bilateral_filter()
//----------------------------------------------------------------------------------------
// TODO:  - iterate through each pixel in the image
//        - for each pixel, consider a neighborhood defined by the filter size d
//        - compute spatial Gaussian weight -> use Euclidean distance
//        - compute range Gaussian weight -> intensity differences
//        - multiply neighbor pixel intensity with combined weights
//        - normalize the intensity 
//        - assign computed intensity to output image if weight > 0
//
// hints: - use exp() for computing the Gaussian function
//        - ensure boundary checks for the kernel filtering
//
// parameters:
//  - log_transform: [CV_8UC1] Input log-transformed grayscale image
//  - filtered_image: [CV_8UC1] Output bilateral filtered grayscale image
//
// return: void
//========================================================================================
void algorithms::apply_bilateral_filter(const cv::Mat &log_transform, cv::Mat &filtered_image) {
    const int d = 5;
    const int sigma_s = 10;
    const int sigma_r = 10;
    // Do not change the constants above!

    using namespace cv;

    constexpr auto radius = d / 2;

    Mat bordered_image;
    copyMakeBorder(log_transform, bordered_image, radius, radius, radius, radius, BORDER_CONSTANT, Scalar(0, 0, 0));

    Mat output_image = log_transform.clone();

    for (int m = 0; m < log_transform.rows; m++) {
        for (int n = 0; n < log_transform.cols; n++) {
            float w_mn = 0;
            float grf_sum = 0;

            const uchar f_mn = bordered_image.at<uchar>(m + radius, n + radius);

            for (int k = -radius; k <= radius; k++) {
                for (int l = -radius; l <= radius; l++) {
                    const int mk_sq = k * k;
                    const int nl_sq = l * l;
                    constexpr int sigma_s_sq = sigma_s * sigma_s;

                    const int g_in_brackets = (mk_sq + nl_sq) / (2 * sigma_s_sq);
                    const float g_exp = exp(-g_in_brackets);

                    const uchar f_kl = bordered_image.at<uchar>(m + k + radius, n + l + radius);
                    const int diff = f_mn - f_kl;
                    const int f_mn_kl_sq = diff * diff;
                    constexpr int sigma_r_sq = sigma_r * sigma_r;

                    const int r_in_brackets = f_mn_kl_sq / (2 * sigma_r_sq);
                    const float r_exp = exp(-r_in_brackets);

                    w_mn += g_exp * r_exp;
                    grf_sum += g_exp * r_exp * f_kl;
                }
            }
            if (w_mn > 0.0) {
                float h_mn = grf_sum / w_mn;
                output_image.at<uchar>(m, n) = static_cast<uchar>(std::clamp(h_mn, 0.0f, 255.0f));
            } else {
                output_image.at<uchar>(m, n) = f_mn;
            }
        }
    }
    filtered_image = output_image.clone();
}


//========================================================================================
// canny()
//----------------------------------------------------------------------------------------
// TODO:  - apply the Canny algorithm to find relevant edges in the given image
//
// parameters:
//  - filtered_image: [CV_8UC1] filtered image used as input for the edge detection
//  - weak_edge_threshold: minimal gradient value to classify a weak edge
//  - strong_edge_threshold: minimal gradient value to classify a strong edge
//  - edges: [CV_8UC1] image containing the detected edges
// 
// return: void
//========================================================================================
void algorithms::canny(const cv::Mat &filtered_image, const int weak_edge_threshold,
                       const int strong_edge_threshold, cv::Mat &edges) {
    cv::Canny(filtered_image, edges, weak_edge_threshold, strong_edge_threshold);
}


//========================================================================================
// apply_morph_operation()
//----------------------------------------------------------------------------------------
// TODO:  - apply a morphological operation (either erosion or dilation) to the
//          given image using the specified kernel size and mode
//        - implement the logic behind erosion and dilation according to the
//          assignment sheet
//        - store the result in 'morphed_image'
//
// hints: - the kernel size represents the area around each pixel in the image
//          considered during the morpholocial operation; 
//          the kernel size is given as an odd number to ensure proper centering
//
// parameters:
//  - morph_input: [CV_8UC1] source image that requires a morphological operation
//  - kernel_size: [int] size of the (square-shaped) kernel
//  - mode: [cv::MorphTypes] type of morphological operation; this will either
//          be MORPH_ERODE or MORPH_DILATE
//  - morphed_image: [CV_8UC1] output image after applied morphological operation;
//                   either dilated or eroded image, depending on mode
//
// return: void
//========================================================================================
void algorithms::apply_morph_operation(const cv::Mat &morph_input, const int kernel_size,
                                       const cv::MorphTypes mode, cv::Mat &morphed_image) {
    using namespace cv;

    const int radius = kernel_size / 2;

    const Mat input_image = morph_input.clone();
    Mat output_image = input_image.clone();

    Mat bordered_image;
    copyMakeBorder(morph_input, bordered_image, radius, radius, radius, radius, BORDER_REFLECT);

    if (mode != MORPH_ERODE && mode != MORPH_DILATE) return;

    for (int row = 0; row < input_image.rows; row++) {
        for (int col = 0; col < input_image.cols; col++) {
            constexpr int black = 0;
            constexpr int white = 255;

            int new_center_pixel_color = (mode == MORPH_ERODE) ? white : black;

            for (int i = -radius; i <= radius; i++) {
                for (int j = -radius; j <= radius; j++) {
                    const auto kernel_pixel = bordered_image.at<uchar>(row + i + radius, col + j + radius);
                    if (mode == MORPH_ERODE) {
                        if (kernel_pixel == black) {
                            new_center_pixel_color = black;
                            break;
                        }
                    } else {
                        if (kernel_pixel == white) {
                            new_center_pixel_color = white;
                            break;
                        }
                    }
                }
            }

            output_image.at<uchar>(row, col) = new_center_pixel_color;
        }
    }
    output_image.copyTo(morphed_image);
}


//========================================================================================
// L2_distance_transform()
//----------------------------------------------------------------------------------------
// TODO:  - iterate through each pixel in the binary input image
//        - for every white pixel (foreground), find the minimum Euclidean distance 
//          to the nearest black pixel (background) in the neighborhood window
//        - use a search kernel of size (5x5) centered on each white pixel
//        - if a black pixel is found in the neighborhood, compute its Euclidean distance 
//          and update the minimum distance
//        - store the result in 'transformed_image' 
//
// hints: - in a binary image, white pixels have an intensity value of 255, 
//          while black pixels have an intensity value of 0
//        - the search kernel size represents the area around each pixel in the image
//          considered during the distance transform
//        - ensure proper boundary checking when iterating through the neighborhood window
//        - start with a large initial distance (FLT_MAX) and update as needed
//        - OpenCV automatically handles the mapping of float values to uint8_t
//
// parameters:
//  - source_image: [CV_8UC1] binary source image 
//  - transformed_image: [CV_8UC1] distance transformed image
//
// Return: void
//========================================================================================
void algorithms::L2_distance_transform(const cv::Mat &source_image, cv::Mat &transformed_image) {
    using namespace cv;

    constexpr int kernel_size = 5;
    constexpr int radius = kernel_size / 2;
    constexpr int black = 0;
    constexpr int white = 255;

    Mat input_image = source_image.clone();
    Mat output_image = Mat::zeros(source_image.size(), CV_8UC1);

    Mat border_image;
    copyMakeBorder(source_image, border_image, radius, radius, radius, radius, BORDER_REFLECT);

    auto compute_px_value = [&](const int row, const int col)-> uchar {
        float min_dist = FLT_MAX;

        for (int i = -radius; i <= radius; i++) {
            for (int j = -radius; j <= radius; j++) {

                if (border_image.at<uchar>(row + i + radius, col + j + radius) != black) continue;

                const int dx = i;
                const int dy = j;

                float d = sqrt(dx * dx + dy * dy);

                min_dist = min(min_dist, d);
            }
        }

        min_dist = (min_dist == FLT_MAX) ? 0 : min(min_dist, 255.0f);
        return min_dist;
    };

    for (int row = 0; row < input_image.rows; row++) {
        for (int col = 0; col < input_image.cols; col++) {
            if (const int px = input_image.at<uchar>(row, col); px != white) continue;

            output_image.at<uchar>(row, col) = compute_px_value(row, col);
        }
    }
    output_image.copyTo(transformed_image);
}


//========================================================================================
// match_cracks()
//----------------------------------------------------------------------------------------
// TODO:  - place img1 at the center of the matching frame
//        - iterate through the matching frame in step sizes over rows and 
//          columns to brute-force the best alignment spots
//        - keep img1 always at the center of your matching frame, 
//          add img2 according to the row/column of the iteration
//        - compute the L2 distance matrix using your implementation 
//          of L2_distance_transform() and sum its values to determine 
//          the total distance
//        - identify the position of img2 with the minimum overall distance;
//          store the according distance matrix in minimal_distances

//        - create a white mask at the position of img1 (mask_img1)
//        - create a white mask at the position of img2 (mask_img2)
//
// hints: - use copyTo() together with cv::Rect() to copy sub-matrices 
//          into specific regions of empty matrices based on their 
//          starting point and column-/row-count
//
// parameters:
//  - img1: [CV_8UC1] first morph transformed input image 
//  - img2: [CV_8UC1] second morph transformed input image
//  - matching_frame [CV_8UC1] all-black matrix acting as a frame 
//                             for the matching process in an extended image 
//  - step: [int] step size for traversing the extended image
//  - minimal_distances: [CV_8UC1] output matrix storing the minimum distances
//  - mask_img1: [CV_8UC1] the mask for image 1
//  - mask_img2: [CV_8UC1] the mask for image 2
//
// return: void
//========================================================================================
void algorithms::match_cracks(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &matching_frame, const int step,
                              cv::Mat &minimal_distances, cv::Mat &mask_img1, cv::Mat &mask_img2) {
}


//========================================================================================
// blend_originals()
//----------------------------------------------------------------------------------------
// TODO:  - replace white rectangular region in mask_img1 with content from original_img1
//        - replace white rectangular region in mask_img2 with content from original_img2
//        - merge both modified images and store result in blended_original
//        - construct blended_original by first incorporating pixel intensities 
//          from original_img1, followed by those from original_img2
//
// hints: - the provided masks are binary images, so the white regions 
//          can easily be differentiated with their pixel intensity
//
// parameters:
//  - original_img1: [CV_8UC3] first input image 
//  - original_img2: [CV_8UC3] second input image 
//  - mask_img1: [CV_8UC1] binary mask indicating the region 
//                         to be replaced with original_img1
//  - mask_img2: [CV_8UC1] binary mask indicating the region 
//                         to be replaced with original_img2
//  - blended_original: [CV_8UC3] output image containing the merged result of both images
//
// return: void
//========================================================================================
void algorithms::blend_originals(const cv::Mat &original_img1, const cv::Mat &original_img2,
                                 const cv::Mat &mask_img1, const cv::Mat &mask_img2, cv::Mat &blended_original) {
}


//========================================================================================
// BONUS: compute_gradient()
//----------------------------------------------------------------------------------------
// TODO:  - compute the 1st order Sobel derivative in x and y direction
//        - set the parameter ddepth to CV_32F
//        - compute the per-pixel l2-norm
//
// hint:  - use the arithmetic operations (pow, add, sqrt, ...) of OpenCV
//
// parameters:
//  - source_image: [CV_8UC1] smoothed and filtered image for the gradient calculation
//  - gradient_x: [CV_32FC1] output image for the gradient in x direction
//  - gradient_y: [CV_32FC1] output image for the gradient in y direction
//  - gradient_abs: [CV_32FC1] output image for the gradient norm
//
// return: void
//========================================================================================
void algorithms::compute_gradient(const cv::Mat &source_image, cv::Mat &gradient_x,
                                  cv::Mat &gradient_y, cv::Mat &gradient_abs) {
}


//========================================================================================
// BONUS: non_maxima_suppression()
//----------------------------------------------------------------------------------------
// TODO:
//  - compute all angles and transform to [0, 180]
//  - depending on the gradient direction of the pixel classify each pixel P in one
//    of the following classes:
//    ____________________________________________________________________________
//    | class |direction                | corresponding pixels Q, R               |
//    |-------|-------------------------|-----------------------------------------|
//    | I     | beta <= 22.5            | Q: same row (y), left column (x-1)      |
//    |       |   or beta > 157.5       | R: same row (y), right column (x+1)     |
//    |-------|-------------------------|-----------------------------------------|
//    | II    | 22.5 < beta <= 67.5     | Q: row above (y-1), left column (x-1)   |
//    |       |                         | R: row below (y+1), right column (x+1)  |
//    |-------|-------------------------|-----------------------------------------|
//    | III   | 67.5 < beta <= 112.5    | Q: row above (y-1), same column (x)     |
//    |       |                         | R: row below (y+1), same column (x)     |
//    |-------|-------------------------|-----------------------------------------|
//    | IV    | 112.5 < beta <= 157.5   | Q: row below (y+1), left column (x-1)   |
//    |       |                         | R: row above (y-1), right column (x+1)  |
//    |_______|_________________________|_________________________________________|
//
//  - compare the value of P with the values of Q and R:
//    If Q or R are greater than P -> set P to 0
//
// parameters:
//  - gradient_image: [CV_32FC1] matrix with the gradient image
//  - gradient_x: [CV_32FC1] matrix with the gradient in x direction
//  - gradient_y: [CV_32FC1] matrix with the gradient in y direction
//  - non_max_sup: [CV_32FC1] output matrix for the non maxima suppression
//
// return: void
//========================================================================================
void algorithms::non_maxima_suppression(const cv::Mat &gradient_image, const cv::Mat &gradient_x,
                                        const cv::Mat &gradient_y, cv::Mat &non_max_sup) {
    float RAD2DEG = (180.0 / CV_PI);
}


//========================================================================================
// BONUS: hysteresis()
//----------------------------------------------------------------------------------------
// TODO:
//  - set all pixels under the lower threshold to 0
//  - set all pixels over the high threshold to 255
//  - classify all weak edges (threshold_min <= weak edge < threshold_max)
//    - if one of the 8 surrounding pixel values is higher than threshold_max,
//      also the weak pixel is a strong pixel
//    - check this recursively to be sure not to miss one
//  - set all remaining, non-classified pixels to 0
//
// parameters:
//  - non_max_sup: [CV_8UC1] input image containing the result of the NMS
//  - threshold_min: the lower threshold
//  - threshold_max: the upper threshold
//  - output_image: [CV_8UC1] output image for results of the hysteresis
//
// return: void
//========================================================================================
void algorithms::hysteresis(const cv::Mat &non_max_sup, const int threshold_min,
                            const int threshold_max, cv::Mat &output_image) {
}
