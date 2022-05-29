#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <core/core.hpp>
#include <highgui/highgui.hpp>   
#include <imgproc/imgproc.hpp>


using namespace cv;
using namespace std;

/**
 * @brief detectHSColor 提取图像中具有特定颜色范围的区域，图像是3 通道 BGR 图像
 * @param image 输入图像，要求是 3 通道 BGR 图像
 * @param minHue  Hue 的最小值，Hue 范围 0-179 （Hue本质是个角度，在0-360之间，OpenCV 用 0-180 表示。0表示红色。）
 * @param maxHue  Hue 的最大值，Hue 范围 0-179
 * @param minSat Sat 的最小值，Sat 范围 0-255
 * @param maxSat Sat 的最大值，Sat 范围 0-255
 * @param mask 提取出的区域
 */
void detectHSColor(const cv::Mat& image,
    double minHue, double maxHue,
    double minSat, double maxSat,
    cv::Mat& mask)
{
    cv::Mat hsv;
    cv::cvtColor(image, hsv,COLOR_BGR2HSV);
    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);
    cv::Mat mask1, mask2, hueMask;
    cv::threshold(channels[0], mask1, maxHue, 255, cv::THRESH_BINARY_INV);
    cv::threshold(channels[0], mask2, minHue, 255, cv::THRESH_BINARY);
    if (minHue < maxHue)
    {
        hueMask = mask1 & mask2;
    }
    else
    {
        hueMask = mask1 | mask2;
    }
    cv::Mat satMask;
    cv::inRange(channels[1], minSat, maxSat, satMask);
    mask = hueMask & satMask;
}

int main() {
    cv::Mat image = cv::imread("E:/OpenCV_image/50.jpg");

    cv::imshow("origin", image);
    cv::Mat mask;
    detectHSColor(image, 100, 124, 50, 255, mask);

    cv::Mat out;
    image.copyTo(out, mask);

    cv::imshow("out", out);
    cv::waitKey(0);
}