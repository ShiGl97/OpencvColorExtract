#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <core/core.hpp>
#include <highgui/highgui.hpp>   
#include <imgproc/imgproc.hpp>


using namespace cv;
using namespace std;

/**
 * @brief detectHSColor ��ȡͼ���о����ض���ɫ��Χ������ͼ����3 ͨ�� BGR ͼ��
 * @param image ����ͼ��Ҫ���� 3 ͨ�� BGR ͼ��
 * @param minHue  Hue ����Сֵ��Hue ��Χ 0-179 ��Hue�����Ǹ��Ƕȣ���0-360֮�䣬OpenCV �� 0-180 ��ʾ��0��ʾ��ɫ����
 * @param maxHue  Hue �����ֵ��Hue ��Χ 0-179
 * @param minSat Sat ����Сֵ��Sat ��Χ 0-255
 * @param maxSat Sat �����ֵ��Sat ��Χ 0-255
 * @param mask ��ȡ��������
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