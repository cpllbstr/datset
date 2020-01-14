#include <opencv2/opencv.hpp>

using namespace std;

int main(int argc, char const *argv[])
{   
    string plates_path = string(argv[1]);
    string backgr_path = string(argv[2]);

    auto plate = cv::imread(plates_path);
    cv::Point2f srcTri[3];
    cv::Point2f dstTri[3];
    srcTri[0] = cv::Point2f( 0,0 );
    srcTri[1] = cv::Point2f(plate.cols - 1, 0 );
    srcTri[2] = cv::Point2f( 0, plate.rows - 1 );

    dstTri[0] = cv::Point2f( plate.cols*0.30, plate.rows*0.33 );
    dstTri[1] = cv::Point2f( plate.cols*0.4, plate.rows*0.25 );
    dstTri[2] = cv::Point2f( plate.cols*0.15, plate.rows*0.7 );

    cv::Mat warp = cv::Mat::zeros(plate.rows, plate.cols, plate.type());
    auto warp_mat = cv::getAffineTransform( srcTri, dstTri );

    cv::warpAffine(plate, warp,warp_mat,warp.size());
    
    cv::imshow("warp", warp);
    cv::waitKey(0);
    return 0;
}
