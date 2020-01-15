#include <opencv2/opencv.hpp>
#include <filesystem>

using namespace std;

int main(int argc, char const *argv[]) { 
    if (argc<3) {
        cout<<"Usage: datset /path/to/plates/ /path/to/background/";
        return -1;
    }
    string plates_path = string(argv[1]);
    string backgr_path = string(argv[2]);
    string platp;
    for (const auto & entry : filesystem::directory_iterator(plates_path)) {
        auto ext = entry.path().filename().extension();
        if (ext == ".jpg" || ext ==".png") { 
            cout << entry.path().filename() << std::endl;
            platp = entry.path();
        }
    }
    cout<<platp<<endl;
    auto plate = cv::imread(platp, cv::IMREAD_UNCHANGED);
    auto back = cv::imread(backgr_path);
    array<cv::Point2f, 4> srcP;
    array<cv::Point2f, 4> dstP;
    srcP[0] = cv::Point2f(0.f,0.f);
    srcP[1] = cv::Point2f(plate.cols - 1.f, 0.f );
    srcP[2] = cv::Point2f(0.f, plate.rows - 1.f );
    srcP[3] = cv::Point2f(plate.cols-1, plate.rows - 1.f );
    
    dstP[0] = cv::Point2f(0, plate.rows*0.33f);
    dstP[1] = cv::Point2f(plate.cols-1, 0.f);
    dstP[2] = cv::Point2f(0, plate.rows*0.66);
    dstP[3] = cv::Point2f(plate.cols-1, plate.rows-1);
    cv::Mat warp = cv::Mat::zeros(plate.rows, plate.cols, plate.type());
    // auto warp_mat = cv::getAffineTransform( srcTri, dstTri );

    auto warp_mat = cv::getPerspectiveTransform(srcP, dstP);

    cv::warpPerspective(plate, warp, warp_mat, cv::Size(plate.cols, plate.rows));
    // cv::warpAffine(plate, warp,warp_mat,warp.size());
    cv::Mat backr, backra;
    cv::resize(back, backr, cv::Size(800, 800));
    cv::cvtColor(backr, backra, cv::COLOR_BGR2BGRA);
    // cv::imshow("back", backr);
    // cv::imshow("warp", warp);
    // cv::waitKey(0);
    // cv::Mat insetImage(backr, cv::Rect(0,0, plate.cols, plate.rows));
    // warp.copyTo(insetImage);
    warp.copyTo(backra(cv::Rect(0,0,warp.cols, warp.rows)));
    cv::imshow("res", backr);
    cv::waitKey(0);
    return 0;
}
