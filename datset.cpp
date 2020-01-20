#include <opencv2/opencv.hpp>
#include <filesystem>
#include <math.h>
#include <random>
#include <chrono>

using namespace std;

void overlayImage(cv::Mat &src, cv::Mat &overlay, const cv::Point& location) {
    for (int y = max(location.y, 0); y < src.rows; ++y)
    {
        int fY = y - location.y;

        if (fY >= overlay.rows)
            break;

        for (int x = max(location.x, 0); x < src.cols; ++x)
        {
            int fX = x - location.x;

            if (fX >= overlay.cols)
                break;

            double opacity = ((double)overlay.data[fY * overlay.step + fX * overlay.channels() + 3]) / 255;

            for (int c = 0; opacity > 0 && c < src.channels(); ++c)
            {
                unsigned char overlayPx = overlay.data[fY * overlay.step + fX * overlay.channels() + c];
                unsigned char srcPx = src.data[y * src.step + x * src.channels() + c];
                src.data[y * src.step + src.channels() * x + c] = srcPx * (1. - opacity) + overlayPx * opacity;
            }
        }
    }
}

cv::Mat clearBg(cv::Mat plate) {
    cv::Mat tmp, alpha, dst;
    cv::cvtColor(plate,tmp,cv::COLOR_BGR2GRAY);
    cv::threshold(tmp,alpha,1,255,cv::THRESH_BINARY);
    cv::Mat rgb[3];
    cv::split(plate,rgb);
    cv::Mat rgba[4]={rgb[0],rgb[1],rgb[2],alpha};
    cv::merge(rgba,4,dst);
    return dst;
}

void rotateImage(const cv::Mat &input,cv::Mat &output, double alpha, double beta, double gamma, double dx, double dy, double dz, double f) {
    alpha = (alpha)*CV_PI/180.;
    beta = (beta)*CV_PI/180.;
    gamma = (gamma)*CV_PI/180.;
    // get width and height for ease of use in cv::Matrices
    double w = (double)input.cols;
    double h = (double)input.rows;
    // Projection 2D -> 3D cv::Matrix
    cv::Mat A1 = (cv::Mat_<double>(4,3) <<
              1, 0, -w/2.,
              0, 1, -h/2.,
              0, 0, 0,
              0, 0, 1);
    // Rotation cv::Matrices around the X, Y, and Z axis
    cv::Mat RX = (cv::Mat_<double>(4, 4) <<
              1,          0,           0, 0,
              0, cos(alpha), -sin(alpha), 0,
              0, sin(alpha),  cos(alpha), 0,
              0,          0,           0, 1);
    cv::Mat RY = (cv::Mat_<double>(4, 4) <<
              cos(beta), 0, -sin(beta), 0,
              0, 1,          0, 0,
              sin(beta), 0,  cos(beta), 0,
              0, 0,          0, 1);
    cv::Mat RZ = (cv::Mat_<double>(4, 4) <<
              cos(gamma), -sin(gamma), 0, 0,
              sin(gamma),  cos(gamma), 0, 0,
              0,          0,           1, 0,
              0,          0,           0, 1);
    // Composed rotation cv::Matrix with (RX, RY, RZ)
    cv::Mat R = RX * RY * RZ;
    // Translation cv::Matrix
    cv::Mat T = (cv::Mat_<double>(4, 4) <<
             1, 0, 0, dx,
             0, 1, 0, dy,
             0, 0, 1, dz,
             0, 0, 0, 1);
    // 3D -> 2D cv::Matrix
    cv::Mat A2 = (cv::Mat_<double>(3,4) <<
              f, 0, w/2, 0,
              0, f, h/2, 0,
              0, 0,   1, 0);
    // Final transforcv::Mation cv::Matrix
    cv::Mat trans = A2 * (T * (R * A1));
    // Apply cv::Matrix transforcv::Mation
    cv::Mat bordered;
    cv::copyMakeBorder(input, bordered, input.rows, input.rows, input.cols*.3, input.cols*.3, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
    cv::warpPerspective(bordered, output, trans, bordered.size(), cv::INTER_LANCZOS4);
}

// void autoCrop(cv::Mat &orig) {
    // cv::Mat grey, img;
    // cv::cvtColor(orig, grey, cv::COLOR_BGR2GRAY);
    // orig = orig(cv::boundingRect(grey));
// }



list<cv::Mat> randomWarps(cv::Mat plate, int n) {
    list<cv::Mat> warps;

    random_device rd;  //Will be used to obtain a seed for the random number engine
    mt19937 gen(rd());
    uniform_real_distribution<> realgen(-1.f, 1.f);

    for (int i =0; i<n; i++) {
        gen.seed(rd());
        auto xangl = realgen(gen)*25.;
        gen.seed(rd());
        auto yangl = realgen(gen)*25.;
        gen.seed(rd());
        auto zangl = realgen(gen)*10.;
        cv::Mat platew, respl, grey, res;
        rotateImage(plate, platew, xangl, yangl,zangl, 0,0,300, 200);
        cv::cvtColor(platew, grey, cv::COLOR_BGR2GRAY);
        res = platew(cv::boundingRect(grey));
        warps.push_back(clearBg(res));
    }
    return warps; 
}



int main(int argc, char const *argv[]) {
    if (argc<5) {
        cout<<"Usage: datset /path/to/plates/ /path/to/background/ output_image_size scale_factor\n";
        return -1;
    }
    string plates_path = string(argv[1]);
    string backgr_path = string(argv[2]);
    int image_size = atoi(argv[3]);
    double plate_scale = atof(argv[4]); 
    string platp;
    for (const auto & entry : filesystem::directory_iterator(plates_path)) {
        auto ext = entry.path().filename().extension();
        if (ext == ".jpg" || ext ==".png") { 
            // cout << entry.path().filename() << std::endl;
            platp = entry.path();
        }
    }

    random_device rd;  //Will be used to obtain a seed for the random number engine
    mt19937 gen(rd());
    uniform_int_distribution<> intgen(1, image_size-plate_scale*image_size);
    
    auto plate = cv::imread(platp, cv::IMREAD_UNCHANGED);
    auto back = cv::imread(backgr_path);
    auto wrps = randomWarps(plate, 10);

    for (auto& plt: wrps) {
        cv::Mat resbg;
        cv::Mat resplt;
        auto scale = plate_scale*image_size/plt.cols;
        cv::resize(back, resbg, cv::Size(image_size,image_size));
        cv::resize(plt, resplt, cv::Size(scale*plt.cols, scale*plt.rows));
        gen.seed(rd());
        overlayImage(resbg, resplt, cv::Point(intgen(gen), intgen(gen)));
        cv::imshow("res", resbg);
        cv::waitKey(0);
    }
    return 0;
}
