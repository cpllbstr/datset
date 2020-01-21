#include <opencv2/opencv.hpp>
#include <filesystem>
#include <math.h>
#include <random>
#include <chrono>
#include <fstream>

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

list<cv::Mat> randomWarps(cv::Mat plate, int n) {
    list<cv::Mat> warps;

    random_device rd;  //Will be used to obtain a seed for the random number engine
    mt19937 gen(rd());
    uniform_real_distribution<> realgen(-1.f, 1.f);

    for (int i =0; i<n; i++) {
        gen.seed(rd());
        auto xangl = realgen(gen)*25.;
        auto yangl = realgen(gen)*25.;
        auto zangl = realgen(gen)*10.;
        cv::Mat platew, respl, grey, res;
        rotateImage(plate, platew, xangl, yangl,zangl, 0,0,300, 200);
        cv::cvtColor(platew, grey, cv::COLOR_BGR2GRAY);
        res = platew(cv::boundingRect(grey));
        warps.push_back(clearBg(res));
    }
    return warps; 
}

cv::Mat loadRandomImage(string path) {
    using fp = bool (*)(const filesystem::path&);
    auto nfiles = count_if(filesystem::directory_iterator(path), filesystem::directory_iterator{}, (fp)filesystem::is_regular_file);
    random_device rd;  
    mt19937 gen(rd());
    uniform_int_distribution<> intgen(0, nfiles-1);
    auto n = intgen(gen);
    int i=0;
    string s;
    for (const auto & file: filesystem::directory_iterator(path)) {
        i++;
        if (i == n) {
            s = file.path();
            break;
        }
    }
    auto back = cv::imread(s);
    return back;
}

void generateDatasetImages(string plate_path, string backgr_path, string output_folder, int image_size, double plate_scale, int nwarps){
    static int imagenum;

    filesystem::create_directory(output_folder);
    
    random_device rd;  //Will be used to obtain a seed for the random number engine
    mt19937 gen(rd());
    uniform_int_distribution<> intgen(1, image_size-plate_scale*image_size);
    auto plate = cv::imread(plate_path, cv::IMREAD_UNCHANGED);
    cv::Mat blured;
    int ksize = (plate.rows/2)%2==0 ? plate.rows/2+1 : plate.rows/2;
    cv::GaussianBlur(plate, blured, cv::Size(ksize,ksize), cv::BORDER_DEFAULT);
    auto wrps = randomWarps(blured, nwarps);

    for (auto& plt: wrps) {
        imagenum++;
        string name = output_folder+"/" + to_string(imagenum);
        ofstream file;
        file.open(name + ".txt");
        auto back = loadRandomImage(backgr_path);
        cv::Mat resplt,resbg;
        auto scale = plate_scale*image_size/plt.cols;
        gen.seed(rd());
        int x = intgen(gen);
        int y =  intgen(gen);
        int width = scale*plt.cols;
        int hight = scale*plt.rows;
        double yolo_x = double(x+width/2)/image_size;
        double yolo_y = double(y+hight/2)/image_size;
        double yolo_w = double(width)/image_size;
        double yolo_h = double(hight)/image_size;
        file <<"1 "<<yolo_x<<" "<<yolo_y<<" "<<yolo_w<<" "<<yolo_h<<endl;
        file.close();
        cv::resize(back, resbg, cv::Size(image_size,image_size));
        cv::resize(plt, resplt, cv::Size(scale*plt.cols, scale*plt.rows));
        overlayImage(resbg, resplt, cv::Point(x,y));
        cv::imwrite(name + ".png", resbg);
        // cv::line(resbg,cv::Point(image_size*yolo_x-image_size*yolo_w/2, image_size*yolo_y-image_size*yolo_h/2),cv::Point(image_size*yolo_x+image_size*yolo_w/2, image_size*yolo_y+image_size*yolo_h/2), cv::Scalar(0,0,255));
        // cv::imshow("res", resbg);
        // cv::waitKey(0);
    }
}

int main(int argc, char const *argv[]) {
    if (argc<6) {
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
            platp = entry.path();
            generateDatasetImages(platp, backgr_path, "./output",  image_size, plate_scale, 5);
        }
    }
    
    return 0;
}
