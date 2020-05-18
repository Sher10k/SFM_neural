// /home/roman/SFM/1_trajectory_test/desktop_tracks.txt 1914 640 360
// /home/roman/SFM/1_trajectory_test/backyard_tracks.txt 1914 640 360
// /home/roman/SFM/data/ 720 512 360    // My
// /home/roman/SFM/data/ 350 240 360    // Sagrada Familia
// /home/roman/SFM/data/ 800 400 225    // кусок камня 

// STD
#include <iostream>
#include <fstream>
#include <vector>

// Eigen
//#include <Eigen/Eigen>

// Open3D
#include <Open3D/Open3D.h>

// CV
#define CERES_FOUND 1
#define OPENCV_TRAITS_ENABLE_DEPRECATED
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>    // imread()
#include <opencv2/highgui.hpp>      // imshow()
#include <opencv2/imgproc.hpp>      // cvtColor()
#include <opencv2/sfm.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/viz.hpp>

using namespace std;
using namespace cv;
using namespace cv::sfm;
using namespace cv::xfeatures2d;
using namespace Eigen;
using namespace open3d;

#define EXAMPLE 0

static void help() {
  cout
      << "\n------------------------------------------------------------------\n"
      << " This program shows the camera trajectory reconstruction capabilities\n"
      << " in the OpenCV Structure From Motion (SFM) module.\n"
      << " \n"
      << " Usage:\n"
      << "        example_sfm_trajectory_reconstruction <path_to_tracks_file> <f> <cx> <cy>\n"
      << " where: is the tracks file absolute path into your system. \n"
      << " \n"
      << "        The file must have the following format: \n"
      << "        row1 : x1 y1 x2 y2 ... x36 y36 for track 1\n"
      << "        row2 : x1 y1 x2 y2 ... x36 y36 for track 2\n"
      << "        etc\n"
      << " \n"
      << "        i.e. a row gives the 2D measured position of a point as it is tracked\n"
      << "        through frames 1 to 36.  If there is no match found in a view then x\n"
      << "        and y are -1.\n"
      << " \n"
      << "        Each row corresponds to a different point.\n"
      << " \n"
      << "        f  is the focal lenght in pixels. \n"
      << "        cx is the image principal point x coordinates in pixels. \n"
      << "        cy is the image principal point y coordinates in pixels. \n"
      << "------------------------------------------------------------------\n\n"
      << endl;
}


/* Build the following structure data
 *
 *            frame1           frame2           frameN
 *  track1 | (x11,y11) | -> | (x12,y12) | -> | (x1N,y1N) |
 *  track2 | (x21,y11) | -> | (x22,y22) | -> | (x2N,y2N) |
 *  trackN | (xN1,yN1) | -> | (xN2,yN2) | -> | (xNN,yNN) |
 *
 *
 *  In case a marker (x,y) does not appear in a frame its
 *  values will be (-1,-1).
 */

#if EXAMPLE == 0
struct Track
{
    Track()
    {
        begin = -1;
        now = -1;
        trainIdx = -1;
        end = -1;
    }
    Track( Point2f _p1, Point2f _p2, int _b = -1, int _n = -1, int _ti = -1, int _e = -1 )
        : begin(_b), now(_n), trainIdx(_ti), end(_e)
    {
        points.push_back( _p1 );
        points.push_back( _p2 );
    }
    
    int begin;
    int now;
    int trainIdx;
    int end;
    vector< Point2f > points;
};

void matches_2_tracks( const vector< vector< KeyPoint >> &_keypoints, 
                       vector< vector< DMatch >> _matches,
                       std::vector<Mat> &points2d )
{
    size_t n_frames = _keypoints.size();
    size_t n_tracks = 0;
    size_t sum = 0;
    
    //vector< vector< Vec2d > > tracks;
    vector< Track > tracks;
    
    for ( size_t i = 0; i < _matches.size(); i++ )
    {
        sum += _matches.at(i).size();
        if ( _matches.at(i).size() ) 
        {
            if ( !tracks.empty() )
            {
                for ( Track &k : tracks )
                {
                    for ( auto &j : _matches.at(i) )
                    {
                            // Проверяем соответствия между треками и связями
                        if ( (j.imgIdx >= 0) &&             // --- возможно можно просто строго >
                             (k.end == int(i)) && 
                             (j.queryIdx == k.trainIdx) )
                        {
                            k.points.push_back( _keypoints.at(i+1).at( size_t(j.trainIdx) ).pt );
                            k.trainIdx = j.trainIdx;
                            k.now++;
                            k.end++;
                            j.imgIdx = -1;
                        }
                    }
                        // Треки которым не нашлось продолжения
                    if ( k.now == int(i) )  
                    {
                        k.points.push_back( Point2f(-1, -1) );
                        k.now++;
                    }
                }
                    // Добавляем новые треки
                for ( auto &j : _matches.at(i) )
                {
                    if ( j.imgIdx > 0)
                    {
                        tracks.push_back( Track( _keypoints.at(i).at( size_t(j.queryIdx) ).pt,
                                                 _keypoints.at(i+1).at( size_t(j.trainIdx) ).pt,
                                                 int(i), int(i+1), j.trainIdx, int(i+1) ) );
                        j.imgIdx = -1;
                    }
                }
            }
            else    // --- если первая итерация или каждром ранне не было ключевых точек
            {
                for ( auto &j : _matches.at(i) )
                {
                    tracks.push_back( Track( _keypoints.at(i).at( size_t(j.queryIdx) ).pt,
                                             _keypoints.at(i+1).at( size_t(j.trainIdx) ).pt,
                                             int(i), int(i+1), j.trainIdx, int(i+1) ) );
                    j.imgIdx = -1;
                }
            }
        }
        else
        {
            for ( Track &k : tracks )
            {
                k.points.push_back( Point2f(-1, -1) );
                k.now++;
            }
        }
    }
    n_tracks = tracks.size();
    
    
    vector< vector< Vec2d > > tracksV;
    for ( size_t j = 0; j < n_tracks; j++ )
    {
        vector< Vec2d > track;
        for ( size_t i = 0; i < n_frames; i++ )
        {
            if ( tracks.at(j).begin <= int(i) )
            {
                float x = tracks.at(j).points.at( i - size_t(tracks.at(j).begin) ).x;
                float y = tracks.at(j).points.at( i - size_t(tracks.at(j).begin) ).y;
                if ( (x > 0) && (y > 0) )
                  track.push_back( Vec2d( double(x), double(y) ) );
                else
                  track.push_back( Vec2d(-1,-1) );
            }
            else
                track.push_back( Vec2d(-1,-1) );
        }
        tracksV.push_back(track);
    }
    // --- Save 3d points to file
    std::ofstream out;
    out.open("3d_cloud.txt");
    for ( auto track : tracksV )
    {
        for ( auto point : track )
            out << point;
        out << "\n";
    }
    out.close();
    
    for ( size_t i = 0; i < n_frames; ++i )
    {
        Mat_< double > frame( 2, int(n_tracks) );
        
        for ( size_t j = 0; j < n_tracks; ++j )
        {
            frame(0,int(j)) = tracksV[j][i][0];
            frame(1,int(j)) = tracksV[j][i][1];
        }
        points2d.push_back( Mat( frame ) );
    }
}

vector< DMatch > filtration_matches( const vector< DMatch > &_matches, float _threshold = -1 )
{
    // --- Find min and max value of match distance
    float max_dist = 0, min_dist = 1000;
    for ( auto j : _matches )
    {
        float dist = j.distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }
    
    // --- Filtration by threshold
    vector< DMatch > tempMatches;
    float threshold;
    if ( _threshold < 0 ) threshold = 10 * min_dist;
    else threshold = _threshold;
    for ( auto j : _matches )
        if ( j.distance < threshold )
            tempMatches.push_back( j );
    
    return tempMatches;
}

#else
// --- Example --- //
static void parser_2D_tracks(const String &_filename, std::vector<Mat> &points2d )
{
  ifstream myfile(_filename.c_str());

  if (!myfile.is_open())
  {
    cout << "Unable to read file: " << _filename << endl;
    exit(0);

  } else {

    double x, y;
    string line_str;
    int n_frames = 0, n_tracks = 0;

    // extract data from text file

    vector<vector<Vec2d> > tracks;
    for ( ; getline(myfile,line_str); ++n_tracks)
    {
      istringstream line(line_str);

      vector<Vec2d> track;
      for ( n_frames = 0; line >> x >> y; ++n_frames)
      {
        if ( x > 0 && y > 0)
          track.push_back(Vec2d(x,y));
        else
          track.push_back(Vec2d(-1));
      }
      tracks.push_back(track);
    }

    // embed data in reconstruction api format

    for (size_t i = 0; i < size_t(n_frames); ++i)
    {
      Mat_<double> frame(2, n_tracks);

      for (size_t j = 0; j < size_t(n_tracks); ++j)
      {
        frame(0,int(j)) = tracks[j][i][0];
        frame(1,int(j)) = tracks[j][i][1];
      }
      points2d.push_back(Mat(frame));
    }

    myfile.close();
  }

}
bool camera_pov = false;
static void keyboard_callback(const viz::KeyboardEvent &event, void* cookie)
{
  if ( event.action == 0 &&!event.symbol.compare("s") )
    camera_pov = !camera_pov;
}
#endif

// --- --- MAIN --- --- ---------------------------------------------------- //
int main(int argc, char** argv)
{
    //std::cout << cv::getBuildInformation() << std::endl;
    
    // --- Checking number of input parameters
    if ( argc != 5 )
    {
        help();
        exit(0);
    }
    
#if EXAMPLE == 0
    // --- Read input parameters --- and temp parameters --- !!!
    string path = argv[1];
    bool filtrationMask = true;
    bool filtrationMatches = true;
    
        // Read path of folder with img files
    cout << " --- Read path of folder with img files ... ";
    vector< string > imgPath[4];
//    glob( path + "depth_img_bgr/*.png", imgPath[0] );
//    glob( path + "rgb_img_bgr/*.png", imgPath[1] );
//    glob( path + "bin_mask/*.png", imgPath[2] );
//    glob( path + "sem_img_bgr/*.png", imgPath[3] );

//    glob( path + "depth_img_temp/*.png", imgPath[0] );
    glob( path + "rgb_img_temp/*.png", imgPath[1] );
    glob( path + "bin_mask_temp/*.png", imgPath[2] );
//    glob( path + "sem_img_temp/*.png", imgPath[3] );
    
//    glob( path + "rgb_img_temp/*", imgPath[1] );
    cout << "[DONE]" << endl;
        // Set the camera calibration matrix
    const double f  = atof(argv[2]), 
                 cx = atof(argv[3]), 
                 cy = atof(argv[4]);
    Matx33d K = Matx33d( f, 0, cx,
                         0, f, cy,
                         0, 0,  1);
/*   
//    for ( size_t i = 0; i < imgPath[2].size(); i++ )
//    {
//        Mat tempSem = imread( imgPath[2].at(i), IMREAD_COLOR );
//        Mat building, fence, other, pole, roadLine, road, sidewalk, wall, trafficSign;
//        inRange( tempSem, Vec3b(70, 70, 70), Vec3b(70, 70, 70), building );
//        inRange( tempSem, Vec3b(153, 153, 190), Vec3b(153, 153, 190), fence );
//        inRange( tempSem, Vec3b(160, 170, 250), Vec3b(160, 170, 250), other );
//        inRange( tempSem, Vec3b(153, 153, 153), Vec3b(153, 153, 153), pole );
//        inRange( tempSem, Vec3b(50, 234, 157), Vec3b(50, 234, 157), roadLine );
//        inRange( tempSem, Vec3b(128, 64, 128), Vec3b(128, 64, 128), road );
//        inRange( tempSem, Vec3b(232, 35, 244), Vec3b(232, 35, 244), sidewalk );
//        inRange( tempSem, Vec3b(156, 102, 102), Vec3b(156, 102, 102), wall );
//        inRange( tempSem, Vec3b(0, 220, 220), Vec3b(0, 220, 220), trafficSign );
//        Mat temp = Mat::zeros( tempSem.size(), CV_8U );
//        bitwise_or( temp, building, temp, noArray() );
//        bitwise_or( temp, fence, temp, noArray() );
//        bitwise_or( temp, other, temp, noArray() );
//        bitwise_or( temp, pole, temp, noArray() );
//        bitwise_or( temp, roadLine, temp, noArray() );
//        bitwise_or( temp, road, temp, noArray() );
//        bitwise_or( temp, sidewalk, temp, noArray() );
//        bitwise_or( temp, wall, temp, noArray() );
//        bitwise_or( temp, trafficSign, temp, noArray() );
//        imwrite( "bin_mask/mask_0" + to_string(i) + ".png", temp );
//    }
*/
    
    // --- Find keypoints and them features
    cout << " --- Find keypoints and them features ... " << " ";
    //Ptr< SIFT > detector = SIFT::create( 0, 4, 0.04, 10, 1.6 );     // 0, 3, 0.04, 10, 1.6
    Ptr< SURF > detector = SURF::create( 100, 4, 4, true, false );         // 100, 4, 3, false, false
    vector< vector< KeyPoint > > keypoints;                             // Key points
    vector< Mat > descriptors;                                          // Descriptors key points
    for ( size_t i = 0; i < imgPath[1].size(); i++ )
    {
            // Temporarily read the image and calculate key points
        Mat tempFrame = imread( imgPath[1].at(i), IMREAD_COLOR );
        vector< KeyPoint > tempKeypoints;
        Mat tempDescriptors;
        detector->detectAndCompute( tempFrame, 
                                    filtrationMask ? imread( imgPath[2].at(i), IMREAD_GRAYSCALE ) : noArray(), 
                                    tempKeypoints, 
                                    tempDescriptors, false );
        keypoints.push_back( tempKeypoints );
        descriptors.push_back( tempDescriptors );
    }
    cout << "[DONE]" << endl;
    
    // --- Matching keypoints by them descriptions
    cout << " --- Matching keypoints by them descriptions ... " << " ";
    vector< vector< DMatch > > matches;
    for( size_t i = 0; i < descriptors.size() - 1; i++ )
    {
        vector< DMatch > tempMatches;
        Ptr< DescriptorMatcher > matcher = DescriptorMatcher::create( DescriptorMatcher::BRUTEFORCE );
        matcher->match( descriptors.at(i), descriptors.at(i+1), tempMatches );
        for ( auto &j : tempMatches ) j.imgIdx = int(i);
        
            // Filtration matches
//        if ( filtrationMatches ) matches.push_back( filtration_matches( tempMatches, 100 ) );
//        else matches.push_back( tempMatches );
        matches.push_back( filtrationMatches ? filtration_matches( tempMatches, 0.1f ) : tempMatches );   //0.35f
    }
    cout << "[DONE]" << endl;
    
    // --- Converting keypoints tracks to reconstruction api format
    cout << " --- Converting keypoints tracks to reconstruction api format ... " << " ";
    std::vector< Mat > points2d;
    matches_2_tracks( keypoints, matches, points2d );
    cout << "[DONE]" << endl;
    
#else
    std::vector< Mat > points2d;
    parser_2D_tracks( argv[1], points2d );
    const double f  = atof(argv[2]),
                 cx = atof(argv[3]), cy = atof(argv[4]);
    Matx33d K = Matx33d( f, 0, cx,
                         0, f, cy,
                         0, 0,  1);
#endif
    
    
    // --- Reconstruct the scene using the 2d correspondences
    cout << " --- Reconstruct the scene using the 2d correspondences ... " << " ";
    bool is_projective = true;
    vector< Mat > Rs_est, ts_est, points3d_estimated;
    reconstruct( points2d, Rs_est, ts_est, K, points3d_estimated, is_projective );
    cout << "[DONE]" << endl;
    
    // --- Print output
    cout << "\n----------------------------\n" << endl;
    cout << "Reconstruction: " << endl;
    cout << "============================" << endl;
    cout << "Estimated 3D points: " << points3d_estimated.size() << endl;
    cout << "Estimated cameras: " << Rs_est.size() << endl;
    cout << "Refined intrinsics: " << endl << K << endl << endl;
    
    cout << "3D Visualization: " << endl;
    cout << "============================" << endl;
    
    // --- 3D visialisation
    visualization::Visualizer vis;
    vis.CreateVisualizerWindow( "Open3D", 1600, 900, 50, 50 );
        // Add Coordinate
    auto coord = geometry::TriangleMesh::CreateCoordinateFrame( 0.3, Vector3d( 0.0, 0.0, 0.0 ) );
    coord->ComputeVertexNormals();
    vis.AddGeometry( coord );
    
        // points3d to cloud for visialisation
    auto cloud = make_shared< geometry::PointCloud >();
    double box = 30;
    for ( Mat point3d : points3d_estimated )
    {
        if ( (point3d.at< double >(0,0) > -box) && (point3d.at< double >(0,0) < box) &&
             (point3d.at< double >(1,0) > -box) && (point3d.at< double >(1,0) < box) &&
             (point3d.at< double >(2,0) > -box) && (point3d.at< double >(2,0) < box) )
        {
            Vector3d temPoint, tempColor;
            temPoint.x() = point3d.at< double >(0,0);
            temPoint.y() = point3d.at< double >(1,0);
            temPoint.z() = point3d.at< double >(2,0);
            tempColor.x() = 1.0;
            tempColor.y() = 0.0;
            tempColor.z() = 0.0;
            cloud->points_.push_back( temPoint );
            cloud->colors_.push_back( tempColor );
        }
    }
        // Recovering cameras
    cout << " --- Recovering cameras ... " << " ";
    vector< cv::Affine3d > path_est;
    auto cloud_coord = make_shared< geometry::PointCloud >();
    for ( size_t i = 0; i < Rs_est.size(); ++i )
    {
        cv::Affine3d affine = cv::Affine3d(Rs_est[i],ts_est[i]);
        path_est.push_back( affine );
        
        Vector3d tempPoint = Vector3d( affine.matrix.val[3], 
                                       affine.matrix.val[7] ,
                                       affine.matrix.val[11]);
        Vector3d tempColor = Vector3d(0.0, 0.0 ,1.0);
        
        cloud_coord->points_.push_back( tempPoint );
        cloud_coord->colors_.push_back( tempColor );
    }
    vis.AddGeometry( cloud_coord );
    cout << "[DONE]" << endl;
    
        // Add Point cloud
    vis.AddGeometry( cloud );
        // Start visualization
    vis.Run();
    
    
    
/*    
//    /// Create 3D windows
//    viz::Viz3d window_est("Estimation Coordinate Frame");
//    window_est.setBackgroundColor(); // black by default
//    window_est.registerKeyboardCallback(&keyboard_callback);
    
//    // Create the pointcloud
//    cout << "Recovering points  ... ";
    
//    // recover estimated points3d
//    vector<Vec3f> point_cloud_est;
//    for (size_t i = 0; i < points3d_estimated.size(); ++i)
//        point_cloud_est.push_back(Vec3f(points3d_estimated[i]));
    
//    cout << "[DONE]" << endl;
    
    
//    /// Recovering cameras
//    cout << "Recovering cameras ... ";
    
//    vector< cv::Affine3d > path_est;
//    for (size_t i = 0; i < Rs_est.size(); ++i)
//        path_est.push_back( cv::Affine3d(Rs_est[i],ts_est[i]) );
    
//    cout << "[DONE]" << endl;
    
//    /// Add cameras
//    cout << "Rendering Trajectory  ... ";
    
//    /// Wait for key 'q' to close the window
//    cout << endl << "Press:                       " << endl;
//    cout <<         " 's' to switch the camera pov" << endl;
//    cout <<         " 'q' to close the windows    " << endl;
    
    
//    if ( path_est.size() > 0 )
//    {
//        // animated trajectory
//        int idx = 0, forw = -1, n = static_cast<int>(path_est.size());
        
//        while(!window_est.wasStopped())
//        {
//            /// Render points as 3D cubes
//            for (size_t i = 0; i < point_cloud_est.size(); ++i)
//            {
//                Vec3d point = point_cloud_est[i];
//                cv::Affine3d point_pose(Mat::eye(3,3,CV_64F), point);
                
//                char buffer[50];
//                sprintf (buffer, "%d", static_cast<int>(i));
                
//                viz::WCube cube_widget(Point3f(0.1f,0.1f,0.0f), Point3f(0.0,0.0,-0.1f), true, viz::Color::blue());
//                cube_widget.setRenderingProperty(viz::LINE_WIDTH, 2.0);
//                window_est.showWidget("Cube"+String(buffer), cube_widget, point_pose);
//            }
            
//            cv::Affine3d cam_pose = path_est[ size_t(idx) ];
            
//            viz::WCameraPosition cpw(0.25); // Coordinate axes
//            viz::WCameraPosition cpw_frustum(K, 0.3, viz::Color::yellow()); // Camera frustum
            
//            if ( camera_pov )
//                window_est.setViewerPose(cam_pose);
//            else
//            {
//                // render complete trajectory
//                window_est.showWidget("cameras_frames_and_lines_est", viz::WTrajectory(path_est, viz::WTrajectory::PATH, 1.0, viz::Color::green()));
                
//                window_est.showWidget("CPW", cpw, cam_pose);
//                window_est.showWidget("CPW_FRUSTUM", cpw_frustum, cam_pose);
//            }
            
//            // update trajectory index (spring effect)
//            forw *= (idx==n || idx==0) ? -1: 1; idx += forw;
            
//            // frame rate 1s
//            window_est.spinOnce(1, true);
//            window_est.removeAllWidgets();
//        }
        
//    }
*/
    
    return 0;
}




//// -input /home/roman/SFM/data/
//// /home/roman/SFM/data/
//// STD
//#include <iostream>
//#include <string>
//#include <vector>

//#define CERES_FOUND 1

//// CV
//#include <opencv2/core.hpp>
//#include <opencv2/core/utility.hpp> // parser
//#include <opencv2/videoio.hpp>      // VideoWriter
//#include <opencv2/imgcodecs.hpp>    // imread()
//#include <opencv2/highgui.hpp>      // imshow()
//#include <opencv2/imgproc.hpp>      // cvtColor()
////#include <opencv2/sfm.hpp>
////#include <opencv2/viz.hpp>

//using namespace std;
//using namespace cv;


//int main( int argc, char *argv[] )
//{
//    //std::cout << cv::getBuildInformation() << std::endl;
    
//    if ( argc != 2 )
//    {
//        cout << "erroe folder" << endl;
//        exit(0);
//    }
//        // Read path of folder with img files
////    CommandLineParser parser( argc, argv, "{@input|./|input folder}");
////    cv::String path = parser.get< cv::String >("@input");
//    string path = argv[1];
    
//        // Open img once for find them size
//    vector< cv::String > imgPath[3];
//    glob( path + "depth_img/*.png", imgPath[0] );
//    glob( path + "rgb_img/*.png", imgPath[1] );
//    glob( path + "sem_img/*.png", imgPath[2] ); 
//    Mat depthFrame = imread( imgPath[0].front(), IMREAD_COLOR );
//    Mat rgbFrame = imread( imgPath[1].front(), IMREAD_COLOR );
//    Mat semFrame = imread( imgPath[2].front(), IMREAD_COLOR );
    
//        // Open video for recording
//    int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');    // ('M', 'J', 'P', 'G') ('X', '2', '6', '4')
//    cv::VideoWriter writer_depth, writer_rgb, writer_sem;
//    writer_depth.open( path + "/depth.avi", codec, 30, depthFrame.size(), true );
//    writer_rgb.open( path + "/rgb.avi", codec, 30, rgbFrame.size(), true );
//    writer_sem.open( path + "/sem.avi", codec, 30, semFrame.size(), true );
        
//        // Record frame
////    vector<int> compression_params;
////    compression_params.push_back( IMWRITE_PNG_COMPRESSION );
////    compression_params.push_back( 1 );
//    for ( size_t i = 0; i < imgPath[0].size(); i++ )
//    {
//            // Record depth frame  
//        Mat tempFrame = imread( imgPath[0].at(i), IMREAD_COLOR );
//        cvtColor( tempFrame, tempFrame, cv::COLOR_RGB2BGR );
//        writer_depth << tempFrame;
////        imwrite( "depth_0" + to_string(i) + ".png", tempFrame );
//            // Record rgb frame
//        tempFrame = imread( imgPath[1].at(i), IMREAD_COLOR );
//        cvtColor( tempFrame, tempFrame, cv::COLOR_RGB2BGR );
//        writer_rgb << tempFrame;
////        imwrite( "rgb_0" + to_string(i) + ".png", tempFrame );
//            // Record semantic frame
//        tempFrame = imread( imgPath[2].at(i), IMREAD_COLOR );
//        cvtColor( tempFrame, tempFrame, cv::COLOR_RGB2BGR );
//        writer_sem << tempFrame;
////        imwrite( "sem_0" + to_string(i) + ".png", tempFrame );
//    }
//    writer_depth.release();
//    writer_rgb.release();
//    writer_sem.release();
    
    
    
    
//    return 0;
//}








//// /home/roman/SFM/scene_reconstruction/images/dataset_files.txt 350 240 360
//// /home/roman/SFM/scene_reconstruction/images/dataset_files.txt 1520.4 302.32 246.87
//// /home/roman/SFM/data/rgb_img/dataset_files.txt 720 512 360
//// /home/roman/SFM/2_scene_reconstruction/images4/dataset_files3.txt 720 512 360
//#define CERES_FOUND 1

//#include <opencv2/sfm.hpp>
//#include <opencv2/viz.hpp>
//#include <opencv2/calib3d.hpp>
//#include <opencv2/core.hpp>

//#include <iostream>
//#include <fstream>

//using namespace std;
//using namespace cv;
//using namespace cv::sfm;

//static void help() {
//  cout
//      << "\n------------------------------------------------------------------------------------\n"
//      << " This program shows the multiview reconstruction capabilities in the \n"
//      << " OpenCV Structure From Motion (SFM) module.\n"
//      << " It reconstruct a scene from a set of 2D images \n"
//      << " Usage:\n"
//      << "        example_sfm_scene_reconstruction <path_to_file> <f> <cx> <cy>\n"
//      << " where: path_to_file is the file absolute path into your system which contains\n"
//      << "        the list of images to use for reconstruction. \n"
//      << "        f  is the focal length in pixels. \n"
//      << "        cx is the image principal point x coordinates in pixels. \n"
//      << "        cy is the image principal point y coordinates in pixels. \n"
//      << "------------------------------------------------------------------------------------\n\n"
//      << endl;
//}


//static int getdir(const string _filename, vector<String> &files)
//{
//  ifstream myfile(_filename.c_str());
//  if (!myfile.is_open()) {
//    cout << "Unable to read file: " << _filename << endl;
//    exit(0);
//  } else {
//    size_t found = _filename.find_last_of("/\\");
//    string line_str, path_to_file = _filename.substr(0, found);
//    while ( getline(myfile, line_str) )
//      files.push_back(path_to_file+string("/")+line_str);
//  }
//  return 1;
//}


//int main(int argc, char* argv[])
//{
//  // Read input parameters

//  if ( argc != 5 )
//  {
//    help();
//    exit(0);
//  }

//  // Parse the image paths

//  vector<String> images_paths;
//  getdir( argv[1], images_paths );


//  // Build intrinsics

//  float f  = atof(argv[2]),
//        cx = atof(argv[3]), cy = atof(argv[4]);

//  Matx33d K = Matx33d( f, 0, cx,
//                       0, f, cy,
//                       0, 0,  1);


//  /// Reconstruct the scene using the 2d images

//  bool is_projective = true;
//  vector<Mat> Rs_est, ts_est, points3d_estimated;
//  reconstruct(images_paths, Rs_est, ts_est, K, points3d_estimated, is_projective);


//  // Print output

//  cout << "\n----------------------------\n" << endl;
//  cout << "Reconstruction: " << endl;
//  cout << "============================" << endl;
//  cout << "Estimated 3D points: " << points3d_estimated.size() << endl;
//  cout << "Estimated cameras: " << Rs_est.size() << endl;
//  cout << "Refined intrinsics: " << endl << K << endl << endl;
//  cout << "3D Visualization: " << endl;
//  cout << "============================" << endl;


//  /// Create 3D windows

//  viz::Viz3d window("Coordinate Frame");
//             window.setWindowSize(Size(500,500));
//             window.setWindowPosition(Point(150,150));
//             window.setBackgroundColor(); // black by default

//  // Create the pointcloud
//  cout << "Recovering points  ... ";

//  // recover estimated points3d
//  vector<Vec3f> point_cloud_est;
//  for (int i = 0; i < points3d_estimated.size(); ++i)
//    point_cloud_est.push_back(Vec3f(points3d_estimated[i]));

//  cout << "[DONE]" << endl;


//  /// Recovering cameras
//  cout << "Recovering cameras ... ";

//  vector<Affine3d> path;
//  for (size_t i = 0; i < Rs_est.size(); ++i)
//    path.push_back(Affine3d(Rs_est[i],ts_est[i]));

//  cout << "[DONE]" << endl;


//  /// Add the pointcloud
//  if ( point_cloud_est.size() > 0 )
//  {
//    cout << "Rendering points   ... ";

//    viz::WCloud cloud_widget(point_cloud_est, viz::Color::green());
//    window.showWidget("point_cloud", cloud_widget);

//    cout << "[DONE]" << endl;
//  }
//  else
//  {
//    cout << "Cannot render points: Empty pointcloud" << endl;
//  }


//  /// Add cameras
//  if ( path.size() > 0 )
//  {
//    cout << "Rendering Cameras  ... ";

//    window.showWidget("cameras_frames_and_lines", viz::WTrajectory(path, viz::WTrajectory::BOTH, 0.1, viz::Color::green()));
//    window.showWidget("cameras_frustums", viz::WTrajectoryFrustums(path, K, 0.1, viz::Color::yellow()));

//    window.setViewerPose(path[0]);

//    cout << "[DONE]" << endl;
//  }
//  else
//  {
//    cout << "Cannot render the cameras: Empty path" << endl;
//  }

//  /// Wait for key 'q' to close the window
//  cout << endl << "Press 'q' to close each windows ... " << endl;

//  window.spin();

//  return 0;
//}
