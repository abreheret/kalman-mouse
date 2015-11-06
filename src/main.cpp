#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"

using namespace cv;
using namespace std;

template<class T,int max_size>
class CircularBuff {
	/// private data for circular buffer
	int      _start        ;///< index of oldest element              
	int      _end          ;///< index at which to write new element  
	T        _elt[max_size];///< vector of elements    
	int      _size         ;///< number of elements

public :	

	CircularBuff() {
		_size  = 0;
		_start = 0;
		_end   = 0;
	}
	virtual ~CircularBuff() {}
	int size() const {return _size;}
	int isFull() const { return _size+1 >= max_size; }
	int isEmpty() const { return _size == 0; }
	void push(const T & elem) {
		_elt[_end] = elem;
		_end = (_end + 1) % (max_size);
		_size++;
		if (_end == _start) {
			_start = (_start + 1) % (max_size); // full and overwrite 
			_size = max_size-1;
		}
	}
	T pop() {
		int i = _start;
		_start = (_start + 1) % (max_size);
		_size--;
		return _elt[i];
	}

	const T & operator [] (int index) const { 
		if(index >= max_size || index < 0) {
			throw std::exception("out of bound exception");
		} else {
			return _elt[(index+_start)%(max_size)];
		}
	}
};

void drawCross(cv::Mat & img,cv::Point center,cv::Scalar color,int d ) {                                
	line( img, Point( center.x - d, center.y - d ), Point( center.x + d, center.y + d ), color, 2, CV_AA, 0); 
	line( img, Point( center.x + d, center.y - d ), Point( center.x - d, center.y + d ), color, 2, CV_AA, 0); 
}


cv::Point mousePos;
float measure_noise = 10.f;
float process_noise = 1e-5f;
float error_cov = 0.1f;
char * name_window = "KalmanMouse";

void mouseCallback(int event, int x, int y, int flags, void* param) {
	mousePos.x = x;	
	mousePos.y = y;
}

KalmanFilter initKalman6() {
	KalmanFilter kf6(6, 2, 0);
	setIdentity(kf6.measurementMatrix);
	kf6.transitionMatrix = *(Mat_<float>(6, 6) << 
		1,0,1,0,.5,0,
		0,1,0,1,0,.5, 
		0,0,1,0,1,0,  
		0,0,0,1,0,1,
		0,0,0,0,1,0,
		0,0,0,0,0,1);

	kf6.processNoiseCov = *(Mat_<float>(6, 6) << 
		process_noise,0,0,0,0,0,
		0,process_noise,0,0,0,0, 
		0,0,process_noise,0,0,0,  
		0,0,0,process_noise,0,0,
		0,0,0,0,1e-2f,0,
		0,0,0,0,0,1e-2f);
	
	setIdentity(kf6.measurementNoiseCov, Scalar::all(measure_noise*measure_noise));
	setIdentity(kf6.errorCovPost, Scalar::all(error_cov));
	kf6.statePost.at<float>(0) = (float)mousePos.x;
	kf6.statePost.at<float>(1) = (float)mousePos.y;
	kf6.statePost.at<float>(2) = 0;
	kf6.statePost.at<float>(3) = 0;
	kf6.statePost.at<float>(4) = 0;
	kf6.statePost.at<float>(5) = 0;
	return kf6;
}

KalmanFilter initKalman4() {
	KalmanFilter kf4(4, 2, 0);
	setIdentity(kf4.measurementMatrix);
	kf4.transitionMatrix = *(Mat_<float>(4, 4) << 
		1,0,1,0,
		0,1,0,1,
		0,0,1,0,
		0,0,0,1 );
	kf4.processNoiseCov = *(Mat_<float>(4, 4) << 
		process_noise,0,0,0,
		0,process_noise,0,0,
		0,0,1e-2f,0,
		0,0,0,1e-2f );
	setIdentity(kf4.measurementNoiseCov, Scalar::all(measure_noise*measure_noise));
	setIdentity(kf4.errorCovPost, Scalar::all(error_cov));
	kf4.statePost.at<float>(0) = (float)mousePos.x;
	kf4.statePost.at<float>(1) = (float)mousePos.y;
	kf4.statePost.at<float>(2) = 0;
	kf4.statePost.at<float>(3) = 0;
	return kf4;
}

int main( ) { 
	mousePos.x = 0 ;
	mousePos.y = 0 ;
	Mat img(800,1200, CV_8UC3);
	img = Scalar::all(0);
	imshow(name_window, img);
	cvSetMouseCallback(name_window,&mouseCallback );
	
	KalmanFilter kf6 = initKalman6();
	KalmanFilter kf4 = initKalman4();

	cv::RNG random;
	CircularBuff<Point,50> mouse,mouse_noise,cb6,cb4;
	cv::Point prev_p = mousePos;
	while(1) {
		
// 		if(mousePos.x == prev_p.x && prev_p.y == mousePos.y) {
// 			if(waitKey(10)==27)break;  
// 			continue;
// 		}
		Mat_<float> measurement(2,1);
		measurement.at<float>(0) = float(mousePos.x+ random.gaussian(measure_noise));
		measurement.at<float>(1) = float(mousePos.y+ random.gaussian(measure_noise));

		
		Mat prediction6 = kf6.predict();
		Point predictPt6(int(prediction6.at<float>(0)),int(prediction6.at<float>(1)));
		Mat estimated6 = kf6.correct(measurement);
		Point statePt6(int(estimated6.at<float>(0)),int(estimated6.at<float>(1)));
		
		
		Mat prediction4 = kf4.predict();
		Point predictPt4(int(prediction4.at<float>(0)),int(prediction4.at<float>(1)));
		Mat estimated4 = kf4.correct(measurement);
		Point statePt4(int(estimated4.at<float>(0)),int(estimated4.at<float>(1)));
		
		img = Scalar::all(0);
		mouse.push(mousePos);
		mouse_noise.push(Point(int(measurement(0)),int(measurement(1))));
		cb6.push(statePt6);
		cb4.push(statePt4);
		drawCross(img, statePt6, Scalar(255,255,255), 5 );
		drawCross(img, statePt4, Scalar(255,  0,255), 5 );

		for (int i = 0; i < mouse.size()-1; i++) line(img, mouse[i], mouse[i+1], Scalar(255,255,0), 1);
		for (int i = 0; i < mouse.size(); i++) drawCross(img, mouse[i], Scalar(0,255,255), 1);
		for (int i = 0; i < mouse_noise.size(); i++) drawCross(img, mouse_noise[i], Scalar(0,0,255), 1);
		for (int i = 0; i < cb6.size()-1; i++) line(img, cb6[i], cb6[i+1], Scalar(0,155,255), 2);
 		for (int i = 0; i < cb6.size(); i++) drawCross(img, cb6[i], Scalar(0,255,0), 1);
		for (int i = 0; i < cb4.size()-1; i++) line(img, cb4[i], cb4[i+1], Scalar(255,155,0), 2);
		for (int i = 0; i < cb4.size(); i++) drawCross(img, cb4[i], Scalar(255,0,0), 1);

		imshow(name_window, img);
		if(waitKey(10)==27)break;  

		prev_p = mousePos;
	}
	return 0;
}