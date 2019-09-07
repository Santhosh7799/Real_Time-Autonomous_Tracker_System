//////////////////////////////////////////////////////////////////////////////////////////
//		@Authors: Sai Raghavnedra Sankrantipati, Santosh Rao Thummnapalli
//		Description: This program can track an object in green color. There are 4 services
//						Service 1: Image Capture Service
//						Service 2: Image Filtering Service
//						Service 3: Object Tracking Service
//						Service 4: Video Saving Thread
//
//						 Service	WCET 	Grace Time
//							1		9ms		15ms
//							2		108ms	135ms
//							3		6ms		10ms
//							4		34ms	40ms
//						 Deadline: 200ms
//						 5fps
//
//		Usage: Using Makefile. 
//				make all
//				sudo ./Final_Project
//		Reference: 1. We used Sam Siewert's sequence code for service creation an execution
//					2. Kyle Hounslow's tutorials for object tracking
////////////////////////////////////////////////////////////////////////////////////////// 


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <semaphore.h>

#include <syslog.h>
#include <sys/time.h>

#include <errno.h>
#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <syslog.h>
#include <math.h>

#define USEC_PER_MSEC (1000)
#define NANOSEC_PER_SEC (1000000000)
#define NUM_THREADS (4+1)
#define TRUE (1)
#define FALSE (0)

int abortTest=FALSE;
int abortS1=FALSE, abortS2=FALSE, abortS3=FALSE, abortS4=FALSE, abortS5=FALSE, abortS6=FALSE, abortS7=FALSE;
sem_t semS1, semS2, semS3, semS4, semS5, semS6, semS7;
struct timeval start_time_val;

//Declaration of services
void *Sequencer(void *threadp);

void *Service_1(void *threadp);
void *Service_2(void *threadp);
void *Service_3(void *threadp);
void *Service_4(void *threadp);

using namespace cv;
using namespace std;

//Resolution of screen
#define HRES 640
#define VRES 480


const char windowName[] = "Normal Image";
const char windowName1[] = "After HSV transformation";
const char windowName2[] = "Filtered Image";

//HSV min and max values for Green color
int min_h = 30;
int max_h = 80;
int min_s = 55;
int max_s = 215;
int min_v = 55;
int max_v = 205;

//OpenCV global variables
IplImage* frame_captured;
vector< vector<Point> > contours;
vector<Vec4i> hierarchy;
Moments moment;
Mat hsv, hsv_threshold, erode_element, dilate_element, image_temp, cam_feed;
CvCapture* capture;
Mat *address;


//This function captures one image
void capture_image(){
	frame_captured=cvQueryFrame(capture);
	if(!frame_captured){
		printf("No Frame capture thread 1\n");
	}
}

//This function filters image and produces a threshold image
void filter_image(){

	if(!frame_captured){
		printf("No Frame capture thread 2\n");		
	}

	//mat_frame is declared as static as it used in other functions	
	static Mat mat_frame(frame_captured);
	medianBlur(mat_frame, mat_frame, 3);
	//BGR image is converted to HSV
	cvtColor(mat_frame, hsv, CV_BGR2HSV);
	//Separate portion of image with in given limits
	inRange(hsv, Scalar(min_h, min_s, min_v), Scalar(max_h, max_s, max_v), hsv_threshold);
	//Erode and Dialate functions to filter noise
	Mat erode_element = getStructuringElement( MORPH_RECT,Size(3,3));
	Mat dilate_element = getStructuringElement( MORPH_RECT,Size(8,8));
	erode(hsv_threshold, hsv_threshold, erode_element );
	erode(hsv_threshold, hsv_threshold, erode_element );
	dilate(hsv_threshold, hsv_threshold, dilate_element);
	dilate(hsv_threshold, hsv_threshold, dilate_element);

	hsv_threshold.copyTo(image_temp);
	//mat_frame's address is used in other functions
	address = &mat_frame;

}

//This functions tracks the image and displays it on original image
void track_image(){

	//points to track object
	int x=0, y=0;
	findContours(image_temp, contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE );
	double ref_area = 0;
	bool found = false;
	if (hierarchy.size() > 0){
		int total = hierarchy.size();
		//in a noise environments total objects would be greater than 50
		if(total<50){
			for (int index = 0; index >= 0; index = hierarchy[index][0]) {
				//Calculate moments
				Moments moment = moments((cv::Mat)contours[index]);
				double area = moment.m00;
				//Look for object by comparing areas
				if(area>(20*20) && area<((int)(640*480)/1.5)){
					x = moment.m10/area;
					y = moment.m01/area;
					found = true;
					ref_area = area;
				} else
					found = false;
			}	
			if(found == true){
				
				//Write tracking.. on left corner of image
				putText(*address,"Tracking..",Point(0,50),FONT_HERSHEY_SIMPLEX, 1,Scalar(0,0,0),2);
				//Draw a black circle on the object
				circle(*address,Point(x,y),25,Scalar(0,0,0),2);	
				//Draw two perpendicular diameters on circle
				if(y-20>0)
    			line(*address,Point(x,y),Point(x,y-25),Scalar(0,0,0),2);
    			else line(*address,Point(x,y),Point(x,0),Scalar(0,0,0),2);
			    if(y+20<480)
    			line(*address,Point(x,y),Point(x,y+25),Scalar(0,0,0),2);
   			 	else line(*address,Point(x,y),Point(x,480),Scalar(0,0,0),2);
    			if(x-20>0)
    			line(*address,Point(x,y),Point(x-25,y),Scalar(0,0,0),2);
    			else line(*address,Point(x,y),Point(0,y),Scalar(0,0,0),2);
    			if(x+20<640)
    			line(*address,Point(x,y),Point(x+25,y),Scalar(0,0,0),2);
    			else line(*address,Point(x,y),Point(640,y),Scalar(0,0,0),2);
			}
			
		} 	
				
	}		
		
	//Show Original image
	imshow(windowName,*address);
	//show bgr to hsv transformed image
	imshow(windowName1,hsv);
	//Show differntiated image
	imshow(windowName2,hsv_threshold);
		
				
}


//This function writes image to a video as capture.avi in the same folder
void video_image(){

	VideoWriter output_cap("capture.avi", CV_FOURCC('M','J','P','G'), 1, Size ( 640,480));
	if(!output_cap.isOpened())
		printf("Unable to open capture.avi\n");
	//Write original image to the video file	
	output_cap.write(*address);
	
}
	


void print_scheduler(void)
{
   int schedType;

   schedType = sched_getscheduler(getpid());

   switch(schedType)
   {
       case SCHED_FIFO:
           printf("Pthread Policy is SCHED_FIFO\n");
           break;
       case SCHED_OTHER:
           printf("Pthread Policy is SCHED_OTHER\n"); exit(-1);
         break;
       case SCHED_RR:
           printf("Pthread Policy is SCHED_RR\n"); exit(-1);
           break;
       default:
           printf("Pthread Policy is UNKNOWN\n"); exit(-1);
   }
}


int main(int argc, char *argv[] )
{

	struct timeval current_time_val;
    int i, rc, scope;
    pthread_t threads[NUM_THREADS];
    pthread_attr_t rt_sched_attr[NUM_THREADS];
    int rt_max_prio, rt_min_prio;
    struct sched_param rt_param[NUM_THREADS];
    struct sched_param main_param;
    pthread_attr_t main_attr;
    pid_t mainpid;
    
    gettimeofday(&start_time_val, (struct timezone *)0);
    gettimeofday(&current_time_val, (struct timezone *)0);
    syslog(LOG_CRIT, "Sequencer @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);
    
    int dev=0;
    
    if(argc > 1)
    {
        sscanf(argv[1], "%d", &dev);
        printf("using %s\n", argv[1]);
    }
    else if(argc == 1)
        printf("using default\n");

    else
    {
        printf("usage: capture [dev]\n");
        exit(-1);
    }
    
    
    namedWindow( windowName, CV_WINDOW_AUTOSIZE );
    namedWindow( windowName1, CV_WINDOW_AUTOSIZE );
    namedWindow( windowName2, CV_WINDOW_AUTOSIZE );
    
    capture = (CvCapture *)cvCreateCameraCapture(dev);
    
    
	
	/* Setting the resolution using the cvSetCaptureProperty interface */
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, HRES);
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, VRES);
    


    if (sem_init (&semS1, 0, 0)) { printf ("Failed to initialize S1 semaphore\n"); exit (-1); }
    if (sem_init (&semS2, 0, 0)) { printf ("Failed to initialize S2 semaphore\n"); exit (-1); }
    if (sem_init (&semS3, 0, 0)) { printf ("Failed to initialize S3 semaphore\n"); exit (-1); }
    if (sem_init (&semS4, 0, 0)) { printf ("Failed to initialize S4 semaphore\n"); exit (-1); }
    
    mainpid=getpid();
    
    rt_max_prio = sched_get_priority_max(SCHED_FIFO);
    rt_min_prio = sched_get_priority_min(SCHED_FIFO);
    
    rc=sched_getparam(mainpid, &main_param);
    main_param.sched_priority=rt_max_prio;
    rc=sched_setscheduler(getpid(), SCHED_FIFO, &main_param);
    if(rc < 0) perror("main_param");
    print_scheduler();
    
    pthread_attr_getscope(&main_attr, &scope);

    if(scope == PTHREAD_SCOPE_SYSTEM)
      printf("PTHREAD SCOPE SYSTEM\n");
    else if (scope == PTHREAD_SCOPE_PROCESS)
      printf("PTHREAD SCOPE PROCESS\n");
    else
      printf("PTHREAD SCOPE UNKNOWN\n");
      
    printf("rt_max_prio=%d\n", rt_max_prio);
    printf("rt_min_prio=%d\n", rt_min_prio);
    
    for(i=0; i < NUM_THREADS; i++)
    {

      //CPU_ZERO(&threadcpu);
      //CPU_SET(3, &threadcpu);

      rc=pthread_attr_init(&rt_sched_attr[i]);
      rc=pthread_attr_setinheritsched(&rt_sched_attr[i], PTHREAD_EXPLICIT_SCHED);
      rc=pthread_attr_setschedpolicy(&rt_sched_attr[i], SCHED_FIFO);
      //rc=pthread_attr_setaffinity_np(&rt_sched_attr[i], sizeof(cpu_set_t), &threadcpu);

      rt_param[i].sched_priority=rt_max_prio-i;
      pthread_attr_setschedparam(&rt_sched_attr[i], &rt_param[i]);

      //threadParams[i].threadIdx=i;
    }
    
    rt_param[1].sched_priority=rt_max_prio-3;
    pthread_attr_setschedparam(&rt_sched_attr[1], &rt_param[1]);
    rc=pthread_create(&threads[1],               // pointer to thread descriptor
                      &rt_sched_attr[1],         // use specific attributes
                      //(void *)0,               // default attributes
                      Service_1,                 // thread function entry point
                      NULL // parameters to pass in
                     );
	if(rc < 0)
        perror("pthread_create for service 1");
    else
        printf("pthread_create successful for service 1\n");
        
        
    rt_param[2].sched_priority=rt_max_prio-3;
    pthread_attr_setschedparam(&rt_sched_attr[2], &rt_param[2]);
    rc=pthread_create(&threads[2], &rt_sched_attr[2], Service_2, NULL);
    if(rc < 0)
        perror("pthread_create for service 2");
    else
        printf("pthread_create successful for service 2\n");
        
        
    rt_param[3].sched_priority=rt_max_prio-3;
    pthread_attr_setschedparam(&rt_sched_attr[3], &rt_param[3]);
    rc=pthread_create(&threads[3], &rt_sched_attr[3], Service_3, NULL);
    if(rc < 0)
        perror("pthread_create for service 3");
    else
        printf("pthread_create successful for service 3\n");
        
        
    rt_param[4].sched_priority=rt_max_prio-3;
    pthread_attr_setschedparam(&rt_sched_attr[4], &rt_param[4]);
    rc=pthread_create(&threads[4], &rt_sched_attr[4], Service_4, NULL);
    if(rc < 0)
        perror("pthread_create for service 4");
    else
        printf("pthread_create successful for service 4\n");
        
        
    sem_post(&semS1);
	usleep(1000000);
	    
    rt_param[0].sched_priority=rt_max_prio;
    pthread_attr_setschedparam(&rt_sched_attr[0], &rt_param[0]);
    rc=pthread_create(&threads[0], &rt_sched_attr[0], Sequencer, NULL);
    if(rc < 0)
        perror("pthread_create for sequencer service 0");
    else
        printf("pthread_create successful for sequeencer service 0\n");


   for(i=0;i<NUM_THREADS;i++)
       pthread_join(threads[i], NULL);
   cvReleaseCapture(&capture);

}

// Service	WCET 	Grace Time
//	1		9ms		15ms
//	2		108ms	135ms
//	3		6ms		10ms
//	4		34ms	40ms
// Deadline: 200ms
// 5fps

void *Sequencer(void *threadp)
{
	while(1)
	{

		usleep(15000);
		sem_post(&semS2);
		
		usleep(135000);
		sem_post(&semS3);//This service is for filtering image
//	Wcet : 108ms
		
		usleep(10000);
		sem_post(&semS4);
		
		usleep(40000);
		sem_post(&semS1);
		cvWaitKey(10);
		
	}
}


//This service is for capturing image
//	Wcet : 9ms
void *Service_1(void *threadp)
{

    struct timeval current_time_val, previous_time_val;
    double current_time;
    unsigned long long S1Cnt=0;

    gettimeofday(&current_time_val, (struct timezone *)0);

    printf("Start: Frame capture thread @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);

    while(!abortS1)
    {
        sem_wait(&semS1);
        gettimeofday(&previous_time_val, (struct timezone *)0);
        S1Cnt++;
        //Image capture function
		capture_image();
        gettimeofday(&current_time_val, (struct timezone *)0);
		//time taken to execute capture_image()
        printf("Service 1: Frame capture thread | Count:%llu @ sec=%d, msec=%d\n", S1Cnt, (int)(current_time_val.tv_sec-previous_time_val.tv_sec), (int)(current_time_val.tv_usec - previous_time_val.tv_usec)/USEC_PER_MSEC);
    }

    pthread_exit((void *)0);
}

//This service is for filtering image
//	Wcet : 108ms
void *Service_2(void *threadp)
{
    struct timeval current_time_val, previous_time_val;
    double current_time;
    unsigned long long S2Cnt=0;

    gettimeofday(&current_time_val, (struct timezone *)0);

    printf("Start: Image Filtering thread @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);

    while(!abortS2)
    {
        sem_wait(&semS2);
        gettimeofday(&previous_time_val, (struct timezone *)0);
        S2Cnt++;
        //Image filtering function
		filter_image();
        gettimeofday(&current_time_val, (struct timezone *)0);
        //time taken to execute filter_image()
        printf("Service 2: Image Filtering thread | Count:%llu @ sec=%d, msec=%d\n", S2Cnt, (int)(current_time_val.tv_sec-previous_time_val.tv_sec), (int)(current_time_val.tv_usec - previous_time_val.tv_usec)/USEC_PER_MSEC);
    }

    pthread_exit((void *)0);
}

//This service is for tracking object
//	Wcet : 6ms
void *Service_3(void *threadp)
{
    struct timeval current_time_val, previous_time_val;
    double current_time;
    unsigned long long S3Cnt=0;
    gettimeofday(&current_time_val, (struct timezone *)0);
  
    printf("Start: Object Tracking thread @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);

    while(!abortS3)
    {
        sem_wait(&semS3);
        gettimeofday(&previous_time_val, (struct timezone *)0);
        S3Cnt++;
        //function to track object
		track_image();
        gettimeofday(&current_time_val, (struct timezone *)0);
        //time taken to execute track-image
        printf("Service 3: Object Tracking  thread | Count:%llu @ sec=%d, msec=%d\n", S3Cnt, (int)(current_time_val.tv_sec-previous_time_val.tv_sec), (int)(current_time_val.tv_usec - previous_time_val.tv_usec)/USEC_PER_MSEC);
    }

    pthread_exit((void *)0);
}


//Service to write image to a video
//Wcet : 34ms
void *Service_4(void *threadp)
{
    struct timeval current_time_val, previous_time_val;
    double current_time;
    unsigned long long S4Cnt=0;
    gettimeofday(&current_time_val, (struct timezone *)0);

    printf("Start: Video Save thread @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);

    while(!abortS4)
    {
        sem_wait(&semS4);
        gettimeofday(&previous_time_val, (struct timezone *)0);
        S4Cnt++;
        //image to write to a video thread
		video_image();
        gettimeofday(&current_time_val, (struct timezone *)0);
        //Time taken to execute video_image thread
        printf("Service 4: Video Save thread | Count:%llu @ sec=%d, msec=%d\n", S4Cnt, (int)(current_time_val.tv_sec-previous_time_val.tv_sec), (int)(current_time_val.tv_usec - previous_time_val.tv_usec)/USEC_PER_MSEC);
    }

    pthread_exit((void *)0);
}

	
