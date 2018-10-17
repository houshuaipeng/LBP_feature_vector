#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <iostream>
 
using namespace cv;
using namespace std;
 
//ѭ������
int ror (int val,int length)
{
	int rest = val<<(8-length);
	rest=rest&255;
	val = val>>length;
	val = rest|val;
	return val;
}

//ԭʼLBP
Mat LBP(Mat img)
{
   Mat result;
   result.create(img.rows - 2, img.cols -2 , img.type());
   result.setTo(0);
 
   for(int i = 1; i<img.rows - 1; i++)
   {
      for(int j = 1;j<img.cols -1; j++)
	  {
	     uchar center = img.at<uchar>(i, j);
		 uchar code = 0;
		 code |= (img.at<uchar>(i-1, j-1) >= center)<<7;//code����������λ����������������code��<<�ǽ��������յ���������ƣ�����code������һ��8λ��2������
		 code |= (img.at<uchar>(i-1, j) >= center)<<6;
		 code |= (img.at<uchar>(i-1, j+1) >= center)<<5;
		 code |= (img.at<uchar>(i, j+1) >= center)<<4;
		 code |= (img.at<uchar>(i+1, j+1) >= center)<<3;
		 code |= (img.at<uchar>(i+1, j) >= center)<<2;
		 code |= (img.at<uchar>(i+1, j-1) >= center)<<1;
		 code |= (img.at<uchar>(i, j-1) >= center)<<0;
		 result.at<uchar>(i -1, j -1) = code;//��8λ2����������result�У�result��һ����ԭͼ�����С����еľ������˱�Ե��һȦ��
	  }
   }
   return result;
}
 
//Բ��LBP
Mat ELBP(Mat img, int radius, int neighbors)
{
   Mat result;
   result.create(img.rows-radius*2, img.cols-radius*2, img.type());
   result.setTo(0);
 
   for(int n=0; n<neighbors; n++)
	 {
        // sample points
        float x = static_cast<float>(radius * cos(2.0*CV_PI*n/static_cast<float>(neighbors)));
        float y = static_cast<float>(-radius * sin(2.0*CV_PI*n/static_cast<float>(neighbors)));
        // relative indices
        int fx = static_cast<int>(floor(x));
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x));
        int cy = static_cast<int>(ceil(y));
        // fractional part
        float ty = y - fy;
        float tx = x - fx;
        // set interpolation weights
        float w1 = (1 - tx) * (1 - ty);
        float w2 =      tx  * (1 - ty);
        float w3 = (1 - tx) *      ty;
        float w4 =      tx  *      ty;
        // iterate through your data
        for(int i=radius; i < img.rows-radius;i++) 
		{
            for(int j=radius;j < img.cols-radius;j++) 
			{
                // calculate interpolated value
                float t = static_cast<float>(w1*img.at<uchar>(i+fy,j+fx) + w2*img.at<uchar>(i+fy,j+cx) + w3*img.at<uchar>(i+cy,j+fx) + w4*img.at<uchar>(i+cy,j+cx));
                // floating point precision, so check some machine-dependent epsilon
                result.at<uchar>(i-radius,j-radius) += ((t > img.at<uchar>(i,j)) || (std::abs(t-img.at<uchar>(i,j)) < std::numeric_limits<float>::epsilon())) << n;
            }
	    }
   }
   return result;
}
 
//��λ�������������
int getHopCount(uchar i)
{
	uchar a[8] ={0};
	int cnt =0;
	int k = 7;
 
	while(k)
	{
	   a[k] = i&1;
	   i = i>>1;
	   --k;
	}
 
	for(int k =0; k<7;k++)
	{
	   if(a[k] !=a[k+1])
		   ++cnt;
	}
 
	if(a[0] != a[7])
		++cnt;
 
	return cnt;
}
 
//��ת����LBP
Mat RILBP(Mat img)
{
   uchar RITable[256];
   int temp;
   int val;
   Mat result;
   result.create(img.rows - 2, img.cols -2 , img.type());
   result.setTo(0);
 
   for(int i = 0; i<256; i++)
   {
	   val =i;
      for(int j =0; j<7; j++)
	  {
	     temp = ror(i,j);//��ԭ�������Ʊ�Ϊѭ������
		 if(val>temp)
		 {
		   val = temp;
		 }
	  }
    RITable[i] = val;
   }
 
   for(int i = 1; i<img.rows - 1; i++)
   {
      for(int j = 1;j<img.cols -1; j++)
	  {
	     uchar center = img.at<uchar>(i, j);
		 uchar code = 0;
		 code |= (img.at<uchar>(i-1, j-1) >= center)<<7;
		 code |= (img.at<uchar>(i-1, j) >= center)<<6;
		 code |= (img.at<uchar>(i-1, j+1) >= center)<<5;
		 code |= (img.at<uchar>(i, j+1) >= center)<<4;
		 code |= (img.at<uchar>(i+1, j+1) >= center)<<3;
		 code |= (img.at<uchar>(i+1, j) >= center)<<2;
		 code |= (img.at<uchar>(i+1, j-1) >= center)<<1;
		 code |= (img.at<uchar>(i, j-1) >= center)<<0;
		 result.at<uchar>(i -1, j -1) = RITable[code];	   
	  }
   }
   return result;
}
 
//UniformLBP
Mat UniformLBP(Mat img)
{
	uchar UTable[256];
	memset(UTable, 0, 256*sizeof(uchar));
	uchar temp =1;
   for(int i =0; i<256; i++)
   {
	   if(getHopCount(i)<=2)
	   {
	      UTable[i] = temp;
		  ++temp;
	   }
   }
	 Mat result;
   result.create(img.rows - 2, img.cols -2 , img.type());
  
   result.setTo(0);
 
   for(int i = 1; i<img.rows - 1; i++)
   {
      for(int j = 1;j<img.cols -1; j++)
	  {
	     uchar center = img.at<uchar>(i, j);
		 uchar code = 0;
		 code |= (img.at<uchar>(i-1, j-1) >= center)<<7;
		 code |= (img.at<uchar>(i-1, j) >= center)<<6;
		 code |= (img.at<uchar>(i-1, j+1) >= center)<<5;
		 code |= (img.at<uchar>(i, j+1) >= center)<<4;
		 code |= (img.at<uchar>(i+1, j+1) >= center)<<3;
		 code |= (img.at<uchar>(i+1, j) >= center)<<2;
		 code |= (img.at<uchar>(i+1, j-1) >= center)<<1;
		 code |= (img.at<uchar>(i, j-1) >= center)<<0;
		 result.at<uchar>(i -1, j -1) = UTable[code];	   
	  }
   }
   return result;
}

vector<int> getVector(const Mat &_t1f)
{
	Mat t1f;
	_t1f.convertTo(t1f, CV_64F);
	return (vector<int>)(t1f.reshape(1, t1f.rows*t1f.cols));
}

 /*������������*/
vector<int> getResult(Mat img)
{
	int div =8;//ͼƬ�ֿ��������div*div�飩
	int patterns = 256;//LBP��ֵģʽ��������ת������36��uniform��9��������256��
	/*�ȼ���ͼƬ�ĳ��Ϳ�Ϊ16�������������������ɾ��*/
	Mat roi(img.rows/div*div,img.cols/div*div,CV_8U);

	for(int i=0;i<roi.rows;i++)
		for(int j=0;j<roi.cols;j++)
			roi.at<char>(i,j)=img.at<char>(i,j);
	 /*ֱ��ͼ���⻯*/
	equalizeHist(roi,roi);
	vector<vector<int>> a(div*div,vector<int>(patterns,0));//��ʼ��һ��div*div�У�patterns�еľ���ÿһ�ж�Ӧһ��С���ֱ��ͼ����
	int x=roi.rows/div;
	int y=roi.cols/div;//����ÿ������ĳ����
	for(int i=0;i<roi.rows;i++)
		for(int j=0;j<roi.cols;j++)//����reshape���LBPͼ��
		{
			int value=roi.at<uchar>(i,j);
			int num=i/x*div+j/y;//num��ʾ��������
			a[num][value] ++;
		}
	vector<int> b(div*div*patterns,0);//��ʼ��һ��һά������������յ���������
	for(int i = 0; i<div*div ;i++)
		for(int j = 0; j < patterns ; j++)
			b[i*patterns+j] = a[i][j];
	return b;
}
int main()
{
   Mat src = imread("C:/Users/14518/Desktop/��������ѡͼƬ/1.jpg", 0);
   Mat dst =LBP(src);
   imshow("ԭʼͼƬ", src);
   imshow("��ת����LBP", dst);
   vector<int> result = getResult(dst);
   cout<<"��������Ϊ:\n"<<endl;
   cout<<result.size()<<endl;
   for(int i = 0;i<result.size();i++)
   {
		printf("%d",result[i]);
   }

  /* Mat edst = ELBP(src, 1, 8);
   Mat pic = RILBP(src);
   Mat img = UniformLBP(src);
 
   imshow("ԭʼͼƬ", src);
   imshow("ԭʼLBP", dst);
   imshow("Բ��LBP", edst);
   imshow("��ת����LBP", pic);
   imshow("UniformLBP", img);*/
   waitKey(0);
   return 0;
}
