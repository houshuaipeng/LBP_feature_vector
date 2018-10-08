#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <iostream>
 
using namespace cv;
using namespace std;
 
//原始LBP
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
		 code |= (img.at<uchar>(i-1, j-1) >= center)<<7;//code与后面的做按位或操作并将结果赋给code，<<是将结果存入凑得里，并向左移，最终code里寸的是一个8位的2进制数
		 code |= (img.at<uchar>(i-1, j) >= center)<<6;
		 code |= (img.at<uchar>(i-1, j+1) >= center)<<5;
		 code |= (img.at<uchar>(i, j+1) >= center)<<4;
		 code |= (img.at<uchar>(i+1, j+1) >= center)<<3;
		 code |= (img.at<uchar>(i+1, j) >= center)<<2;
		 code |= (img.at<uchar>(i+1, j-1) >= center)<<1;
		 code |= (img.at<uchar>(i, j-1) >= center)<<0;
		 result.at<uchar>(i -1, j -1) = code;//将8位2进制数存入result中，result是一个比原图少两行、两列的矩阵（少了边缘的一圈）
	  }
   }
   return result;
}
 
//圆形LBP
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
 
//八位二进制跳变次数
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
 
//旋转不变LBP
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
	     temp = i>>1;
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

 /*生成特征向量*/
vector<int> getResult(Mat img)
{
	/*先剪切图片的长和宽为16的整倍数，多余的像素删除*/
	Mat roi(img.rows/16*16,img.cols/16*16,CV_8U);

	for(int i=0;i<roi.rows;i++)
		for(int j=0;j<roi.cols;j++)
			roi.at<char>(i,j)=img.at<char>(i,j);
	 /*直方图均衡化*/
	equalizeHist(roi,roi);
	vector<vector<int>> a(256,vector<int>(256,0));//初始化一个256行，256列的矩阵，每一行对应一个小块的直方图数据
	int x=roi.rows/16;
	int y=roi.cols/16;//计算每个区块的长与宽
	for(int i=0;i<roi.rows;i++)
		for(int j=0;j<roi.cols;j++)//遍历reshape后的LBP图像
		{
			int value=roi.at<uchar>(i,j);
			int num=i/x*16+j/y;//num表示区块的序号
			a[num][value] ++;
		}
	vector<int> b(256*256,0);//初始化一个一维向量，存放最终的特征向量
	for(int i = 0; i<256 ;i++)
		for(int j = 0; j < 256 ; j++)
			b[i*256+j] = a[i][j];

	return b;
}
int main()
{
   Mat src = imread("G:/2.png", 0);
   Mat dst = LBP(src);
   imshow("原始图片", src);
   imshow("原始LBP", dst);
   vector<int> result = getResult(dst);
   cout<<"特征向量为:\n"<<endl;
   for(int i = 0;i<result.size();i++)
   {
		printf("%d",result[i]);
   }

  /* Mat edst = ELBP(src, 1, 8);
   Mat pic = RILBP(src);
   Mat img = UniformLBP(src);
 
   imshow("原始图片", src);
   imshow("原始LBP", dst);
   imshow("圆形LBP", edst);
   imshow("旋转不变LBP", pic);
   imshow("UniformLBP", img);*/
   waitKey(0);
   return 0;
}
