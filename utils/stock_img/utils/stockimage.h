#ifndef _STOCKIMAGE__H
#define _STOCKIMAGE__H

#include <iostream>
#include <fstream>
#include<math.h>
#include <set>
#include <vector>
#include<string>
#include <sstream>
#include "Astock.h"
#include<opencv2/opencv.hpp>
#pragma comment(lib, "opencv_world341.lib")



class StockImage {

public:

	StockImage(int size) { 
		m_size = size;
	}
	~StockImage() {}

	void createImage(std::vector<stock_t> data, float  vol_b, float k_b, float k_t, cv::Mat &dst)
	{
		AStock stock;

		float max_vol = stock.getVolMaxfromSet(data);
		float min_p = std::min(stock.getPriceMinfromSet(data),stock.getMeanMinfromSet(data));
		float max_p = std::max(stock.getPriceMaxfromSet(data), stock.getMeanMaxfromSet(data));

		int size = m_size * m_dis;

		int vol_bottom = floor(vol_b*size);
		int k_bottom = floor(k_b*size);
		int k_top = floor(k_t *size);


		int vol = vol_bottom - k_bottom - 2;
		int price = k_bottom - k_top - 2;
		cv::Scalar color;

		if (dst.empty()) {
			dst = cv::Mat(size, size, CV_8UC3, cv::Scalar::all(0));
		}

		int fmean5 = -1;
		int fmean10 = -1;
		int fmean20 = -1;
		int mean5, mean10, mean20;

		for (int k = 0; k < m_size; k++) {
			int volheight = vol_bottom - round(data[k].vol * vol / max_vol);
			int open = k_bottom - round((data[k].open - min_p) * price / (max_p - min_p));
			int close = k_bottom - round((data[k].close - min_p) * price / (max_p - min_p));
			int high = k_bottom - round((data[k].high - min_p) * price / (max_p - min_p));
			int low = k_bottom - round((data[k].low - min_p) * price / (max_p - min_p));

			mean5 = k_bottom - round((data[k].mean[0] - min_p) * price / (max_p - min_p));
			mean10 = k_bottom - round((data[k].mean[1] - min_p) * price / (max_p - min_p));
			mean20 = k_bottom - round((data[k].mean[2] - min_p) * price / (max_p - min_p));
			if (data[k].close >= data[k].open)
				color = cv::Scalar(127, 127, 127);
			else
				color = cv::Scalar(255, 255, 255);

			cv::rectangle(dst, cv::Point(k * m_dis, volheight), cv::Point(k * m_dis + m_squ - 1, vol_bottom), color, CV_FILLED);
			cv::rectangle(dst, cv::Point(k * m_dis, close), cv::Point(k* m_dis + m_squ - 1, open), color, CV_FILLED);
			cv::line(dst, cv::Point(k * m_dis + 1, high), cv::Point(k * m_dis + 1, close), color, 1);
			cv::line(dst, cv::Point(k * m_dis + 1, low), cv::Point(k * m_dis + 1, open), color, 1);

			if (fmean5 > 0) {
				color = cv::Scalar(255, 0, 0);
				cv::line(dst, cv::Point((k - 1) * m_dis + 1, fmean5), cv::Point(k * m_dis + 1, mean5), color, 1);
			}
			fmean5 = mean5;

			if (fmean10 > 0) {
				color = cv::Scalar(0, 255, 0);
				cv::line(dst, cv::Point((k - 1) * m_dis + 1, fmean10), cv::Point(k * m_dis + 1, mean10), color, 1);
			}
			fmean10 = mean10;

			if (fmean20 > 0) {
				color = cv::Scalar(0, 0, 255);
				cv::line(dst, cv::Point((k - 1) * m_dis + 1, fmean20), cv::Point(k * m_dis + 1, mean20), color, 1);
			}
			fmean20 = mean20;
		}
	}
	

private:
	int m_dis = 4;
	int m_squ = 3;
	int m_size;	
};

#endif















































































