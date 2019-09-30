#ifndef _ASTOCK__H
#define _ASTOCK__H

#include <iostream>
#include <fstream>
#include<math.h>
#include <set>
#include <vector>
#include<string>
#include <sstream>
#include <map>

struct info_t {
	std::string code;
	float  sy; //每股收益
	float  zc; //每股净资产
	float  tradable_shares; // 流通股数
	float  total_shares; //总股数
	std::string  name; //股票名称
};

struct stock_t {
	std::string code;
	std::string time;
	float open;
	float close;
	float high;
	float low;
	float vol;
	std::vector<float> mean;
};

#define SH_DP  "sh000001"
#define SZ_DP  "sz399001"
#define ZXB    "sz399005"
#define CYB    "sz399006"
#define A50    "sh000016"

class AStock {
public:
	AStock() {}
	~AStock() {}

	/*
	   遍历所有股票代码
	   int code: 初始值为0
	   int start[] = { 1 ,300001 ,600000 ,603000 };
	   int  end[] = { 2892, 300691 ,601999,603999 };
	*/
	int iterator(int& code) {
		if (code == 0) code = 1;
		else if (code >= 1 && code < 2892) code++;
		else if (code == 2892) code = 300001;
		else if (code >= 300001 && code < 300691) code++;
		else if (code == 300691) code = 600000;
		else if (code >= 600000 && code < 601999) code++;
		else if (code == 601999) code = 603000;
		else if (code >= 603000 && code < 603999) code++;
		else return 0;
		return 1;
	}

	void calMean(std::vector<stock_t> &data) {

		for (int i = 0; i < data.size(); i++) {

			data[i].mean.resize(3);

			int count = 0;
			float m = 0.0;

			for (int j = i; j >= 0; j--) {
				count++;
				m = (m*(count - 1) + data[j].close) / count;
				if (count == 5)
					data[i].mean[0] = m;
				if (count == 10)
					data[i].mean[1] = m;
				if (count == 20) {
					data[i].mean[2] = m;
					break;
				}
			}
			if (count < 5) {
				data[i].mean[0] = m;
				data[i].mean[1] = m;
				data[i].mean[2] = m;
			}
			else if (count < 10) {
				data[i].mean[1] = m;
				data[i].mean[2] = m;
			}
			else if (count < 20)
				data[i].mean[2] = m;
		}
	}

			
	/*
	返回股票属于什么类型
	code in {1, 2892} || {300001,300691}, It is SZ
	code in {600000, 601999} || {603000,603999}, It is SH
	*/
	std::string  getLoc(int code) {
		char buf[9];
		std::string str;
		if ((code >= 1 && code <= 2892) || (code >= 300001 && code <= 300691)) {
			sprintf_s(buf, "sz%06d", code);
			str = buf;
		}
		else if ((code >= 600000 && code <= 601999) || (code >= 603000 && code <= 603999)) {
			sprintf_s(buf, "sh%06d", code);
			str = buf;
		}
		return str;
	}			

	std::string coreespod_dp(int code) {
		if (code >= 1 && code < 2000) return SZ_DP;
		if (code >= 2000 && code <= 2892) return ZXB;
		if (code >= 300001 && code <= 300691) return CYB;
		if ((code >= 600000 && code <= 601999) || (code >= 603000 && code <= 603999)) return  SH_DP;
		return  SH_DP;
	}

	int  getSet(std::map<std::string, stock_t> Kdata, std::string start, int num, std::vector<stock_t> &data) {

		data.resize(num);
		auto iter = Kdata.rbegin();
		int count = num - 1;
		while (iter != Kdata.rend())
		{
			if (iter->first > start) {
				++iter;
				continue;
			}
			data[count] = iter->second;
			++iter;
			count--;
			if (count < 0) break;
		}
		if (count >= 0) return 0;
		return 1;
	}

	int  getSet(std::map<std::string, stock_t> Kdata, std::vector<stock_t> cmp, std::vector<stock_t> &data) {

		data.resize(cmp.size());
		for (int i = 0; i < cmp.size(); i++) {
			auto iter = Kdata.find(cmp[i].time);
			if (iter == Kdata.end()) return 0;
			data[i] = iter->second;
		}
		return 1;
	}

	void getRelatPerSet(std::vector<stock_t> data, std::vector<stock_t> &relat) {

		relat.resize(data.size());

		relat[0] = data[0];
		relat[0].close = (data[0].close - data[0].close) / data[0].close;
		relat[0].open = (data[0].open - data[0].close )/ data[0].close;
		relat[0].high = (data[0].high - data[0].close) / data[0].close;
		relat[0].low = (data[0].low - data[0].close) / data[0].close;
		relat[0].vol = (data[0].vol - data[0].vol) / data[0].vol;

		for (int i = 1; i < data.size(); i++)
		{
			relat[i] = data[i];
			relat[i].close = (data[i].close - data[i-1].close) / data[i-1].close;
			relat[i].open = (data[i].open - data[i-1].close) / data[i-1].close;
			relat[i].high = (data[i].high - data[i-1].close) / data[i-1].close;
			relat[i].low = (data[i].low - data[i-1].close) / data[i-1].close;
			relat[i].vol = (data[i].vol - data[i-1].vol) / data[i-1].vol;
		}
	}

	float getPriceMaxfromSet(std::vector<stock_t> data) {

		float max = data[0].high;
		for (int i = 1; i < data.size(); i++)
		{
			if (data[i].high > max) max = data[i].high;
		}
		return  max;
	}

	float getPriceMinfromSet(std::vector<stock_t> data) {

		float min = data[0].low;
		for (int i = 1; i < data.size(); i++)
		{
			if (data[i].low < min) min = data[i].low;
		}
		return  min;
	}

	float getVolMaxfromSet(std::vector<stock_t> data) {

		float max = data[0].vol;
		for (int i = 1; i < data.size(); i++)
		{
			if (data[i].vol > max) max = data[i].vol;
		}
		return  max;
	}

	float  getMeanMaxfromSet(std::vector<stock_t> data) {

		auto iter = std::max_element(data[0].mean.begin(), data[0].mean.end());
		float  max = *iter;

		for (int i = 1; i < data.size(); i++)
		{
			auto iter = std::max_element(data[i].mean.begin(), data[i].mean.end());
			if (max < *iter) max = *iter;
		}
		return max;
	}

	float  getMeanMinfromSet(std::vector<stock_t> data) {

		auto iter = std::min_element(data[0].mean.begin(), data[0].mean.end());
		float  min = *iter;

		for (int i = 1; i < data.size(); i++)
		{
			auto iter = std::min_element(data[i].mean.begin(), data[i].mean.end());
			if (min > *iter) min = *iter;
		}
		return min;
	}

	void write_BaseInfo(info_t &info, std::string filepath) {

		std::ofstream f;

		f.open(filepath, std::ios::out | std::ios::trunc);
		if (!f.is_open())
		{
			printf("open file %s error!\n", filepath.c_str());
			return ;
		}
		f << info.code << " " << info.sy << " " << info.zc << " " << info.tradable_shares << " " << info.total_shares << " "
			<< info.name << std::endl;
		f.close();
	}

	void write_Kline(std::map<std::string, stock_t> &Kdata, std::string filepath ) {

		std::ofstream f;
		f.open(filepath, std::ios::out | std::ios::trunc);
		if (!f.is_open())
		{
			printf("open file %s error!\n", filepath.c_str());
			return ;
		}

		auto iter = Kdata.begin();
		while (iter != Kdata.end()) {
			f << iter->second.time << " " << iter->second.open << " " << iter->second.close << " " << iter->second.high << " " << iter->second.low
				<< " " << iter->second.vol << std::endl;
			iter++;
		}

		f.close();
	}

private:
};
#endif