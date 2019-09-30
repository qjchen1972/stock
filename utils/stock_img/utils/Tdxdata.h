#ifndef _TDXDATA__H
#define _TDXDATA__H

#include <iostream>
#include <fstream>
#include<math.h>
#include <set>
#include <vector>
#include<string>
#include <sstream>
//#include <unordered_map>
#include <map>
#include "Astock.h"

//日线
struct tdx_t{
	unsigned int date;
	int  open; //开盘价 *0.01
	int  high; //最高价 *0.01
	int  low;  //最低价 *0.01
	int  close; //收盘价*0.01
	float amount; //成交金额
	int vol;  //成交量(股数)
	int reserved;//保留
};

//5分钟线
/*
 year = day / 2048 + 2004
 month = (day % 2048) / 100
 day =  (day % 2048) % 100
 hour: 从0点开始的多少分钟
*/
struct lc5_t{
	unsigned short day; 
	unsigned short hour; 
	float   open; //实际价格
	float   high;
	float   low;
	float   close;
	float   amount;
	int   vol; //股数
	float   reserved;
};




#define  KDAY  0
#define  K5MIN 1

class TdxData {

public:
	TdxData(){ }
	~TdxData() {}

	int loaddata(std::string  code, std::map<std::string, stock_t> &Kdata,int type = KDAY, int num = -1) {

		std::string file = path(code,type);
		if (file.empty()) return 0;
	
		std::ifstream f;
		f.open(file, std::ios::in | std::ios::binary | std::ios::ate);
		if (!f.is_open())
		{
			//cout << file << "  open error!" << endl;
			return 0;
		}
		int size = (int)f.tellg();
		int  stock_num;
		if(type == KDAY)
			stock_num = size / sizeof(tdx_t);
		else
			stock_num = size / sizeof(lc5_t);
		if (stock_num == 0) return 0;

		int len = num;
		if (stock_num <= num || num == -1) len = stock_num;
		Kdata.clear();
		if (type == KDAY) {
			f.seekg(-1 * sizeof(tdx_t)*len, std::ios::end);
			std::vector<tdx_t> data;
			data.resize(len);
			f.read((char*)&data[0], len * sizeof(tdx_t));
			for (int i = 0; i < len; i++) {
				stock_t day;
				day.code = code;
				day.time = std::to_string(data[i].date);
				day.open = 0.01 * data[i].open;
				day.close = 0.01 * data[i].close;
				day.high = 0.01 * data[i].high;
				day.low = 0.01 * data[i].low;
				day.vol = 0.01* data[i].vol;
				Kdata[day.time] = day;
			}
		}
		else {
			f.seekg(-1 * sizeof(lc5_t)*len, std::ios::end);
			std::vector<lc5_t> data;
			data.resize(len);
			f.read((char*)&data[0], len * sizeof(lc5_t));
			char timestr[128];
			for (int i = 0; i < len; i++) {
				stock_t day;
				int y, m, d, min;
				getLc5Time(data[i],y,m,d,min);
				sprintf_s(timestr, "%04d%02d%02d%03d", y, m, d,min);
				day.time = timestr;
				day.code = code;
				day.open = data[i].open;
				day.close = data[i].close;
				day.high = data[i].high;
				day.low = data[i].low;
				day.vol = 0.01*data[i].vol;
				Kdata[day.time] = day;
			}
		}
		f.close();
		return 1;
	}


private:		

	std::string path(std::string code, int type) {		

		std::string strPath = "D:/soft_list/tdx/vipdoc/";
		char file[128];

		std::string dir;
		std::string tail;
		if (type == KDAY) {
			dir = "lday";
			tail = "day";
		}
		else {
			dir = "fzline";
			tail = "lc5";
		}
		std::string head = code.substr(0,2);

		sprintf_s(file, "%s/%s/%s.%s", head.c_str(),dir.c_str(),code.c_str(),tail.c_str());
		strPath += file;
		return strPath;
	}

	void getLc5Time(lc5_t lc, int &year, int &month, int &day, int &min) {
		year = lc.day / 2048 + 2004;
		month = (lc.day % 2048) / 100;
		day = (lc.day % 2048) % 100;
		min = lc.hour;
	}
};
#endif