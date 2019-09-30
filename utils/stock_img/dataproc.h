#pragma once
#include <iostream>
#include <fstream>
#include<math.h>
#include <unordered_map>
#include<ctime>

#include "utils/Astock.h"
#include "utils/Tdxdata.h"
#include "utils/statics.h"
#include "utils/stockimage.h"

class DataProc {
public:
	DataProc() { srand(time(NULL)); }
	~DataProc() {}	

	void proc_stock(int code, std::string time, int num, int type=KDAY) {

		AStock stock;
		TdxData tdx;

		std::map<std::string, stock_t> Kdata;
		if (!tdx.loaddata(stock.getLoc(code), Kdata, type)) return;

		std::map<std::string, stock_t> Kdp;
		if (!tdx.loaddata(stock.coreespod_dp(code), Kdp,type)) return;

		std::vector<stock_t> data;
		if (!stock.getSet(Kdata, time, num, data)) return;
		std::vector<stock_t> dpdata;
		if (!stock.getSet(Kdp, data, dpdata)) return;


		std::vector<std::vector<float>> vec;
		stock2vec(data, dpdata, vec);

		Statics sta;
		Eigen::MatrixXf m;
		sta.vec2matix(vec, m);

		Eigen::MatrixXf covm;
		sta.covmat(m, covm);

		Eigen::MatrixXf svdm;
		sta.svd(m, 2, svdm);
	}

	void proc_data() {

		AStock stock;
		TdxData tdx;		

		std::vector<std::vector<float>> vec;
		for (int i = 0; i < 10; i++) {
			std::vector<float> a;
			for (int j = 0; j < 5; j++) {
				a.push_back(rand() % 10);
			}
			vec.push_back(a);
		}

		
		Statics sta;
		Eigen::MatrixXf m;
		sta.vec2matix(vec, m);

		Eigen::MatrixXf covm;
		sta.covmat(m, covm);

		Eigen::MatrixXf svdm;
		sta.svd(m, 2, svdm);
	}

	

	void create_trainData(const int dayLen, const int imgLen, std::string startday, std::string endday) {

		const char* label = "mini/hopelabel.txt";
		const char* trainTxt = "mini/train.txt";
		const char* testTxt = "mini/test.txt";
		const char* valTxt = "mini/val.txt";
		const char* imgDir = "mini/trainimg";
		createData(dayLen, imgLen, startday, endday, label, imgDir);
		split_label(90, 70, label, trainTxt, testTxt, valTxt);
	}

	void create_testData(const int dayLen, const int imgLen, std::string startday, std::string endday) {

		const char* label = "mini/test/test.txt";
		const char* imgDir = "mini/test/testimg";
		createData(dayLen, imgLen, startday, endday, label, imgDir,1, 0);
	}

	void create_predData(const int dayLen, const int imgLen, std::string startday, std::string endday) {

		const char* label = "mini/pred/pred.txt";
		const char* imgDir = "mini/pred/predimg";
		createData(dayLen, imgLen, startday, endday, label, imgDir, 0, 0);
	}

private:	

	void stock2vec(std::vector<stock_t> data, std::vector<stock_t> dp, std::vector<std::vector<float>> &vec) {

		AStock stock;

		float mp =  stock.getPriceMaxfromSet(data);
		float mv =  stock.getVolMaxfromSet(data);
		float dmp = stock.getPriceMaxfromSet(dp);
		float dmv = stock.getVolMaxfromSet(dp);

		for (int i = 0; i < data.size(); i++) {
			std::vector<float> a;
			a.push_back(data[i].open / mp);
			a.push_back(data[i].close / mp);
			a.push_back(data[i].high / mp);
			a.push_back(data[i].low / mp);
			a.push_back(data[i].vol / mv);
			a.push_back(dp[i].open / dmp);
			a.push_back(dp[i].close / dmp);
			a.push_back(dp[i].high / dmp);
			a.push_back(dp[i].low / dmp);
			a.push_back(dp[i].vol / dmv);
			vec.push_back(a);
		}
	}

	void write_label(std::ofstream &lf, std::vector<stock_t> data, std::vector<stock_t> dpdata, int code,
		             int imgLen, const char* imgDir, int labeltime,int rnd) {

		if (rand() % 10 < rnd)  return;

		StockImage  image(imgLen);
		char file_name[128];
		char img_name[128];	

		cv::Mat img;
		image.createImage(data, 0.56, 0.44, 0, img);
		image.createImage(dpdata, 1.0, 0.90, 0.56, img);
		sprintf(img_name, "%06d_%s.jpg", code, data[imgLen - 1].time.c_str());
		sprintf_s(file_name, "%s/%s", imgDir, img_name);
		imwrite(file_name, img);
		lf << img_name << " ";
		if (labeltime != 0)
			lf << 100.0*(data[imgLen].close - data[imgLen - 1].close) / data[imgLen - 1].close;
		else
			lf << 0;
		lf<< std::endl;
	}

	void createData(const int dayLen, const int imgLen, std::string startday, std::string endday,
		const char* label, const char* imgDir, int labeltime = 1,int rnd = 8) {

		AStock stock;
		TdxData tdx;

		std::unordered_map<std::string, std::map<std::string, stock_t>> Kdp;
		tdx.loaddata(SH_DP, Kdp[SH_DP]);
		tdx.loaddata(SZ_DP, Kdp[SZ_DP]);
		tdx.loaddata(ZXB, Kdp[ZXB]);
		tdx.loaddata(CYB, Kdp[CYB]);

		std::ofstream lf;
		lf.open(label, std::ios::out | std::ios::trunc);
		if (!lf.is_open()) {
			printf("open file %s error!\n", label);
			return;
		}		

		int code = 0;	
		std::map<std::string, stock_t> Kdata;
		while (stock.iterator(code)) {

			if (!tdx.loaddata(stock.getLoc(code), Kdata, KDAY, dayLen)) continue;
			std::string timestr = endday;
			while (timestr >= startday) {
				std::vector<stock_t> data;
				if (!stock.getSet(Kdata, timestr, imgLen + labeltime, data)) {
					printf("getset err %s %d \n", timestr.c_str(), code);
					break;
				}	
				if (data[imgLen + labeltime - 1].time < startday) break;
				
				std::vector<stock_t> dpdata;
				if (!stock.getSet(Kdp[stock.coreespod_dp(code)], data, dpdata)) {
					printf("get dp set err %s %d \n", timestr.c_str(), code);
					break;
				}
				stock.calMean(data);
				stock.calMean(dpdata);
				write_label(lf, data, dpdata, code, imgLen, imgDir, labeltime,rnd);
				timestr = data[imgLen + labeltime - 2].time;
			}
		}
		lf.close();
	}


	void split_label( int  valRate, int  testRate,
		       const char* allLabel, const char* trainLabel, const char* testlabel, const char* valLabel) {

		std::ifstream allfile;
		std::ofstream trainfile, testfile, valfile;

		allfile.open(allLabel, std::ios::in);
		if (!allfile.is_open()){
			printf("open file %s error!\n", allLabel);
			return;
		}

		trainfile.open(trainLabel, std::ios::out | std::ios::trunc);
		if (!trainfile.is_open()){
			printf("open file %s error!\n", trainLabel);
			return;
		}

		testfile.open(testlabel, std::ios::out | std::ios::trunc);
		if (!testfile.is_open()){
			printf("open file %s error!\n", testlabel);
			return;
		}

		valfile.open(valLabel, std::ios::out | std::ios::trunc);
		if (!valfile.is_open()){
			printf("open file %s error!\n", valLabel);
			return;
		}

		std::string name;
		int total[3] = { 0 };

		while (!allfile.eof()) {
			std::string str;
			std::getline(allfile,str);
			if (allfile.fail()) break;

			int select = rand() % 100;
			if (select >= valRate) {
				valfile << str << std::endl;
				total[0]++;
			}
			else if (select >= testRate) {
				testfile << str << std::endl;
				total[1]++;
			}
			else {
				trainfile << str << std::endl;
				total[2]++;
			}
		}
		allfile.close();
		valfile.close();
		testfile.close();
		trainfile.close();

		printf("all = %d, train = %d, test = %d, val =%d \n", total[0] + total[1] + total[2], total[2], total[1], total[0]);
	}
};