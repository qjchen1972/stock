#ifndef _STOCK__H
#define _STOCK__H

#include<opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include<math.h>
#include <set>

using namespace std;
using namespace cv;

struct tdx_t
{
	unsigned int date;
	int  open; //开盘价 *0.01
	int  high; //最高价 *0.01
	int  low;  //最低价 *0.01
	int  close; //收盘价*0.01
	float amount; //成交金额
	int vol;  //成交量(手数)
	int reserved;//保留
};


class Stock
{
public:
	Stock(int h = 128,int w= 128, int circle=32)
	{
		m_height = h;
		m_width = w;
		m_circle_day = circle;
		init(h,w,circle);
	}
	~Stock(){}

	void clear_file(const char* label_file)
	{
		set <string> ss;
		string textline;
		set <string>::iterator iter;
		ifstream infile;
		ofstream ofile;

		infile.open(label_file);
		while (getline(infile, textline))
			ss.insert(textline);
		infile.close();
		ofile.open(label_file);
		int type;
		char str[128];
		int total0 = 0;
		int total1 = 0;
		int  total2 = 0;
		int all = 0;
		for (iter = ss.begin(); iter != ss.end(); iter++)
		{
			sscanf((*iter).c_str(), "%s  %d", str,&type);
			all++;
			if (type == 0) total0++;
			else if(type == 1) total1++;
			else total2++;

			ofile << *iter << endl;
		}
		ofile.close();
		printf("all = %d, ( %d   %d   %d ) \n", all, total0, total1, total2);
	}

	void split(const char* label_file, const char* telabel_file)
	{
		set <string> ss;
		string textline;
		set <string>::iterator iter;
		ifstream infile;
		ofstream ofile, tfile;

		infile.open(label_file);
		while (getline(infile, textline))
			ss.insert(textline);
		infile.close();
		ofile.open(label_file);
		tfile.open(telabel_file);

		int type;
		char str[128];
		int total0 = 0;
		int total1 = 0;
		int  total2 = 0;
		int total3 = 0;
		int  total4 = 0;
		int  total5 = 0;

		int all = 0;
		int te0 = 0;
		int te1 = 0;
		int  te2 = 0;
		int te3 = 0;
		int  te4 = 0;
		int  te5 = 0;

		srand(time(NULL));
		for (iter = ss.begin(); iter != ss.end(); iter++)
		{
			sscanf((*iter).c_str(), "%s  %d", str, &type);
			all++;
			if (rand() % 100 >= 99)
			{
				if (type == 0) te0++;
				else if (type == 1) te1++;
				else if (type == 2) te2++;
				else if (type == 3) te3++;
				else if (type == 4) te4++;
				else te5++;
				tfile << *iter << endl;
			}
			else
			{
				if (type == 0) total0++;
				else if (type == 1) total1++;
				else if (type == 2) total2++;
				else if (type == 3) total3++;
				else if (type == 4) total4++;
				else total5++;
				ofile << *iter << endl;
			}
		}
		ofile.close();
		printf("all = %d, tr( %d   %d   %d  %d  %d  %d) te( %d   %d   %d  %d  %d  %d) \n", all, total0, total1, total2,
			total3, total4, total5,te0,te1,te2, te3, te4,te5);
	}

	/*void create_randtrain_data(const char* label_file, const char* img_dir,int tr_cycle, int ans_cycle, int num )
	{
		srand(time(NULL));

		ofstream lf;
		lf.open(label_file, ios::out | ios::trunc);
		if (!lf.is_open())
		{
			printf("open file %s error!\n", label_file);
			return;
		}
		int count = 0;
		int code = 0;
		char prefix[128];
		int len = tr_cycle * m_circle_day + ans_cycle * 3;
		int ret;

		tdx_t *stock = new tdx_t[len];
		tdx_t *img_data = new tdx_t[m_circle_day];
		tdx_t  ans_data[3];
		char filename[128];
		char imgname[128];

		while (count < num)
		{
			get_randcode(code, prefix);
			ret = get_data_with_code(code, prefix, stock, len, 1);
			if (!ret) continue;
			if (!is_valid(stock, len)) continue;
			merge(stock, tr_cycle * m_circle_day, img_data, m_circle_day);
			merge(stock + tr_cycle * m_circle_day, ans_cycle * 3, ans_data, 3);

			if (!is_wave(img_data, m_circle_day)) continue;

			Mat  img;
			create_oneimg(img_data,img);
			int type = get_imgtype(img_data[m_circle_day-1],ans_data[2]);

			sprintf_s(filename, "%s/%06d_%d.jpg", img_dir,code, img_data[m_circle_day-1].date);
			sprintf_s(imgname, "%06d_%d.jpg", code, img_data[m_circle_day-1].date);
			imwrite(filename, img);
			lf << imgname << "  " << type << endl;
			m_type[type]++;
			count++;
		}
		delete[] img_data;
		delete[] stock;

		lf.close();
		printf("type 0 == %d, 1 == %d  2 == %d\n", m_type[0], m_type[1], m_type[2]);
	}*/

	void create_serialdata(const char* label_file, const char* img_dir, int tr_cycle, int ans_cycle, int num)
	{
		ofstream lf;
		lf.open(label_file, ios::out | ios::trunc);		
		if (!lf.is_open())
		{
			printf("open label file error!\n");
			return;
		}

		char prefix[128];		
		int one_len = tr_cycle * m_circle_day + ans_cycle;
		int len = num - 1 + one_len;
		int ret;

		tdx_t *stock = new tdx_t[len];
		tdx_t *img_data = new tdx_t[m_circle_day];
		tdx_t *dp_stock = new tdx_t[tr_cycle * m_circle_day];
		tdx_t *dp_data = new tdx_t[m_circle_day];
		tdx_t  ans_data;
		char filename[128];
		char imgname[128];

		int start[] = { 1 ,300001 ,600000 ,603000 };
		int  end[] = { 2892, 300691 ,601999,603999 };
		for (int i = 0; i < 4; i++)
		{
			for (int code = start[i]; code <= end[i]; code++)
			{
				len = num - 1 + one_len;
				if( i <= 1 )
					ret = get_data_with_code(code, sz_prefix, stock, len, 0);
				else
					ret = get_data_with_code(code, sh_prefix, stock, len, 0);

				if (!ret || len < one_len) continue;
				
				for (int k = 0; k < len - one_len + 1; k++)
				{
					//if (k > 0) break;
					if (!is_valid(stock + k, one_len))
					{
						//printf("%06d  %d is not valid \n", code,stock[k].date);
						continue;
					}
					merge(stock + k, tr_cycle * m_circle_day, img_data, m_circle_day);
					if (ans_cycle)
						merge(stock + k + tr_cycle * m_circle_day, ans_cycle, &ans_data, 1);
					if (img_data[m_circle_day - 1].date > 20170815) continue;
					//if (img_data[m_circle_day - 1].date > 20170915 ||img_data[m_circle_day - 1].date <= 20170815) continue;
					//if (img_data[m_circle_day - 1].date <= 20170915) continue;
					//if (!is_wave(img_data, m_circle_day)) continue;
					if (!is_max_wave(img_data, m_circle_day)) continue;
					if (i <= 1)
					{
						if (!get_dapan(stock + k, m_sz_dapan, m_sz_dapan_len, dp_stock, tr_cycle * m_circle_day))
							continue;
					}
					else
					{
						if (!get_dapan(stock + k, m_sh_dapan, m_sh_dapan_len, dp_stock, tr_cycle * m_circle_day))
							continue;
					}
					merge(dp_stock, tr_cycle * m_circle_day, dp_data, m_circle_day);

					Mat  img;
					create_oneimg(img_data, dp_data,img);
					int type = 0;
					if (ans_cycle)
						type = get_imgtype(img_data[m_circle_day - 1], stock + k + tr_cycle * m_circle_day, ans_cycle);
						//type = get_imgtype(img_data[m_circle_day - 1], ans_data);
					sprintf_s(filename, "%s/%06d_%d.jpg", img_dir, code, img_data[m_circle_day - 1].date);
					sprintf_s(imgname, "%06d_%d.jpg", code, img_data[m_circle_day - 1].date);
					//if (img_data[m_circle_day - 1].date < 20180801) continue;
					imwrite(filename, img);
					lf << imgname << "  " << type << endl;
					m_type[type]++;
				}
			}
		}
		delete[] img_data;
		delete[] stock;
		lf.close();
		float yy = (m_type[5] - m_type[0]) * 3 + (m_type[4] - m_type[1]) * 2 + (m_type[3] - m_type[2]) * 0.5;
		int  tt = m_type[5] + m_type[4] + m_type[3] + m_type[2] + m_type[1] + m_type[0];

		printf("type 0 == %d, 1 == %d  2 == %d  3 == %d   4 == %d  5 == %d    %d   %f\n", 
			m_type[0], m_type[1], m_type[2], m_type[3], m_type[4], m_type[5], tt, yy/tt);

		//float yy = m_type[2] - m_type[0];
		//int  tt = m_type[2] + m_type[1] + m_type[0];

		//printf("type 0 == %d, 1 == %d  2 == %d      %d   %f\n", 
			//m_type[0], m_type[1], m_type[2], tt, yy/tt);

	}

	int init(int h = 128, int w = 128, int circle = 32)
	{
		m_height = h;
		m_width = w;
		m_circle_day = circle;
		if (!init_dapan(sh_dapan, &m_sh_dapan, m_sh_dapan_len)) return 0;
		if (!init_dapan(sz_dapan, &m_sz_dapan, m_sz_dapan_len)) return 0;
		m_split_line = 72;
		m_up_vol = 56;
		m_down_vol = 116;
		m_dis = m_width / circle;
		m_squ = 3;
		return 1;
	}
	
private:
	const char* sh_prefix= "D:/soft_list/tdx/vipdoc/sh/lday/sh"; 
	const char* sz_prefix = "D:/soft_list/tdx/vipdoc/sz/lday/sz";
	const char* sh_dapan = "D:/soft_list/tdx/vipdoc/sh/lday/sh000001.day";
	const char* sz_dapan = "D:/soft_list/tdx/vipdoc/sz/lday/sz399001.day";

	tdx_t *m_sh_dapan = nullptr;
	tdx_t *m_sz_dapan = nullptr;
	int   m_sh_dapan_len;
	int   m_sz_dapan_len;

	int m_circle_day = 32;
	int m_width = 128;
	int m_height = 128;

	int m_split_line = 80;
	int m_up_vol = 65;
	int m_down_vol = 112;

	
	int m_squ;
	int m_dis;
	int m_type[6] = {0,0,0,0,0,0};	

	

	int init_dapan(const char *dapan_file, tdx_t **st,  int &st_len)
	{
		ifstream f;

		f.open(dapan_file, ios::in | ios::binary | ios::ate);
		if (!f.is_open())
		{
			printf("open %s error\n", dapan_file);
			return 0;
		}
		int size = (int)f.tellg();
		f.seekg(0, ios::beg);
		int  dpsh_num = size / sizeof(tdx_t);
		*st = new tdx_t[dpsh_num];
		f.read((char*)*st, size);
		f.close();
		st_len = dpsh_num;
		return 1;
	}

	int get_dapan(tdx_t *img, tdx_t *src, int src_len, tdx_t *dest, int dest_len)
	{
		int start = 0;
		int end = src_len - 1;
		int mid;
		while (1)
		{
			if (start == end)
			{
				if (src[start].date == img[0].date)
				{
					mid = start;
					break;
				}
				else return 0;
			}
			else if (start > end) return 0;
			mid = start + (end  - start) / 2;	
			
			if (src[mid].date == img[0].date) break;
			if (src[mid].date < img[0].date) start = mid + 1;
			else end = mid - 1;
		}
		if (src_len - mid < dest_len) return 0;
		int i = 0;
		int k = 0;
		while (mid + k < src_len && i < dest_len)
		{
			if (img[i].date != src[mid + k].date)
			{
				k++;
				continue;
			}
			memcpy((char*)&dest[i], (char*)&src[mid+k], sizeof(tdx_t));
			i++;
			k++;
		}
		if (i < dest_len) return 0;
		//memcpy((char*)dest, (char*)(src+mid), dest_len * sizeof(tdx_t));
		return 1;
	}

	int get_data_with_code(int code, const char *prefix, tdx_t *stock, int &len,int type)
	{
		ifstream f;
		char filename[128];
		sprintf_s(filename, "%s%06d.day", prefix, code);

		f.open(filename, ios::in | ios::binary | ios::ate);
		if (!f.is_open())
		{
			//printf("open %s error\n", filename);
			return 0;
		}
		int size = (int)f.tellg();
		int  stock_num = size / sizeof(tdx_t);
		if (stock_num < len)
		{
			len = stock_num;
		}
		int start;
		if (type == 0)
		{
			start = len;
		}
		else
		{
			int real = stock_num - len + 1;
			start = rand() % real + len;
		}	
		size = start * sizeof(tdx_t);
		f.seekg(-size, ios::end);
		f.read((char*)stock, len* sizeof(tdx_t));
		f.close();
		return 1;
	}

	void get_randcode(int &code, char *prefix)
	{
		int  type = rand() % 4;
		if (type == 0)
		{
			code = rand() % 2892 + 1;
			strcpy(prefix, sz_prefix);
		}
		else if (type == 1)
		{
			code = rand() % 691 + 300001;
			strcpy(prefix, sz_prefix);
		}
		else if (type == 2)
		{
			code = rand() % 2000 + 600000;
			strcpy(prefix, sh_prefix);
		}
		else
		{
			code = rand() % 1000 + 603000;
			strcpy(prefix, sh_prefix);
		}
	}

	int get_imgtype(tdx_t last, tdx_t now3/*, tdx_t now5, tdx_t now7*/)
	{
		float temp = 1.0 *(now3.close - last.close) / last.close;
		int p1;

		if (temp < -0.03) p1 = 0;
		else if (temp < -0.01) p1 = 1;
		else if (temp < 0.0) p1 = 2;
		else if(temp < 0.01) p1 = 3;
		else if (temp < 0.03) p1 = 4;
		else p1 = 5;
		int type = p1;
		return type;
	}

	int get_imgtype(tdx_t last, tdx_t *now,int size)
	{
		float total = now[0].close;
		for (int i = 1; i < size; i++)
		{
			total += now[i].close;
		}
		total = total / size;

		float temp = 1.0 *(total - last.close) / last.close;
		int p1;

		if (temp < -0.03) p1 = 0;
		else if (temp < -0.01) p1 = 1;
		else if (temp < 0.0) p1 = 2;
		else if (temp < 0.01) p1 = 3;
		else if (temp < 0.03) p1 = 4;
		else p1 = 5;
		int type = p1;
		return type;
	}

	int is_valid(tdx_t *stock, int size)
	{
		for (int i = 0; i < size - 1; i++)
		{
			float temp = 1.0*(stock[i].close - stock[i + 1].open) / stock[i].close;
			if (temp > 0.101)
			{
				//printf("it is %d \n ", stock[i].date);
				return 0;
			}
		}	
		return 1;
	}

	int is_onewave(tdx_t *stock, int size, int index)
	{
		if (index >= size || index - 3 < 0 )  return 0;
		if (stock[index - 2].close >= stock[index - 3].close) return 0;
		if (stock[index - 1].close >= stock[index - 2].close) return 0;
		if (stock[index].close < stock[index - 1].close) return 0;
		return 1;
	}

	int next_wave(tdx_t *stock, int size, int index)
	{
		for (int i = index + 1; i < size; i++)
		{
			if (is_onewave(stock, size, i)) return i;
		}
		return  -1;
	}

	int is_max_wave(tdx_t *stock, int size)
	{
		int max;
		if (!is_onewave(stock, size, size - 1)) return 0;
		get_max_close(stock, size, max);
		if (next_wave(stock, size, max) != size - 1) return 0;
		return 1;	
	}

	
	int is_wave(tdx_t *stock, int size)
	{
		/*int vol = stock[size - 1].vol;
		//if (stock[size - 1].vol < vol)  return 0;
		//if (stock[size - 1].close < stock[size - 1].open) return 0;
		for (int i = size - 2; i >= 0; i--)
		{
			if (stock[i].vol < vol)
			{
				return 0;
			}
		}

		if (stock[size - 1].close == stock[size - 1].open && stock[size - 1].close == stock[size - 1].high &&
			stock[size - 1].close == stock[size - 1].low) return 0;
			*/

		if (stock[size - 2].close >= stock[size - 3].close) return 0;
		if (stock[size - 1].close < stock[size - 2].close)
		{
		if (stock[size - 1].close <= stock[size - 1].open) return 0;
		if (stock[size - 3].close - stock[size - 3].close * 0.03 <= stock[size - 1].close) return 0;
		}
		else
		{
		if (stock[size - 3].close >= stock[size - 4].close) return 0;
		if (stock[size - 2].close > stock[size - 2].open) return 0;
		if (stock[size - 4].close - stock[size - 4].close * 0.03 <= stock[size - 2].close) return 0;
		}


		//if (stock[size - 2].close >= stock[size - 3].close) return 0;
		//if (stock[size - 3].close >= stock[size - 4].close) return 0;
		//if (stock[size - 1].close < stock[size - 2].close) return 0;
		//if (stock[size - 4].close - stock[size - 4].close * 0.03 <= stock[size - 2].close) return 0;

		return 1;
	}

	void merge(tdx_t *src, int src_size, tdx_t *dest, int dst_size)
	{
		int size = src_size / dst_size;
		for (int i = 0; i < src_size; i = i + size)
		{
			int k = i / size;
			dest[k].open = src[i].open;
			dest[k].close = src[i + size - 1].close;
			dest[k].vol = src[i].vol;
			dest[k].high = src[i].high;
			dest[k].low = src[i].low;
			dest[k].date = src[i + size - 1].date;
			for (int j = 1; j < size; j++)
			{
				dest[k].vol += src[i + j].vol;
				if (src[i + j].high > dest[k].high) dest[k].high = src[i + j].high;
				if (src[i + j].low < dest[k].low) dest[k].low = src[i + j].low;
			}
		}
	}

	void get_max(tdx_t *stock_tdx, int &maxvol, int &min_p, int &max_p)
	{
		maxvol = stock_tdx[0].vol;
		min_p = stock_tdx[0].low;
		max_p = stock_tdx[0].high;

		for (int i = 1; i < m_circle_day; i++)
		{
			if (stock_tdx[i].vol > maxvol) maxvol = stock_tdx[i].vol;
			if (stock_tdx[i].low < min_p) min_p = stock_tdx[i].low;
			if (stock_tdx[i].high > max_p) max_p = stock_tdx[i].high;
		}
	}

	void get_max_close(tdx_t *stock_tdx, int size, int &max_p)
	{		
		max_p = 0;

		for (int i = 1; i < size; i++)
		{
			if (stock_tdx[i].close > stock_tdx[max_p].close) max_p = i;
		}
	}

	void create_oneimg(tdx_t *one_data, tdx_t *dp_data,Mat &dst)
	{
		int max_vol, min_p, max_p;
		Mat img(m_height, m_width, CV_8UC3, Scalar::all(0));
		get_max(one_data, max_vol, min_p, max_p);
		
		int vol = m_split_line - m_up_vol - 2;
		int price = m_up_vol - 2;

		float mean_price = 0;
		float mean_vol = 0;

		for (int k = 0; k < m_circle_day; k++)
		{
			int volheight = m_split_line - round(1.0*one_data[k].vol * vol / max_vol);
			int open = m_up_vol - round(1.0*(one_data[k].open - min_p) * price / (max_p - min_p));
			int close = m_up_vol - round(1.0*(one_data[k].close - min_p) * price / (max_p - min_p));
			int high = m_up_vol - round(1.0*(one_data[k].high - min_p) * price / (max_p - min_p));
			int low = m_up_vol - round(1.0*(one_data[k].low - min_p) * price / (max_p - min_p));
			int prev_price = m_up_vol - round( 1.0*(mean_price - min_p)* price / (max_p - min_p));
			mean_price = 0;
			int count = 0;
			for (int m = k; m >= 0; m--)
			{
				mean_price += one_data[m].close;
				count++;
				if (count >= 5) break;
			}
			mean_price = mean_price / count;

			//mean_price = 1.0*(k *mean_price + one_data[k].close) / (k + 1);
			int now_price = m_up_vol - round(1.0*(mean_price - min_p)* price / (max_p - min_p));

			int prev_vol = m_split_line - round(1.0*mean_vol * vol / max_vol);
			mean_vol = 0;
			count = 0;
			for (int m = k; m >= 0; m--)
			{
				mean_vol += one_data[m].vol;
				count++;
				if (count >= 5) break;
			}
			mean_vol = mean_vol / count;
			//mean_vol = 1.0 *(k *mean_vol + one_data[k].vol) / (k + 1);
			int now_vol = m_split_line - round(1.0*mean_vol * vol / max_vol);

			if (one_data[k].close >= one_data[k].open)
			{
				rectangle(img, cvPoint(k * m_dis, volheight), cvPoint(k * m_dis + m_squ - 1, m_split_line), Scalar(127, 127, 127), CV_FILLED);
				rectangle(img, cvPoint(k * m_dis, close), cvPoint(k * m_dis + m_squ - 1, open), Scalar(127, 127, 127), CV_FILLED);
				line(img, Point(k * m_dis + 1, high), Point(k * m_dis + 1, close), Scalar(127, 127, 127), 1);
				line(img, Point(k * m_dis + 1, low), Point(k * m_dis + 1, open), Scalar(127, 127, 127), 1);
			}
			else
			{
				rectangle(img, cvPoint(k * m_dis, volheight), cvPoint(k * m_dis + m_squ - 1, m_split_line), Scalar(255, 255, 255), CV_FILLED);
				rectangle(img, cvPoint(k * m_dis, open), cvPoint(k * m_dis + m_squ - 1, close), Scalar(255, 255, 255), CV_FILLED);
				line(img, Point(k * m_dis + 1, high), Point(k * m_dis + 1, open), Scalar(255, 255, 255), 1);
				line(img, Point(k * m_dis + 1, low), Point(k * m_dis + 1, close), Scalar(255, 255, 255), 1);
			}
			if (k != 0)
			{
				line(img, Point((k - 1) * m_dis + 1, prev_price), Point(k * m_dis + 1, now_price), Scalar(64, 64, 64), 1, 4);
				line(img, Point((k - 1) * m_dis + 1, prev_vol), Point(k * m_dis + 1, now_vol), Scalar(196, 196, 196), 1, 4);

			}
		}

		get_max(dp_data, max_vol, min_p, max_p);

		vol = m_height - m_down_vol - 2;
		price = m_down_vol - m_split_line - 3;

		for (int k = 0; k < m_circle_day; k++)
		{
			int volheight = m_height - round(1.0*dp_data[k].vol * vol / max_vol);
			int open = m_down_vol - round(1.0*(dp_data[k].open - min_p) * price / (max_p - min_p));
			int close = m_down_vol - round(1.0*(dp_data[k].close - min_p) * price / (max_p - min_p));
			int high = m_down_vol - round(1.0*(dp_data[k].high - min_p) * price / (max_p - min_p));
			int low = m_down_vol - round(1.0*(dp_data[k].low - min_p) * price / (max_p - min_p));

			int prev_price = m_down_vol - round(1.0*(mean_price - min_p)* price / (max_p - min_p));
			mean_price = 0;
			int count = 0;
			for (int m = k; m >= 0; m--)
			{
				mean_price += dp_data[m].close;
				count++;
				if (count >= 5) break;
			}
			mean_price = mean_price / count;
			//mean_price = 1.0*(k *mean_price + dp_data[k].close) / (k + 1);
			int now_price = m_down_vol - round(1.0*(mean_price - min_p)* price / (max_p - min_p));

			int prev_vol = m_height - round(1.0*mean_vol * vol / max_vol);
			mean_vol = 0;
			count = 0;
			for (int m = k; m >= 0; m--)
			{
				mean_vol += dp_data[m].vol;
				count++;
				if (count >= 5) break;
			}
			mean_vol = mean_vol / count;
			//mean_vol = 1.0*(k *mean_vol + dp_data[k].vol) / (k + 1);
			int now_vol = m_height - round(1.0*mean_vol * vol / max_vol);

			if (dp_data[k].close >= dp_data[k].open)
			{
				rectangle(img, cvPoint(k * m_dis, volheight), cvPoint(k * m_dis + m_squ - 1, m_height), Scalar(127, 127, 127), CV_FILLED);
				rectangle(img, cvPoint(k * m_dis, close), cvPoint(k * m_dis + m_squ - 1, open), Scalar(127, 127, 127), CV_FILLED);
				line(img, Point(k * m_dis + 1, high), Point(k * m_dis + 1, close), Scalar(127, 127, 127), 1);
				line(img, Point(k * m_dis + 1, low), Point(k * m_dis + 1, open), Scalar(127, 127, 127), 1);
			}
			else
			{
				rectangle(img, cvPoint(k * m_dis, volheight), cvPoint(k * m_dis + m_squ - 1, m_height), Scalar(255, 255, 255), CV_FILLED);
				rectangle(img, cvPoint(k * m_dis, open), cvPoint(k * m_dis + m_squ - 1, close), Scalar(255, 255, 255), CV_FILLED);
				line(img, Point(k * m_dis + 1, high), Point(k * m_dis + 1, open), Scalar(255, 255, 255), 1);
				line(img, Point(k * m_dis + 1, low), Point(k * m_dis + 1, close), Scalar(255, 255, 255), 1);
			}
			if (k != 0)
			{
				line(img, Point((k - 1) * m_dis + 1, prev_price), Point(k * m_dis + 1, now_price), Scalar(64, 64, 64), 1, 4);
				line(img, Point((k - 1) * m_dis + 1, prev_vol), Point(k * m_dis + 1, now_vol), Scalar(196, 196, 196), 1, 4);
			}
		}
		//imshow("ok", img);
		//waitKey(0);
		img.copyTo(dst);
	}
};

#endif
