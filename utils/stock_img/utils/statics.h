#pragma once
#include <iostream>
#include <fstream>
#include<math.h>
#include"Astock.h"
#include <Eigen/Dense>
#include <Eigen/SVD>

class Statics {
public:
	Statics() {}
	~Statics() {}	

	void vec2matix(std::vector<std::vector<float>> data, Eigen::MatrixXf &m) {
		m.resize(data.size(), data[0].size());
		for (int i = 0; i < data.size(); i++)
			m.row(i) = Eigen::VectorXf::Map(&data[i][0], data[i].size());
		std::cout << "Here is the matrix m:" << std::endl << m << std::endl;
	}

	/*
	协方差矩阵
	*/
	void covmat(Eigen::MatrixXf m, Eigen::MatrixXf &covm) {
	
		std::cout << "Here is the matrix m:" << std::endl << m << std::endl;
		Eigen::MatrixXf meanVec = m.colwise().mean();		
		Eigen::RowVectorXf meanVecRow(Eigen::RowVectorXf::Map(meanVec.data(), m.cols()));

		Eigen::MatrixXf zeroMeanMat = m;
		zeroMeanMat.rowwise() -= meanVecRow;
		if (m.rows() == 1)
			covm = (zeroMeanMat.adjoint()*zeroMeanMat) / double(m.rows());
		else
			covm = (zeroMeanMat.adjoint()*zeroMeanMat) / double(m.rows() - 1);

		std::cout << "Here is the covmat:" << std::endl << covm << std::endl;
	}

	/*
	svd分解,返回近似矩阵
	*/
	void svd(Eigen::MatrixXf m, int size, Eigen::MatrixXf &svdmat) {

		std::cout << "Here is the matrix m:" << std::endl << m << std::endl;
		Eigen::JacobiSVD<Eigen::MatrixXf> svd(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
		
		Eigen::VectorXf sgu = svd.singularValues();
		Eigen::MatrixXf smat(sgu.rows(), sgu.rows());
		smat.setZero();
		for (int i = 0; i < size; i++)
			smat(i, i) = sgu(i);
		Eigen::MatrixXf sVec = smat.block(0, 0, size, size);
		std::cout << "Its singular values are:" << std::endl << sgu << std::endl;
		//std::cout << "Its singular values are:" << std::endl << smat << std::endl;

		Eigen::MatrixXf U = svd.matrixU();
		Eigen::MatrixXf UVec = U.block(0, 0, U.rows(), size);
		std::cout << "Its left singular vectors are the columns of the thin U matrix:" << std::endl << U << std::endl;
		//std::cout << "Its left singular vectors are the columns of the thin U matrix:" << std::endl << UVec << std::endl;


		Eigen::MatrixXf V = svd.matrixV();
		Eigen::MatrixXf VVec = V.block(0, 0, V.rows(), size);
		std::cout << "Its right singular vectors are the columns of the thin V matrix:" << std::endl << V << std::endl;
		//std::cout << "Its right singular vectors are the columns of the thin V matrix:" << std::endl << VVec << std::endl;

		svdmat = UVec * sVec * VVec.transpose();
		std::cout << "Here is the matrix X:" << std::endl << svdmat << std::endl;

	}	

private:		

};

