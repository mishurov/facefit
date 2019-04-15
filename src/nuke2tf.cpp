/* ************************************************************************
 * Copyright 2019 Alexander Mishurov
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************/

#include "nuke2tf.h"

#include <dlib/image_transforms.h>
#include <fstream>
#include <cmath>

using namespace dlib;
using namespace tensorflow;


Nuke2TensorFlow::Nuke2TensorFlow(int resolution)
{
	_resolution = resolution;
	_detector = get_frontal_face_detector();
	_destPoints = {
		{ 0, 0 },
		{ 0, _resolution - 1},
		{ _resolution - 1, 0},
	};
	_points.resize(_resolution * _resolution);
	_planeHeight = 0;
}


Nuke2TensorFlow::StaticData::StaticData(const std::string& detectorModelPath,
			const std::string& trianglesPath,
			const std::string& faceIndicesPath,
			const std::string& kptIndicesPath,
			int resolution)
{
	std::cout << "Loading the detector model...\n";
	deserialize(detectorModelPath) >> net;

	std::cout << "Loading the indices data...\n";
	readIndices(faceIndicesPath, _faceIndices);
	for (int i = 0; i < _faceIndices.size(); i++)
		_face2all[i] = _faceIndices[i];

	std::vector<int> kptIndices2d;
	readIndices(kptIndicesPath, kptIndices2d);
	int size = kptIndices2d.size() / 2;
	for (int i = 0; i < size; i++) {
		int x = kptIndices2d.at(i + size);
		int y = kptIndices2d.at(i);
		_kptIndices.push_back(x * resolution + y);
	}
	readIndices(trianglesPath, _triIndices);

	std::cout << "Generating default points and UVs...\n";
	int numPoints = resolution * resolution;
	_uvs.resize(numPoints);
	_defaultPoints.resize(numPoints);

	for (int i = 0; i < resolution; i++) {
		for (int j = 0; j < resolution; j++) {
			int i_flat = i * resolution + j;

			_defaultPoints.at(i_flat).set(i * 2, j * 2, 0);

			_uvs[i_flat] = DD::Image::Vector3(
				(float)j / (float)resolution,
				1 - (float)i / (float)resolution,
				0.0
			);

		}
	}

	_endList = { 16, 21, 26, 41, 47, 30, 35, 67 };
}


void Nuke2TensorFlow::StaticData::readIndices(const std::string& path,
						std::vector<int>& indices)
{
	std::fstream ifs;
	ifs.open(path);
	if (!ifs) {
                std::cout << "Error reading indices file.\n";
                return;
        }
	double index;
	while(ifs >> index) {
		indices.push_back((int)index);
	}
	ifs.clear();
	ifs.close();
}


unsigned char Nuke2TensorFlow::linear2srgb(float c)
{
	if (c > 0.0031308)
		c = 1.055 * (std::pow(c, (1.0 / 2.4))) - 0.055;
	else
		c = 12.92 * c;
	return clamp((int)(c * 255), 0, 255);
}


void Nuke2TensorFlow::extractDataFromTensor(Tensor& tensor)
{
	auto et = tensor.flat_inner_dims<float, 3>();
	
	// this coefficient 1.1, and the coefficients below for expanding
	// a facial bounding box were taken from PRNet's Python code,
	// I've no idea whether they're empirical or have a precise meaning
	float mult = (float)_resolution * 1.1;
	float frac = mult / _pointTransform.get_m()(0, 0);

	_pointTransform = inv(_pointTransform);
	
	parallel_for(size_t(0), _resolution, [&](size_t i) {
	    for (int j = 0; j < _resolution; j++) {
		float x = et(i, j, 0) * mult;
		float y = et(i, j, 1) * mult;
		float z = et(i, j, 2) * frac;

		vector<double, 2> v = {x, y};
		v = _pointTransform({x, y});

		x = v(0);
		y = _planeHeight - 1 - v(1);
		
		_points.at(i * _resolution + j).set(x, y, z);
	    }
	});
}


void Nuke2TensorFlow::plane2img(const DD::Image::ImagePlane& plane,
						matrix<rgb_pixel>& img)
{
	auto ipBox = plane.bounds();
	int x = ipBox.x(), t = ipBox.t(), r = ipBox.r(),
	    y = ipBox.y(), h = ipBox.h(), w = ipBox.w();
	_planeHeight = (float)h;
	_planeWidth = (float)w;

	img.set_size(h, w);

	parallel_for(size_t(1), h + 1, [&](size_t i) {
		for (int j = 0; j < w; j++) {
			unsigned char r = linear2srgb(plane.at(x + j, t - i, 0));
			unsigned char g = linear2srgb(plane.at(x + j, t - i, 1));
			unsigned char b = linear2srgb(plane.at(x + j, t - i, 2));
			img(i - 1, j) = rgb_pixel(r, g, b);
		}
	});
}


Tensor Nuke2TensorFlow::extractFaceTensor(const matrix<rgb_pixel>& inImg,
					int l, int r, int t, int b,
					bool detected)
{
	//std::cout << "Processing image data for TensorFlow...\n";
	int c[2] = { r - (r - l) / 2, b - (b - t) / 2 };
	int bboxSize = (r - l + b - t) / 2;

	int halfSize;
	if (detected) {
		int size = (int)((float)bboxSize * 1.6);
		halfSize = size / 2;
	} else {
		halfSize = bboxSize / 2;
	}
	
	WarpPointList srcPoints = {
		{ c[0] - halfSize, c[1] - halfSize },
		{ c[0] - halfSize, c[1] + halfSize },
		{ c[0] + halfSize, c[1] - halfSize },
	};

	matrix<rgb_pixel> outImg;
	outImg.set_size(_resolution, _resolution);

	_pointTransform = find_affine_transform(srcPoints, _destPoints);
	transform_image(inImg, outImg,
				interpolate_quadratic(), inv(_pointTransform));

	return img2Tensor(outImg);
}


Tensor Nuke2TensorFlow::img2Tensor(const matrix<rgb_pixel>& img)
{
	Tensor tensor(DT_FLOAT, TensorShape({1, _resolution, _resolution, 3}));
	auto tensorMapped = tensor.tensor<float, 4>();

	parallel_for(size_t(0), _resolution, [&](size_t y) {
	    for (int x = 0; x < _resolution; ++x) {
		rgb_pixel p = img(y, x);
		tensorMapped(0, y, x, 0) = (float)p.red / 255.;
		tensorMapped(0, y, x, 1) = (float)p.green / 255.;
		tensorMapped(0, y, x, 2) = (float)p.blue / 255.;
	    }
	});
	return tensor;
}


Tensor Nuke2TensorFlow::imagePlane2Tensor(const DD::Image::ImagePlane& plane,
					const DD::Image::Box& userBBox,
					bool useDetector)
{
	matrix<rgb_pixel> img;
	plane2img(plane, img);

	if (useDetector) {
		//std::cout << "Detecting faces...\n";

		matrix<rgb_pixel> imgP(img);
		pyramid_down<2> pyr;

		unsigned int levels = _upsample;
        	while (levels > 0) {
			levels--;
			pyramid_up(imgP, pyr);
		}
		//auto dets = data.net(imgP);
		// HOG detector
		auto dets = _detector(imgP);

		if (dets.size() < 1) {
			std::cout << "No faces found.\n";
			return Tensor();
		}
		//auto detBBox = pyr.rect_down(dets.at(0).rect, _upsample);
		// HOG detector
		auto detBBox = pyr.rect_down(dets.at(0), _upsample);

		return extractFaceTensor(img, detBBox.left(), detBBox.right(),
				detBBox.top(), detBBox.bottom(), true);
	}
	auto ipBBox = plane.bounds();

	//std::cout << "Using the provided bounding box for inference.\n";
	return extractFaceTensor(img, userBBox.x(), userBBox.r(),
				ipBBox.h() - userBBox.t(),
				ipBBox.h() - userBBox.y(), false);
}
