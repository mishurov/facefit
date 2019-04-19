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


// http://www.codeincodeblock.com/2013/07/modern-opengl3d-model-obj-loaderparser.html
Nuke2TensorFlow::StaticData::StaticData(const std::string& faceObjPath)
{
	std::cout << "Loading the canonical face obj...\n";
	std::ifstream in(faceObjPath, std::ios::in);
	if (!in) {
		std::cerr << "Error opening file " << faceObjPath << "\n";
		return;
	}
	std::string line;
	std::vector<int> faceUvIs;
	std::vector<double> unorderedUvs;
	while (std::getline(in, line)) {
		if (line.substr(0, 2) == "v ") {
			std::istringstream v(line.substr(2));
			double x, y, z;
			v >> x;
			v >> y;
			v >> z;
			_defaultPoints.push_back(DD::Image::Vector3(x, y ,z));
		}
		else if (line.substr(0, 2) == "vt") {
			std::istringstream v(line.substr(3));
			double U, V;
			v >> U;
			v >> V;
			unorderedUvs.push_back(U);
			unorderedUvs.push_back(V);
		}
		else if (line.substr(0, 2) == "f ") {
			std::istringstream v(line.substr(2));
			for (int i = 0; i < 3; i ++) {
				int index, uv;
				v >> index;
				v.get();
				v >> uv;
				_faceIndices.push_back(--index);
				faceUvIs.push_back(--uv);
			}
		}
	}
	// It's subpoptimal, because it assigns the same data several times
	// yet it isn't crucial since the method is called only once
	_uvs.resize(_defaultPoints.size());
	for (int i = 0; i < faceUvIs.size(); i++) {
		int uv = faceUvIs[i];
		int ci = _faceIndices[i];
		double U = unorderedUvs[uv * 2];
		double V = unorderedUvs[uv * 2 + 1];
		_uvs[ci].set(U, V, 0);
	}
}


Nuke2TensorFlow::Nuke2TensorFlow(int resolution)
{
	_resolution = resolution;
	_detector = get_frontal_face_detector();
	_destPoints = {
		{ 0, 0 },
		{ 0, _resolution - 1},
		{ _resolution - 1, 0},
	};
	_planeHeight = 0;
	_points.resize(data.defaultPoints().size());
}


unsigned char Nuke2TensorFlow::linear2srgb(float c)
{
	if (c > 0.0031308)
		c = 1.055 * (std::pow(c, (1.0 / 2.4))) - 0.055;
	else
		c = 12.92 * c;
	return clamp((int)(c * 255), 0, 255);
}


void Nuke2TensorFlow::extractDataFromTFLite(float* in)
{
	float frac = -1 / _pointTransform.get_m()(0, 0);
	_pointTransform = inv(_pointTransform);

	parallel_for(size_t(0), data.defaultPoints().size(), [&](size_t i) {
		int flatI = i * 3;
		float x = in[flatI];
		float y = in[flatI + 1];
		float z = (in[flatI + 2] - _resolution / 2) * frac;

		vector<double, 2> v = {x, y};
		v = _pointTransform({x, y});
		x = v(0);
		y = _planeHeight - 1 - v(1);

		_points.at(i).set(x, y, z);
	});
}


void Nuke2TensorFlow::plane2img(const DD::Image::ImagePlane& plane,
						matrix<rgb_pixel>& img)
{
	auto ipBox = plane.bounds();
	int x = ipBox.x(), t = ipBox.t(), r = ipBox.r(),
	    y = ipBox.y(), h = ipBox.h(), w = ipBox.w();
	_planeHeight = (float)h;

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


bool Nuke2TensorFlow::extractFacePixels(const matrix<rgb_pixel>& inImg,
					int l, int r, int t, int b,
					bool detected, float* out)
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

	img2tflite(outImg, out);
	return true;
}


void Nuke2TensorFlow::img2tflite(const matrix<rgb_pixel>& img, float* out)
{
	parallel_for(size_t(0), _resolution, [&](size_t y) {
		for (int x = 0; x < _resolution; ++x) {
			int flatI = (x * _resolution + y) * 3;
			out[flatI] = (float)img(x, y).red / 255.;
			out[flatI + 1] = (float)img(x, y).green / 255.;
			out[flatI + 2] = (float)img(x, y).blue / 255.;
		}
	});
}


bool Nuke2TensorFlow::imagePlane2tflite(const DD::Image::ImagePlane& plane,
					const DD::Image::Box& userBBox,
					bool useDetector, float* out)
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
		auto dets = _detector(imgP);

		if (dets.size() < 1) {
			std::cout << "No faces found.\n";
			return false;
		}
		auto detBBox = pyr.rect_down(dets.at(0), _upsample);

		return extractFacePixels(img, detBBox.left(), detBBox.right(),
				detBBox.top(), detBBox.bottom(), true, out);
	}
	auto ipBBox = plane.bounds();

	//std::cout << "Using the provided bounding box for inference.\n";
	return extractFacePixels(img, userBBox.x(), userBBox.r(),
				ipBBox.h() - userBBox.t(),
				ipBBox.h() - userBBox.y(), false, out);
}
