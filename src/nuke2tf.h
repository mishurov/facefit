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

#ifndef NUKE2TF_H_
#define NUKE2TF_H_

#include <DDImage/ImagePlane.h>
#include <DDImage/GeoInfo.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/dnn.h>
#include <tensorflow/core/framework/tensor.h>
#include <string>


typedef std::vector<dlib::vector<int,2>> WarpPointList;
typedef std::vector<DD::Image::Vector3> UVList;


/* The CNN face detector */
template <long num_filters, typename SUBNET> using con5d =
	dlib::con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5 =
	dlib::con<num_filters,5,5,1,1,SUBNET>;

template <typename SUBNET> using downsampler =
	dlib::relu<dlib::affine<con5d<32, dlib::relu<dlib::affine<
		con5d<32, dlib::relu<dlib::affine<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5 =
	dlib::relu<dlib::affine<con5<45,SUBNET>>>;

using net_type = dlib::loss_mmod<dlib::con<1,9,9,1,1,rcon5<rcon5<
	rcon5<downsampler<dlib::input_rgb_image_pyramid<
		dlib::pyramid_down<6>>>>>>>>;

class Nuke2TensorFlow {
public:
	Nuke2TensorFlow(int resolution);
	tensorflow::Tensor imagePlane2Tensor(const DD::Image::ImagePlane& plane,
						const DD::Image::Box& userBBox,
						bool useDetector);
	void extractDataFromTensor(tensorflow::Tensor&);
	const DD::Image::PointList& points() { return _points; }

	struct StaticData
	{
	private:
		void readIndices(const std::string& path,
					std::vector<int>& indices);
		DD::Image::PointList _defaultPoints;
		std::vector<int> _faceIndices;
		std::vector<int> _kptIndices;
		std::vector<int> _triIndices;
		std::map<int,int> _face2all;
		UVList _uvs;
		std::set<int> _endList;
	public:
		net_type net;
		StaticData(const std::string& DetectorModelPath,
			const std::string& trianglesPath,
			const std::string& faceIndicesPath,
			const std::string& kptIndicesPath,
			int resolution);
		const DD::Image::PointList& defaultPoints() {
			return _defaultPoints;
		}
		const std::vector<int>& faceIndices() { return _faceIndices; }
		const std::map<int,int>& face2all() { return _face2all; }
		const std::vector<int>& kptIndices() { return _kptIndices; }
		const std::vector<int>& triIndices() { return _triIndices; }
		const UVList& uvs() { return _uvs; }
		const std::set<int>& endList() { return _endList; }
	};
	static StaticData data;

private:
	DD::Image::PointList _points;
	// An HOG face detector.
	// it's faster and less accurate, it can be enabled in the code
	dlib::frontal_face_detector _detector;
	dlib::point_transform_affine _pointTransform;
	float _planeHeight;
	float _planeWidth;
	int _resolution;
	WarpPointList _destPoints;
	// number of upsamples for more precise detection
	unsigned int _upsample = 1;
	
	void plane2img(const DD::Image::ImagePlane& plane,
			dlib::matrix<dlib::rgb_pixel>& img);
	unsigned char linear2srgb(float c);
	tensorflow::Tensor extractFaceTensor(
		const dlib::matrix<dlib::rgb_pixel>& inImg,
		int l, int r, int t, int b, bool detected);
	tensorflow::Tensor img2Tensor(
		const dlib::matrix<dlib::rgb_pixel>& img);
};


#endif // NUKE2TF_H_
