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
#include <string>


typedef std::vector<dlib::vector<int,2>> WarpPointList;
typedef std::vector<DD::Image::Vector3> UVList;


class Nuke2TensorFlow {
public:
	struct StaticData
	{
		private:
			DD::Image::PointList _defaultPoints;
			std::vector<int> _faceIndices;
			UVList _uvs;
		public:
			StaticData(const std::string& faceObjPath);
			const DD::Image::PointList& defaultPoints() {
				return _defaultPoints;
			}
			const std::vector<int>& faceIndices() {
				return _faceIndices;
			}
			const UVList& uvs() { return _uvs; }
	};
	static StaticData data;

	Nuke2TensorFlow(int resolution);
	bool imagePlane2tflite(const DD::Image::ImagePlane& plane,
						const DD::Image::Box& userBBox,
						bool useDetector, float* out);
	void extractDataFromTFLite(float* in);
	const DD::Image::PointList& points() { return _points; }

private:
	DD::Image::PointList _points;
	dlib::frontal_face_detector _detector;
	dlib::point_transform_affine _pointTransform;
	float _planeHeight;
	int _resolution;
	WarpPointList _destPoints;
	// number of upsamples for more precise detection
	unsigned int _upsample = 1;
	
	void plane2img(const DD::Image::ImagePlane& plane,
			dlib::matrix<dlib::rgb_pixel>& img);
	unsigned char linear2srgb(float c);
	bool extractFacePixels(
		const dlib::matrix<dlib::rgb_pixel>& inImg,
		int l, int r, int t, int b, bool detected, float* out);
	void img2tflite(
		const dlib::matrix<dlib::rgb_pixel>& img, float* out);
};


#endif // NUKE2TF_H_
